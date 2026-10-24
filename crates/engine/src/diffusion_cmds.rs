//! Generative diffusion integration: clean-room communication with stock ComfyUI servers.
//!
//! Exposes catalog discovery, image generation, history tracking, previewing, and color-managed
//! application to documents and layers with undo history.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_diffusion::{Mode, Request};
use photocraft_doc::{DocId, Document, Layer, LayerContent, LayerMask};
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

/// Maximum number of completed generations retained in session memory.
pub const MAX_HISTORY: usize = 16;
/// Maximum aggregate memory for stored generation PNG bytes (256 MiB).
pub const MAX_MEMORY_BYTES: usize = 256 * 1024 * 1024;

/// One completed diffusion generation stored in session memory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationEntry {
    pub job: u64,
    pub mode: Mode,
    pub prompt: String,
    pub doc_id: Option<DocId>,
    pub doc_rev: Option<u64>,
    #[serde(skip)]
    pub mask: Option<Surface>,
    pub images: Vec<Vec<u8>>,
    pub created_ms: f64,
    pub bytes: usize,
}

/// Bounded diffusion generation history for the session.
#[derive(Clone, Debug, Default)]
pub struct State {
    next_id: u64,
    entries: Vec<GenerationEntry>,
    total_bytes: usize,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate the next monotonic diffusion generation ID.
    pub fn alloc_id(&mut self) -> u64 {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.next_id
    }

    /// Store a completed generation, evicting the oldest entries to enforce history and memory bounds.
    pub fn push(&mut self, mut entry: GenerationEntry) {
        let entry_bytes: usize = entry.images.iter().map(|img| img.len()).sum();
        entry.bytes = entry_bytes;

        while self.entries.len() >= MAX_HISTORY || (!self.entries.is_empty() && self.total_bytes.saturating_add(entry_bytes) > MAX_MEMORY_BYTES) {
            if let Some(old) = self.entries.first() {
                self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
                self.entries.remove(0);
            } else {
                break;
            }
        }

        self.total_bytes = self.total_bytes.saturating_add(entry_bytes);
        self.entries.push(entry);
    }

    /// Look up PNG bytes for a completed job and output image index.
    pub fn preview(&self, job: u64, index: usize) -> Option<&[u8]> {
        self.entries.iter().find(|e| e.job == job).and_then(|e| e.images.get(index)).map(Vec::as_slice)
    }

    /// Retrieve an entry by job ID.
    pub fn get(&self, job: u64) -> Option<&GenerationEntry> {
        self.entries.iter().find(|e| e.job == job)
    }

    /// Remove a specific job entry from history. Returns true if found and removed.
    pub fn remove(&mut self, job: u64) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.job == job) {
            let removed = self.entries.remove(pos);
            self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
            true
        } else {
            false
        }
    }

    /// Clear all stored history, releasing image bytes. Returns the count of cleared entries.
    pub fn clear(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        self.total_bytes = 0;
        count
    }

    /// All stored history entries, oldest first.
    pub fn entries(&self) -> &[GenerationEntry] {
        &self.entries
    }

    /// Total memory currently consumed by stored images in bytes.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

/// Retrieve PNG thumbnail or preview bytes for a generation job.
pub fn preview(session: &Session, job: u64, index: usize) -> Option<&[u8]> {
    session.diffusion.preview(job, index)
}

fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

/// Export a document composite converted to standard 8-bit sRGB PNG.
pub fn export_doc_to_srgb_png(doc: &Document) -> Result<Vec<u8>> {
    let mut d = doc.clone();
    if d.depth != SampleType::U8 {
        crate::image_cmds::change_depth(&mut d, SampleType::U8);
    }
    let srgb = photocraft_cms::Builtin::Srgb.profile();
    if d.mode != ColorMode::Rgb || d.icc_profile.as_deref().map(Vec::as_slice) != Some(srgb.to_bytes().as_slice()) {
        crate::color_cmds::convert_document(&mut d, srgb, photocraft_cms::Intent::RelativeColorimetric, true)?;
    }
    let res = photocraft_io::export(&d, "png", &Default::default()).map_err(|e| EngineError::Other(format!("document export for diffusion failed: {e}")))?;
    Ok(res.bytes)
}

/// Export an active selection mask as an opaque RGB PNG (white = editable, black = preserved).
pub fn export_selection_mask_png(doc: &Document) -> Result<Vec<u8>> {
    let sel = doc.selection.as_ref().ok_or_else(|| EngineError::Other("diffusion inpaint requires an active selection".into()))?;
    let sel_gray = sel.try_convert(PixelFormat::GRAY8)?;
    let bounds = doc.bounds();
    let gray_bytes = sel_gray.to_interleaved(bounds);

    let mut rgb = Vec::with_capacity(gray_bytes.len().saturating_mul(3));
    for &g in &gray_bytes {
        rgb.push(g);
        rgb.push(g);
        rgb.push(g);
    }

    let image = photocraft_codecs::Image::from_u8(doc.size.width, doc.size.height, photocraft_codecs::ChannelLayout::Rgb, rgb)
        .map_err(|e| EngineError::Other(format!("mask image conversion failed: {e}")))?;
    photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default())
        .map_err(|e| EngineError::Other(format!("mask png encoding failed: {e}")))
}

/// Apply a generated image from diffusion history back into the session.
pub fn apply_entry(s: &mut Session, job: u64, index: usize, new_doc_opt: Option<bool>) -> Result<Value> {
    let entry = s.diffusion.get(job).ok_or_else(|| EngineError::Other(format!("diffusion job {job} not found in history")))?;
    let png_bytes = entry
        .images
        .get(index)
        .ok_or_else(|| EngineError::Other(format!("image index {index} not found for diffusion job {job} (available: {})", entry.images.len())))?
        .clone();
    let captured_mask = entry.mask.clone();
    let original_doc = entry.doc_id;
    let mode = entry.mode;

    let should_create_new = new_doc_opt.unwrap_or(mode == Mode::Upscale || s.active().is_none());

    if should_create_new {
        let mut imported = crate::file_cmds::import(&format!("diffusion-{job}-{index}.png"), &png_bytes)?;
        imported.name = format!("Diffusion-{job}-{index}");
        let doc_id = imported.id;
        let idx = s.add_document(imported, None);
        Ok(json!({
            "job": job,
            "index": index,
            "newDocument": true,
            "documentId": doc_id.0,
            "documentIndex": idx,
        }))
    } else {
        let active_st = s.active().ok_or(EngineError::NoDocument)?;
        let active_doc_id = active_st.doc.id;

        if let Some(orig) = original_doc
            && orig != active_doc_id
        {
            return Err(EngineError::Other(format!(
                "diffusion job {job} was generated from document {}, but active document is {}; pass `\"newDocument\": true` to open in a new document",
                orig.0, active_doc_id.0
            )));
        }

        let mut imported = crate::file_cmds::import(&format!("diffusion-{job}-{index}.png"), &png_bytes)?;
        let target_depth = active_st.doc.depth;
        let target_mode = active_st.doc.mode;
        let target_profile = crate::color_cmds::document_profile(&active_st.doc);
        let target_format = active_st.doc.pixel_format();

        if imported.depth != target_depth {
            crate::image_cmds::change_depth(&mut imported, target_depth);
        }
        if imported.mode != target_mode || imported.icc_profile != active_st.doc.icc_profile {
            crate::color_cmds::convert_document(&mut imported, &target_profile, photocraft_cms::Intent::RelativeColorimetric, true)?;
        }

        let surf = crate::file_cmds::flattened(&imported, target_format);

        let layer_id = s.edit("Diffusion Apply", |doc, active_layer| {
            let mut layer = Layer::new(format!("Diffusion {job}.{index}"), LayerContent::Raster(surf));
            if let Some(mask_surf) = captured_mask {
                let mask = LayerMask { surface: mask_surf, ..LayerMask::reveal_all() };
                layer.mask = Some(crate::commands::mask_at_depth(mask, doc.depth)?);
            }
            let id = doc.insert_above(*active_layer, layer);
            *active_layer = Some(id);
            Ok(id)
        })?;

        Ok(json!({
            "job": job,
            "index": index,
            "newDocument": false,
            "documentId": active_doc_id.0,
            "layerId": layer_id.0,
        }))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_diffusion_generation(
    ctx: &crate::jobs::JobCtx,
    endpoint: &str,
    request: &Request,
    input_png: Option<&[u8]>,
    mask_png: Option<&[u8]>,
    timeout_secs: u64,
) -> Result<Vec<Vec<u8>>> {
    ctx.progress(0.05, "Connecting to diffusion server");
    let client = photocraft_diffusion::Client::new(endpoint).map_err(|e| EngineError::Other(format!("diffusion endpoint error: {e}")))?;
    ctx.check()?;

    let mut input_remote = None;
    if let Some(bytes) = input_png {
        ctx.progress(0.10, "Uploading document pixels");
        let name = format!("photocraft_in_{}.png", now_ms() as u64);
        let remote = client.upload_png(bytes, &name).map_err(|e| EngineError::Other(format!("failed to upload input image: {e}")))?;
        input_remote = Some(remote);
        ctx.check()?;
    }

    let mut mask_remote = None;
    if let Some(bytes) = mask_png {
        ctx.progress(0.15, "Uploading selection mask");
        let name = format!("photocraft_mask_{}.png", now_ms() as u64);
        let remote = client.upload_png(bytes, &name).map_err(|e| EngineError::Other(format!("failed to upload mask image: {e}")))?;
        mask_remote = Some(remote);
        ctx.check()?;
    }

    ctx.progress(0.20, "Submitting diffusion workflow");
    let prompt_id = client
        .submit(request, input_remote.as_deref(), mask_remote.as_deref())
        .map_err(|e| EngineError::Other(format!("diffusion workflow submission failed: {e}")))?;
    ctx.check()?;

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(timeout_secs);
    let poll_interval = std::time::Duration::from_millis(500);

    let outputs = loop {
        if ctx.cancelled() {
            let _ = client.cancel(&prompt_id);
            return Err(EngineError::Cancelled);
        }
        if start.elapsed() > timeout {
            let _ = client.cancel(&prompt_id);
            return Err(EngineError::Other(format!("diffusion generation timed out after {timeout_secs}s")));
        }

        let elapsed = start.elapsed().as_secs_f32();
        let timeout_f32 = timeout_secs as f32;
        let progress = 0.20 + 0.60 * (elapsed / timeout_f32).clamp(0.0, 0.95);
        ctx.progress(progress, "Generating image");

        std::thread::sleep(poll_interval);

        match client.history(&prompt_id) {
            Ok(Some(outputs)) => break outputs,
            Ok(None) => continue,
            Err(e) => {
                let _ = client.cancel(&prompt_id);
                return Err(EngineError::Other(format!("diffusion generation failed: {e}")));
            }
        }
    };

    if outputs.is_empty() {
        return Err(EngineError::Other("diffusion completed with no image outputs".into()));
    }

    ctx.progress(0.85, "Downloading generated images");
    let mut images = Vec::new();
    let mut total_downloaded: usize = 0;
    for (idx, out) in outputs.iter().enumerate() {
        ctx.check()?;
        let fraction = 0.85 + 0.14 * ((idx as f32) / (outputs.len() as f32));
        ctx.progress(fraction, &format!("Downloading image {}/{}", idx.saturating_add(1), outputs.len()));
        let img_bytes = client.download(out).map_err(|e| EngineError::Other(format!("failed to download generated image: {e}")))?;
        total_downloaded = total_downloaded.saturating_add(img_bytes.len());
        if total_downloaded > MAX_MEMORY_BYTES {
            return Err(EngineError::Other("downloaded images exceed 256 MiB memory limit".into()));
        }
        images.push(img_bytes);
    }

    ctx.progress(1.0, "Complete");
    Ok(images)
}

fn extract_request(p: &Value) -> Result<Request> {
    let req_val = if let Some(r) = p.get("request") {
        r.clone()
    } else {
        let mut map = serde_json::Map::new();
        for key in &[
            "mode",
            "checkpoint",
            "prompt",
            "negative",
            "width",
            "height",
            "steps",
            "cfg",
            "seed",
            "denoise",
            "batch",
            "sampler",
            "scheduler",
            "loras",
            "upscaler",
            "scale",
            "workflow",
        ] {
            if let Some(v) = p.get(*key) {
                map.insert((*key).to_string(), v.clone());
            }
        }
        Value::Object(map)
    };

    serde_json::from_value(req_val)
        .map_err(|e| EngineError::BadParams { cmd: "ai.diffusion.generate".into(), msg: format!("invalid diffusion request parameters: {e}") })
}

/// Commands for the AI diffusion feature.
pub fn specs() -> Vec<crate::commands::CommandSpec> {
    use crate::commands::CommandSpec;

    vec![
        CommandSpec {
            id: "ai.diffusion.catalog",
            label: "Diffusion Catalog",
            menu: &[],
            shortcut: None,
            params: r##"{"endpoint":str?}"##,
            enabled: |_| Ok(()),
            run: |s, p| {
                let endpoint = p.get("endpoint").and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("http://127.0.0.1:8188").to_string();
                crate::jobs::run(
                    s,
                    "Diffusion Catalog",
                    false,
                    {
                        let endpoint = endpoint.clone();
                        move |ctx| {
                            ctx.check()?;
                            #[cfg(target_arch = "wasm32")]
                            {
                                let _ = &endpoint;
                                Err(EngineError::Other("diffusion network operations are not supported on WebAssembly".into()))
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                let client =
                                    photocraft_diffusion::Client::new(&endpoint).map_err(|e| EngineError::Other(format!("diffusion endpoint error: {e}")))?;
                                ctx.check()?;
                                let catalog = client.catalog().map_err(|e| EngineError::Other(format!("failed to fetch diffusion catalog: {e}")))?;
                                Ok(catalog)
                            }
                        }
                    },
                    move |_s, catalog: Value| Ok(json!({"catalog": catalog, "endpoint": endpoint})),
                )
            },
            journal: false,
        },
        CommandSpec {
            id: "ai.diffusion.generate",
            label: "Diffusion Generate",
            menu: &[],
            shortcut: None,
            params: r##"{"endpoint":str?,"consent":bool,"apply":bool=false,"timeoutSecs":u64=120,"request":object?}"##,
            enabled: |_| Ok(()),
            run: |s, p| {
                let endpoint = p.get("endpoint").and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("http://127.0.0.1:8188").to_string();
                let consent = p.get("consent").and_then(Value::as_bool).unwrap_or(false);
                let apply_flag = p.get("apply").and_then(Value::as_bool).unwrap_or(false);
                let timeout_secs = p.get("timeoutSecs").or_else(|| p.get("timeout")).and_then(Value::as_u64).unwrap_or(120).clamp(5, 600);

                let mut request = extract_request(p)?;

                let (doc_id, doc_rev, captured_mask, input_png_bytes, mask_png_bytes) = match request.mode {
                    Mode::Generate => {
                        let (id, rev) = match s.active() {
                            Some(st) => (Some(st.doc.id), Some(st.revision)),
                            None => (None, None),
                        };
                        (id, rev, None, None, None)
                    }
                    Mode::Refine => {
                        let st = s.active().ok_or_else(|| EngineError::Other("diffusion refine requires an active document".into()))?;
                        let doc = &st.doc;
                        if request.width == 512
                            && request.height == 512
                            && doc.size.width >= 64
                            && doc.size.height >= 64
                            && doc.size.width.is_multiple_of(8)
                            && doc.size.height.is_multiple_of(8)
                            && doc.size.width <= 8192
                            && doc.size.height <= 8192
                        {
                            request.width = doc.size.width;
                            request.height = doc.size.height;
                        }
                        let input_bytes = export_doc_to_srgb_png(doc)?;
                        (Some(doc.id), Some(st.revision), None, Some(input_bytes), None)
                    }
                    Mode::Inpaint => {
                        let st = s.active().ok_or_else(|| EngineError::Other("diffusion inpaint requires an active document".into()))?;
                        let doc = &st.doc;
                        if doc.selection.is_none() {
                            return Err(EngineError::Other("diffusion inpaint requires an active selection".into()));
                        }
                        if request.width == 512
                            && request.height == 512
                            && doc.size.width >= 64
                            && doc.size.height >= 64
                            && doc.size.width.is_multiple_of(8)
                            && doc.size.height.is_multiple_of(8)
                            && doc.size.width <= 8192
                            && doc.size.height <= 8192
                        {
                            request.width = doc.size.width;
                            request.height = doc.size.height;
                        }
                        let input_bytes = export_doc_to_srgb_png(doc)?;
                        let mask_bytes = export_selection_mask_png(doc)?;
                        let mask_surf = doc.selection.clone();
                        (Some(doc.id), Some(st.revision), mask_surf, Some(input_bytes), Some(mask_bytes))
                    }
                    Mode::Upscale => {
                        let st = s.active().ok_or_else(|| EngineError::Other("diffusion upscale requires an active document".into()))?;
                        let doc = &st.doc;
                        let input_bytes = export_doc_to_srgb_png(doc)?;
                        (Some(doc.id), Some(st.revision), None, Some(input_bytes), None)
                    }
                    Mode::Custom => {
                        if let Some(st) = s.active() {
                            let doc = &st.doc;
                            let input_bytes = export_doc_to_srgb_png(doc).ok();
                            let mask_bytes = if doc.selection.is_some() { export_selection_mask_png(doc).ok() } else { None };
                            let mask_surf = doc.selection.clone();
                            (Some(doc.id), Some(st.revision), mask_surf, input_bytes, mask_bytes)
                        } else {
                            (None, None, None, None, None)
                        }
                    }
                };

                request
                    .validate()
                    .map_err(|e| EngineError::BadParams { cmd: "ai.diffusion.generate".into(), msg: format!("diffusion request validation failed: {e}") })?;

                if (input_png_bytes.is_some() || mask_png_bytes.is_some()) && !consent {
                    return Err(EngineError::Other(
                        "diffusion generation requires explicit consent (`consent: true`) to upload document pixels to the server".into(),
                    ));
                }

                let job_id = s.diffusion.alloc_id();

                crate::jobs::run(
                    s,
                    "Diffusion Generate",
                    false,
                    {
                        let request = request.clone();
                        let endpoint = endpoint.clone();
                        move |ctx| {
                            ctx.check()?;
                            #[cfg(target_arch = "wasm32")]
                            {
                                let _ = (&endpoint, &request, timeout_secs);
                                Err(EngineError::Other("diffusion network generation is not supported on WebAssembly".into()))
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                run_diffusion_generation(ctx, &endpoint, &request, input_png_bytes.as_deref(), mask_png_bytes.as_deref(), timeout_secs)
                            }
                        }
                    },
                    move |s, images: Vec<Vec<u8>>| {
                        let entry = GenerationEntry {
                            job: job_id,
                            mode: request.mode,
                            prompt: request.prompt.clone(),
                            doc_id,
                            doc_rev,
                            mask: captured_mask,
                            images,
                            created_ms: now_ms(),
                            bytes: 0,
                        };
                        s.diffusion.push(entry);

                        let mut result = json!({
                            "job": job_id,
                            "images": s.diffusion.get(job_id).map(|e| e.images.len()).unwrap_or(0),
                            "applied": false,
                        });

                        if apply_flag && let Ok(applied_val) = apply_entry(s, job_id, 0, None) {
                            result["applied"] = Value::Bool(true);
                            result["applyResult"] = applied_val;
                        }

                        Ok(result)
                    },
                )
            },
            journal: false,
        },
        CommandSpec {
            id: "ai.diffusion.apply",
            label: "Diffusion Apply",
            menu: &[],
            shortcut: None,
            params: r##"{"job":u64,"index":usize=0,"newDocument":bool?}"##,
            enabled: |_| Ok(()),
            run: |s, p| {
                let job = p
                    .get("job")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| EngineError::BadParams { cmd: "ai.diffusion.apply".into(), msg: "`job` is required".into() })?;
                let index = p.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let new_doc_opt = p.get("newDocument").and_then(Value::as_bool);
                apply_entry(s, job, index, new_doc_opt)
            },
            journal: true,
        },
        CommandSpec {
            id: "ai.diffusion.history",
            label: "Diffusion History",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: |_| Ok(()),
            run: |s, _| {
                let list: Vec<Value> = s
                    .diffusion
                    .entries()
                    .iter()
                    .map(|e| {
                        json!({
                            "job": e.job,
                            "mode": e.mode,
                            "prompt": e.prompt,
                            "images": e.images.len(),
                            "bytes": e.bytes,
                            "createdMs": e.created_ms,
                            "docId": e.doc_id.map(|d| d.0),
                            "docRev": e.doc_rev,
                        })
                    })
                    .collect();
                Ok(json!({
                    "history": list,
                    "totalBytes": s.diffusion.total_bytes(),
                    "maxHistory": MAX_HISTORY,
                    "maxBytes": MAX_MEMORY_BYTES,
                }))
            },
            journal: false,
        },
        CommandSpec {
            id: "ai.diffusion.discard",
            label: "Diffusion Discard",
            menu: &[],
            shortcut: None,
            params: r##"{"job":u64?}"##,
            enabled: |_| Ok(()),
            run: |s, p| match p.get("job") {
                Some(v) if !v.is_null() => {
                    let id = v
                        .as_u64()
                        .ok_or_else(|| EngineError::BadParams { cmd: "ai.diffusion.discard".into(), msg: "`job` must be a job id integer".into() })?;
                    let removed = s.diffusion.remove(id);
                    Ok(json!({"discarded": if removed { 1 } else { 0 }}))
                }
                _ => {
                    let cleared = s.diffusion.clear();
                    Ok(json!({"discarded": cleared}))
                }
            },
            journal: false,
        },
        CommandSpec {
            id: "ai.diffusion.cancel",
            label: "Diffusion Cancel",
            menu: &[],
            shortcut: None,
            params: r##"{"job":u64?}"##,
            enabled: |_| Ok(()),
            run: |s, p| {
                let id_opt = p.get("job").and_then(Value::as_u64);
                let mut cancelled_count = 0;
                if let Some(id) = id_opt {
                    if s.cancel_job(crate::jobs::JobId(id)) {
                        cancelled_count += 1;
                    }
                } else {
                    for j in s.jobs() {
                        if j.command == "ai.diffusion.generate" && s.cancel_job(j.id) {
                            cancelled_count += 1;
                        }
                    }
                }
                Ok(json!({"cancelled": cancelled_count}))
            },
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use photocraft_doc::Size;

    use super::*;

    fn create_test_doc(w: u32, h: u32) -> Document {
        let mut doc = Document::new("Test", Size { width: w, height: h }, ColorMode::Rgb, SampleType::U8);
        let surf = Surface::with_default(PixelFormat::RGBA8, &[0.5, 0.5, 0.5, 1.0]);
        let l = Layer::new("Background", LayerContent::Raster(surf));
        doc.layers = vec![l];
        doc
    }

    #[test]
    fn state_history_and_memory_bounds() {
        let mut st = State::new();
        assert_eq!(st.entries().len(), 0);
        assert_eq!(st.total_bytes(), 0);

        // Add 20 entries of 1 MiB each. Should clamp to MAX_HISTORY (16).
        for i in 1..=20 {
            let entry = GenerationEntry {
                job: i,
                mode: Mode::Generate,
                prompt: format!("prompt {i}"),
                doc_id: None,
                doc_rev: None,
                mask: None,
                images: vec![vec![0u8; 1024 * 1024]],
                created_ms: 100.0,
                bytes: 0,
            };
            st.push(entry);
        }

        assert_eq!(st.entries().len(), MAX_HISTORY);
        assert_eq!(st.entries().first().unwrap().job, 5);
        assert_eq!(st.entries().last().unwrap().job, 20);
        assert_eq!(st.total_bytes(), 16 * 1024 * 1024);

        // Preview retrieval
        assert!(st.preview(20, 0).is_some());
        assert_eq!(st.preview(20, 0).unwrap().len(), 1024 * 1024);
        assert!(st.preview(20, 1).is_none());
        assert!(st.preview(999, 0).is_none());

        // Discard one
        assert!(st.remove(20));
        assert_eq!(st.entries().len(), 15);
        assert_eq!(st.total_bytes(), 15 * 1024 * 1024);

        // Clear all
        let cleared = st.clear();
        assert_eq!(cleared, 15);
        assert_eq!(st.entries().len(), 0);
        assert_eq!(st.total_bytes(), 0);
    }

    #[test]
    fn srgb_export_and_mask_export() {
        let mut doc = create_test_doc(64, 64);
        let png = export_doc_to_srgb_png(&doc).unwrap();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

        // No selection mask -> error
        assert!(export_selection_mask_png(&doc).is_err());

        // Add selection mask
        doc.selection = Some(Surface::with_default(PixelFormat::GRAY8, &[1.0]));
        let mask_png = export_selection_mask_png(&doc).unwrap();
        assert!(mask_png.starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn generate_validation_and_consent() {
        let mut s = Session::new();

        // 1. Invalid dimensions (not multiple of 8)
        let err = s
            .execute(
                "ai.diffusion.generate",
                json!({
                    "endpoint": "http://127.0.0.1:8188",
                    "checkpoint": "model.safetensors",
                    "consent": true,
                    "mode": "generate",
                    "width": 511,
                    "height": 512,
                }),
            )
            .unwrap_err();
        assert!(err.to_string().contains("multiples of 8"), "{err}");

        // 2. Refine requires active document
        let err = s
            .execute(
                "ai.diffusion.generate",
                json!({
                    "endpoint": "http://127.0.0.1:8188",
                    "checkpoint": "model.safetensors",
                    "consent": true,
                    "mode": "refine",
                    "width": 512,
                    "height": 512,
                }),
            )
            .unwrap_err();
        assert!(err.to_string().contains("refine requires an active document"), "{err}");

        // 3. Open doc without selection, inpaint requires selection
        let doc = create_test_doc(64, 64);
        s.add_document(doc, None);
        let err = s
            .execute(
                "ai.diffusion.generate",
                json!({
                    "endpoint": "http://127.0.0.1:8188",
                    "checkpoint": "model.safetensors",
                    "consent": true,
                    "mode": "inpaint",
                    "width": 64,
                    "height": 64,
                }),
            )
            .unwrap_err();
        assert!(err.to_string().contains("inpaint requires an active selection"), "{err}");

        // 4. Refine with document but without consent -> rejected
        let err = s
            .execute(
                "ai.diffusion.generate",
                json!({
                    "endpoint": "http://127.0.0.1:8188",
                    "checkpoint": "model.safetensors",
                    "consent": false,
                    "mode": "refine",
                    "width": 64,
                    "height": 64,
                }),
            )
            .unwrap_err();
        assert!(err.to_string().contains("consent"), "{err}");
    }

    #[test]
    fn apply_and_history_lifecycle() {
        let mut s = Session::new();
        let doc = create_test_doc(64, 64);
        s.add_document(doc, None);

        // Generate a 1x1 test PNG
        let img = photocraft_codecs::Image::from_u8(1, 1, photocraft_codecs::ChannelLayout::Rgb, vec![255, 0, 0]).unwrap();
        let fake_png = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();

        // Push test entry into session diffusion state
        let entry = GenerationEntry {
            job: 42,
            mode: Mode::Generate,
            prompt: "test cat".into(),
            doc_id: s.active().map(|st| st.doc.id),
            doc_rev: s.active().map(|st| st.revision),
            mask: None,
            images: vec![fake_png],
            created_ms: 100.0,
            bytes: 0,
        };
        s.diffusion.push(entry);

        // Verify history
        let hist = s.execute("ai.diffusion.history", json!({})).unwrap();
        assert_eq!(hist["history"].as_array().unwrap().len(), 1);
        assert_eq!(hist["history"][0]["job"], 42);

        // Verify preview function
        assert!(preview(&s, 42, 0).is_some());
        assert_eq!(preview(&s, 42, 1), None);

        // Apply to current document as layer
        let apply_res = s.execute("ai.diffusion.apply", json!({"job": 42, "index": 0})).unwrap();
        assert_eq!(apply_res["newDocument"], false);
        assert_eq!(s.active().unwrap().doc.layers.len(), 2);
        assert_eq!(s.active().unwrap().doc.layers.last().unwrap().name, "Diffusion 42.0");

        // Apply as new document
        let apply_new = s.execute("ai.diffusion.apply", json!({"job": 42, "index": 0, "newDocument": true})).unwrap();
        assert_eq!(apply_new["newDocument"], true);
        assert_eq!(s.documents().len(), 2);

        // Discard
        let discard_res = s.execute("ai.diffusion.discard", json!({"job": 42})).unwrap();
        assert_eq!(discard_res["discarded"], 1);
        assert!(s.diffusion.entries().is_empty());
    }
}
