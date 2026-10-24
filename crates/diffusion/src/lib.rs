//! Clean-room stock ComfyUI API workflow construction and bounded native HTTP access.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[cfg(not(target_arch = "wasm32"))]
mod client;
#[cfg(not(target_arch = "wasm32"))]
pub use client::Client;

/// Maximum encoded JSON graph or response size.
pub const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
/// Maximum encoded PNG upload or downloaded image size.
pub const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_NODES: usize = 256;
const MAX_DEPTH: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid diffusion request: {0}")]
    Invalid(String),
    #[error("diffusion server protocol error: {0}")]
    Protocol(String),
    #[error("diffusion job failed: {0}")]
    Job(String),
    #[error("diffusion JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("diffusion HTTP: {0}")]
    Http(#[from] ureq::Error),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Generate,
    Refine,
    Inpaint,
    Upscale,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lora {
    pub name: String,
    pub strength: f64,
}

/// Parameters for classic checkpoint/CLIP/VAE stock-node graphs, not every model architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub mode: Mode,
    pub checkpoint: String,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f64,
    pub seed: u64,
    pub denoise: f64,
    pub batch: u32,
    pub sampler: String,
    pub scheduler: String,
    pub loras: Vec<Lora>,
    pub upscaler: String,
    pub scale: f64,
    pub workflow: Option<Value>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            mode: Mode::Generate,
            checkpoint: String::new(),
            prompt: String::new(),
            negative: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg: 7.0,
            seed: 0,
            denoise: 0.5,
            batch: 1,
            sampler: "euler".into(),
            scheduler: "normal".into(),
            loras: Vec::new(),
            upscaler: String::new(),
            scale: 2.0,
            workflow: None,
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

fn bounded_text(value: &str, label: &str, max: usize, required: bool) -> Result<(), Error> {
    if value.len() > max || value.contains('\0') || (required && value.trim().is_empty()) {
        return Err(invalid(format!("{label} must be {}1..={max} bytes without NUL", if required { "" } else { "0 or " })));
    }
    Ok(())
}

fn finite_range(value: f64, label: &str, low: f64, high: f64) -> Result<(), Error> {
    if !value.is_finite() || !(low..=high).contains(&value) {
        return Err(invalid(format!("{label} must be finite and within {low}..={high}")));
    }
    Ok(())
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        if !(64..=8192).contains(&self.width) || !(64..=8192).contains(&self.height) || !self.width.is_multiple_of(8) || !self.height.is_multiple_of(8) {
            return Err(invalid("dimensions must be multiples of 8 within 64..=8192"));
        }
        if !(1..=8).contains(&self.batch) || u64::from(self.width) * u64::from(self.height) * u64::from(self.batch) > 33_554_432 {
            return Err(invalid("batch must be 1..=8 and total pixels at most 32 MiPixels"));
        }
        if !(1..=150).contains(&self.steps) {
            return Err(invalid("steps must be 1..=150"));
        }
        // JSON numbers cross JavaScript servers too: reject integers that lose precision there.
        if self.seed > 9_007_199_254_740_991 {
            return Err(invalid("seed exceeds the exact JSON integer range"));
        }
        finite_range(self.cfg, "cfg", 0.0, 30.0)?;
        finite_range(self.denoise, "denoise", 0.0, 1.0)?;
        finite_range(self.scale, "scale", 0.125, 8.0)?;
        bounded_text(&self.prompt, "prompt", 32_768, false)?;
        bounded_text(&self.negative, "negative", 32_768, false)?;
        bounded_text(&self.checkpoint, "checkpoint", 1024, !matches!(self.mode, Mode::Custom | Mode::Upscale))?;
        bounded_text(&self.sampler, "sampler", 128, true)?;
        bounded_text(&self.scheduler, "scheduler", 128, true)?;
        bounded_text(&self.upscaler, "upscaler", 1024, self.mode == Mode::Upscale)?;
        if self.loras.len() > 16 {
            return Err(invalid("at most 16 LoRAs are allowed"));
        }
        for lora in &self.loras {
            bounded_text(&lora.name, "LoRA name", 1024, true)?;
            finite_range(lora.strength, "LoRA strength", -4.0, 4.0)?;
        }
        if self.mode == Mode::Custom {
            validate_graph(self.workflow.as_ref().ok_or_else(|| invalid("custom mode requires an API workflow"))?)?;
        } else if self.workflow.is_some() {
            return Err(invalid("workflow is only accepted in custom mode"));
        }
        Ok(())
    }
}

/// Reject UI exports, excessive nesting/node counts, malformed links and dangling edges.
fn validate_graph(graph: &Value) -> Result<(), Error> {
    let nodes = graph.as_object().ok_or_else(|| invalid("workflow must be an API graph object"))?;
    if nodes.is_empty() || nodes.len() > MAX_NODES {
        return Err(invalid("workflow requires 1..=256 nodes"));
    }
    let mut stack = vec![(graph, 0usize)];
    let mut bytes = 0usize;
    let mut values = 0usize;
    while let Some((value, depth)) = stack.pop() {
        values += 1;
        if depth > MAX_DEPTH || values > 100_000 {
            return Err(invalid("workflow nesting or value count exceeds limit"));
        }
        bytes = bytes.saturating_add(match value {
            Value::String(s) => s.len().saturating_mul(6),
            _ => 32,
        });
        if bytes > MAX_JSON_BYTES {
            return Err(invalid("workflow exceeds JSON size limit"));
        }
        match value {
            Value::Array(items) => stack.extend(items.iter().map(|v| (v, depth + 1))),
            Value::Object(items) => {
                for (key, child) in items {
                    bytes = bytes.saturating_add(key.len().saturating_mul(6));
                    stack.push((child, depth + 1));
                }
            }
            _ => {}
        }
    }
    for (id, node) in nodes {
        bounded_text(id, "node id", 128, true)?;
        let class = node.get("class_type").and_then(Value::as_str).ok_or_else(|| invalid("node requires class_type"))?;
        bounded_text(class, "class_type", 256, true)?;
        let inputs = node.get("inputs").and_then(Value::as_object).ok_or_else(|| invalid("node requires inputs object"))?;
        for input in inputs.values() {
            if let Value::Array(link) = input {
                let source = link.first().and_then(Value::as_str);
                let slot = link.get(1).and_then(Value::as_u64);
                if link.len() != 2 || !source.is_some_and(|s| nodes.contains_key(s)) || !slot.is_some_and(|s| s < 256) {
                    return Err(invalid("input array must be a valid [node_id, output_index] link"));
                }
            }
        }
    }
    // Kahn-style elimination detects cycles without recursive traversal of hostile graphs.
    let mut resolved = std::collections::BTreeSet::new();
    for _ in 0..nodes.len() {
        let before = resolved.len();
        for (id, node) in nodes {
            if node.get("inputs").and_then(Value::as_object).is_some_and(|inputs| {
                inputs
                    .values()
                    .all(|input| input.as_array().is_none_or(|link| link.first().and_then(Value::as_str).is_some_and(|source| resolved.contains(source))))
            }) {
                resolved.insert(id.as_str());
            }
        }
        if resolved.len() == nodes.len() {
            return Ok(());
        }
        if before == resolved.len() {
            return Err(invalid("workflow contains a dependency cycle"));
        }
    }
    Err(invalid("workflow could not be resolved"))
}

fn image_name(value: Option<&str>, label: &str) -> Result<String, Error> {
    let name = value.ok_or_else(|| invalid(format!("{label} image is required")))?;
    bounded_text(name, label, 2048, true)?;
    if name.starts_with('/') || name.contains('\\') || name.split('/').any(|part| part == "..") {
        return Err(invalid(format!("{label} must be a server-relative image name")));
    }
    Ok(name.into())
}

/// Independently authored stock-node graphs. Inputs are remote names returned by upload_png.
/// Inpaint masks use red=editable, black=preserved, independently of PNG alpha.
pub fn build_workflow(request: &Request, input: Option<&str>, mask: Option<&str>) -> Result<Value, Error> {
    request.validate()?;
    if request.mode == Mode::Custom {
        return request.workflow.clone().ok_or_else(|| invalid("custom workflow missing"));
    }
    let mut graph = serde_json::Map::new();
    let mut add = |id: &str, class: &str, inputs: Value| {
        graph.insert(id.into(), json!({"class_type": class, "inputs": inputs}));
    };
    if request.mode == Mode::Upscale {
        add("image", "LoadImage", json!({"image": image_name(input, "input")?}));
        add("upscaler", "UpscaleModelLoader", json!({"model_name": request.upscaler}));
        add("upscale", "ImageUpscaleWithModel", json!({"upscale_model": ["upscaler", 0], "image": ["image", 0]}));
        // scale is relative to the learned upscaler output, not the original image.
        add("scale", "ImageScaleBy", json!({"image": ["upscale", 0], "upscale_method": "lanczos", "scale_by": request.scale}));
        add("save", "SaveImage", json!({"images": ["scale", 0], "filename_prefix": "PhotoCraft"}));
        return Ok(Value::Object(graph));
    }
    add("checkpoint", "CheckpointLoaderSimple", json!({"ckpt_name": request.checkpoint}));
    let mut model = json!(["checkpoint", 0]);
    let mut clip = json!(["checkpoint", 1]);
    for (index, lora) in request.loras.iter().enumerate() {
        let id = format!("lora_{index}");
        add(&id, "LoraLoader", json!({"model": model, "clip": clip, "lora_name": lora.name, "strength_model": lora.strength, "strength_clip": lora.strength}));
        model = json!([id, 0]);
        clip = json!([id, 1]);
    }
    add("positive", "CLIPTextEncode", json!({"clip": clip, "text": request.prompt}));
    add("negative", "CLIPTextEncode", json!({"clip": clip, "text": request.negative}));
    let latent = if request.mode == Mode::Generate {
        add("latent", "EmptyLatentImage", json!({"width": request.width, "height": request.height, "batch_size": request.batch}));
        json!(["latent", 0])
    } else {
        add("image", "LoadImage", json!({"image": image_name(input, "input")?}));
        add("encode", "VAEEncode", json!({"pixels": ["image", 0], "vae": ["checkpoint", 2]}));
        let encoded = if request.batch > 1 {
            add("repeat", "RepeatLatentBatch", json!({"samples": ["encode", 0], "amount": request.batch}));
            json!(["repeat", 0])
        } else {
            json!(["encode", 0])
        };
        if request.mode == Mode::Inpaint {
            add("mask_image", "LoadImage", json!({"image": image_name(mask, "mask")?}));
            add("mask", "ImageToMask", json!({"image": ["mask_image", 0], "channel": "red"}));
            add("masked", "SetLatentNoiseMask", json!({"samples": encoded, "mask": ["mask", 0]}));
            json!(["masked", 0])
        } else {
            encoded
        }
    };
    add(
        "sample",
        "KSampler",
        json!({"model": model, "positive": ["positive", 0], "negative": ["negative", 0], "latent_image": latent, "seed": request.seed, "steps": request.steps, "cfg": request.cfg, "sampler_name": request.sampler, "scheduler": request.scheduler, "denoise": if request.mode == Mode::Generate { 1.0 } else { request.denoise }}),
    );
    add("decode", "VAEDecode", json!({"samples": ["sample", 0], "vae": ["checkpoint", 2]}));
    add("save", "SaveImage", json!({"images": ["decode", 0], "filename_prefix": "PhotoCraft"}));
    Ok(Value::Object(graph))
}

/// Available string choices for a node's required or optional input in object_info.
pub fn model_choices<'a>(catalog: &'a Value, node: &str, input: &str) -> Vec<&'a str> {
    input_spec(catalog, node, input)
        .and_then(Value::as_array)
        .and_then(|spec| spec.first())
        .and_then(Value::as_array)
        .map(|choices| choices.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn input_spec<'a>(catalog: &'a Value, node: &str, input: &str) -> Option<&'a Value> {
    let inputs = catalog.get(node)?.get("input")?;
    inputs.get("required").and_then(|v| v.get(input)).or_else(|| inputs.get("optional").and_then(|v| v.get(input)))
}

/// Validate the built graph's node types, required inputs, enum choices and links against live object_info.
/// Custom nodes may perform further semantic validation on the server.
pub fn validate_catalog(graph: &Value, catalog: &Value) -> Result<(), Error> {
    validate_graph(graph)?;
    let nodes = graph.as_object().ok_or_else(|| invalid("expected graph"))?;
    for node in nodes.values() {
        let class = node.get("class_type").and_then(Value::as_str).ok_or_else(|| invalid("missing class_type"))?;
        let definition = catalog.get(class).filter(|v| v.is_object()).ok_or_else(|| invalid(format!("server does not provide node {class}")))?;
        let inputs = node.get("inputs").and_then(Value::as_object).ok_or_else(|| invalid("missing inputs"))?;
        if let Some(required) = definition.get("input").and_then(|v| v.get("required")).and_then(Value::as_object) {
            for name in required.keys() {
                if !inputs.contains_key(name) {
                    return Err(invalid(format!("{class}.{name} is required by server")));
                }
            }
        }
        for (name, value) in inputs {
            let spec = input_spec(catalog, class, name).and_then(Value::as_array).ok_or_else(|| invalid(format!("server has no input {class}.{name}")))?;
            let ty = spec.first().ok_or_else(|| invalid("empty server input schema"))?;
            if let Some(link) = value.as_array() {
                let source = link.first().and_then(Value::as_str).and_then(|id| nodes.get(id)).ok_or_else(|| invalid("missing link source"))?;
                let source_class = source.get("class_type").and_then(Value::as_str).ok_or_else(|| invalid("missing source class"))?;
                let slot = link.get(1).and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| invalid("invalid output slot"))?;
                let output = catalog
                    .get(source_class)
                    .and_then(|v| v.get("output"))
                    .and_then(Value::as_array)
                    .and_then(|v| v.get(slot))
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid(format!("missing output {source_class}.{slot}")))?;
                if let Some(expected) = ty.as_str()
                    && expected != output
                    && expected != "*"
                    && output != "*"
                {
                    return Err(invalid(format!("{class}.{name} expects {expected}, not {output}")));
                }
            } else if let Some(choices) = ty.as_array() {
                if !choices.contains(value) {
                    return Err(invalid(format!("{class}.{name} is not available on server")));
                }
            } else {
                let valid = match ty.as_str() {
                    Some("INT") => value.as_i64().is_some() || value.as_u64().is_some(),
                    Some("FLOAT") => value.as_f64().is_some_and(f64::is_finite),
                    Some("STRING") => value.is_string(),
                    Some("BOOLEAN") => value.is_boolean(),
                    Some("*") => true,
                    _ => false,
                };
                if !valid {
                    return Err(invalid(format!("invalid literal for {class}.{name}")));
                }
                if let Some(number) = value.as_f64()
                    && let Some(options) = spec.get(1)
                    && (options.get("min").and_then(Value::as_f64).is_some_and(|min| number < min)
                        || options.get("max").and_then(Value::as_f64).is_some_and(|max| number > max))
                {
                    return Err(invalid(format!("{class}.{name} exceeds server bounds")));
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    pub filename: String,
    #[serde(default)]
    pub subfolder: String,
    #[serde(rename = "type", alias = "kind")]
    pub kind: String,
}

#[cfg(test)]
mod tests;
