//! Window › AI Diffusion: native floating panel for ComfyUI generative diffusion.
//!
//! Provides clean-room ComfyUI integration:
//! - Connection and node catalog introspection (`ai.diffusion.catalog`).
//! - Text-to-image (Generate), image-to-image (Refine), inpainting with selection mask,
//!   and upscaling (`ai.diffusion.generate`).
//! - Explicit upload consent checkbox (default false) before sending canvas or masks.
//! - Bounded parameter validation and clamping for dimensions, steps, CFG, LoRAs, and seed.
//! - History review, cached preview rendering, and color-managed application to documents
//!   (`ai.diffusion.history`, `ai.diffusion.apply`, `ai.diffusion.discard`, `ai.diffusion.cancel`).
//! - Presets management stored directly in `UiState.diffusion.presets`.

use egui::{CornerRadius, RichText, Stroke, vec2};
use photocraft_diffusion::Mode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::theme::Tokens;
use crate::{PhotocraftApp, icons, widgets};

/// Maximum width and height of the floating diffusion panel.
const PANEL_WIDTH: f32 = 440.0;

/// A LoRA model entry configured in the Diffusion panel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffusionLoraUi {
    pub name: String,
    pub strength: f64,
}

impl DiffusionLoraUi {
    pub fn new(name: impl Into<String>, strength: f64) -> Self {
        Self { name: name.into(), strength }
    }
}

/// A saved parameter preset for diffusion generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffusionPreset {
    pub name: String,
    pub mode: Mode,
    pub checkpoint: String,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f64,
    pub denoise: f64,
    pub sampler: String,
    pub scheduler: String,
    pub loras: Vec<DiffusionLoraUi>,
}

/// Serializable UI state and parameters for AI Diffusion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DiffusionUiState {
    pub endpoint: String,
    pub consent: bool,
    pub mode: Mode,
    pub checkpoint: String,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f64,
    pub seed: u64,
    pub random_seed: bool,
    pub denoise: f64,
    pub batch: u32,
    pub sampler: String,
    pub scheduler: String,
    pub loras: Vec<DiffusionLoraUi>,
    pub upscaler: String,
    pub scale: f64,
    pub custom_workflow: String,
    pub timeout_secs: u64,
    pub apply_on_finish: bool,
    pub presets: Vec<DiffusionPreset>,
}

impl Default for DiffusionUiState {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8188".into(),
            consent: false,
            mode: Mode::Generate,
            checkpoint: String::new(),
            prompt: String::new(),
            negative: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg: 7.0,
            seed: 0,
            random_seed: true,
            denoise: 0.5,
            batch: 1,
            sampler: "euler".into(),
            scheduler: "normal".into(),
            loras: Vec::new(),
            upscaler: String::new(),
            scale: 2.0,
            custom_workflow: String::new(),
            timeout_secs: 120,
            apply_on_finish: false,
            presets: Vec::new(),
        }
    }
}

impl DiffusionUiState {
    /// Convert UI state to a validated `photocraft_diffusion::Request`.
    pub fn to_request(&self) -> Result<photocraft_diffusion::Request, String> {
        let width = (self.width.clamp(64, 8192) / 8) * 8;
        let height = (self.height.clamp(64, 8192) / 8) * 8;
        let steps = self.steps.clamp(1, 150);
        let cfg = if self.cfg.is_finite() { self.cfg.clamp(0.0, 30.0) } else { 7.0 };
        let denoise = if self.denoise.is_finite() { self.denoise.clamp(0.0, 1.0) } else { 0.5 };
        let scale = if self.scale.is_finite() { self.scale.clamp(0.125, 8.0) } else { 2.0 };
        let batch = self.batch.clamp(1, 8);
        let seed = self.seed.min(9_007_199_254_740_991);

        if u64::from(width) * u64::from(height) * u64::from(batch) > 33_554_432 {
            return Err("Dimensions and batch exceed total pixel limit of 32 MiPixels".into());
        }

        let prompt = truncate_chars(&self.prompt, 32_768);
        let negative = truncate_chars(&self.negative, 32_768);
        let checkpoint = truncate_chars(&self.checkpoint, 1024);
        let sampler = if self.sampler.trim().is_empty() { "euler".into() } else { truncate_chars(&self.sampler, 128) };
        let scheduler = if self.scheduler.trim().is_empty() { "normal".into() } else { truncate_chars(&self.scheduler, 128) };
        let upscaler = truncate_chars(&self.upscaler, 1024);

        let mut loras = Vec::new();
        for l in self.loras.iter().take(16) {
            let name = truncate_chars(&l.name, 1024);
            if !name.trim().is_empty() {
                let strength = if l.strength.is_finite() { l.strength.clamp(-4.0, 4.0) } else { 1.0 };
                loras.push(photocraft_diffusion::Lora { name, strength });
            }
        }

        let workflow = if self.mode == Mode::Custom {
            if self.custom_workflow.trim().is_empty() {
                return Err("Custom mode requires an API workflow JSON".into());
            }
            let val: Value = serde_json::from_str(&self.custom_workflow).map_err(|e| format!("Invalid workflow JSON: {e}"))?;
            Some(val)
        } else {
            None
        };

        let req = photocraft_diffusion::Request {
            mode: self.mode,
            checkpoint,
            prompt,
            negative,
            width,
            height,
            steps,
            cfg,
            seed,
            denoise,
            batch,
            sampler,
            scheduler,
            loras,
            upscaler,
            scale,
            workflow,
        };

        req.validate().map_err(|e| e.to_string())?;
        Ok(req)
    }
}

fn truncate_chars(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        s.replace('\0', "")
    } else {
        let mut end = max_bytes;
        while !s.is_char_boundary(end) && end > 0 {
            end = end.saturating_sub(1);
        }
        s.get(..end).unwrap_or("").replace('\0', "")
    }
}

/// Decode raw PNG preview bytes into an `egui::ColorImage`.
pub fn decode_preview_png(bytes: &[u8]) -> Option<egui::ColorImage> {
    if bytes.is_empty() {
        return None;
    }
    if let Ok(dyn_img) = image::load_from_memory(bytes) {
        let rgba = dyn_img.to_rgba8();
        let (w, h) = (rgba.width() as usize, rgba.height() as usize);
        return Some(egui::ColorImage::from_rgba_unmultiplied([w, h], rgba.as_raw()));
    }
    if let Ok(img) = photocraft_codecs::decode(bytes) {
        let conv = img.convert(photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8);
        let (w, h) = (conv.width() as usize, conv.height() as usize);
        if conv.data().len() == w.saturating_mul(h).saturating_mul(4) {
            return Some(egui::ColorImage::from_rgba_unmultiplied([w, h], conv.data()));
        }
    }
    None
}

/// Active tab in the Diffusion panel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DiffusionTab {
    #[default]
    Parameters,
    History,
    Presets,
}

/// One summary record in the history view.
#[derive(Clone, Debug)]
pub struct DiffusionHistoryEntry {
    pub job: u64,
    pub mode: Mode,
    pub prompt: String,
    pub images: usize,
    pub bytes: usize,
    pub created_ms: f64,
}

/// Ephemeral runtime cache stored in `egui::Context` memory.
#[derive(Clone, Default)]
pub struct DiffusionRuntime {
    pub catalog_checkpoints: Vec<String>,
    pub catalog_samplers: Vec<String>,
    pub catalog_schedulers: Vec<String>,
    pub catalog_loras: Vec<String>,
    pub catalog_upscalers: Vec<String>,
    pub catalog_fetched: bool,
    pub active_tab: DiffusionTab,
    pub history: Vec<DiffusionHistoryEntry>,
    pub selected_preview: Option<(u64, usize)>,
    pub preview_tex: Option<egui::TextureHandle>,
    pub preview_key: Option<(u64, usize)>,
    pub status: Option<String>,
    pub status_is_error: bool,
    pub preset_name_input: String,
    pub new_lora_name: String,
}

/// Draw the floating AI Diffusion panel if toggled open.
pub fn window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.ui.panels.diffusion {
        return;
    }

    let t = Tokens::get(ctx);
    let frame = egui::Frame::NONE
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.card_border))
        .corner_radius(CornerRadius::same(t.radius_lg as u8))
        .shadow(egui::Shadow { offset: [0, 10], blur: 30, spread: 0, color: t.shadow })
        .inner_margin(egui::Margin::same(12));

    let canvas = app.last_canvas_rect;
    let default_pos = egui::pos2((canvas.right() - PANEL_WIDTH - 20.0).max(canvas.left() + 10.0), canvas.top() + 30.0);

    let runtime_id = egui::Id::unique("photocraft-diffusion-runtime");
    let mut runtime: DiffusionRuntime = ctx.data_mut(|d| d.get_temp(runtime_id)).unwrap_or_default();

    let mut open = true;
    egui::Window::new(tl!("AI Diffusion"))
        .id(egui::Id::unique("photocraft-ai-diffusion-window"))
        .title_bar(false)
        .resizable(true)
        .default_width(PANEL_WIDTH)
        .default_pos(default_pos)
        .frame(frame)
        .show(ctx, |ui| {
            ui.set_width(PANEL_WIDTH);

            // Title bar with close button
            ui.horizontal(|ui| {
                ui.heading(tl!("AI Diffusion"));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                        open = false;
                    }
                });
            });

            ui.add_space(4.0);

            // Tab bar
            ui.horizontal(|ui| {
                for (tab_val, label) in
                    [(DiffusionTab::Parameters, tl!("Parameters")), (DiffusionTab::History, tl!("History")), (DiffusionTab::Presets, tl!("Presets"))]
                {
                    if widgets::pill_tab(ui, label, runtime.active_tab == tab_val).clicked() {
                        runtime.active_tab = tab_val;
                        if tab_val == DiffusionTab::History {
                            refresh_history(app, &mut runtime);
                        }
                    }
                }
            });

            ui.add_space(6.0);

            #[cfg(target_arch = "wasm32")]
            {
                ui.colored_label(t.warning, tl!("AI Diffusion network operations are not supported on WebAssembly."));
                ui.add_space(4.0);
            }

            // Status message
            let mut dismiss = false;
            if let Some(msg) = &runtime.status {
                let color = if runtime.status_is_error { t.danger } else { t.accent };
                ui.horizontal(|ui| {
                    ui.colored_label(color, msg);
                    if ui.small_button(tl!("Dismiss")).clicked() {
                        dismiss = true;
                    }
                });
                ui.add_space(4.0);
            }
            if dismiss {
                runtime.status = None;
            }

            // Tab contents
            match runtime.active_tab {
                DiffusionTab::Parameters => draw_parameters_tab(app, ui, &t, &mut runtime),
                DiffusionTab::History => draw_history_tab(app, ui, ctx, &t, &mut runtime),
                DiffusionTab::Presets => draw_presets_tab(app, ui, &mut runtime),
            }
        });

    if !open {
        app.ui.panels.diffusion = false;
    }

    ctx.data_mut(|d| d.insert_temp(runtime_id, runtime));
}

fn draw_parameters_tab(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, runtime: &mut DiffusionRuntime) {
    egui::ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
        // Endpoint & Catalog connection
        ui.horizontal(|ui| {
            ui.label(tl!("Endpoint:"));
            ui.add(egui::TextEdit::singleline(&mut app.ui.diffusion.endpoint).desired_width(200.0));
            if ui.button(tl!("Connect")).on_hover_text(tl!("Fetch node catalog from server")).clicked() {
                refresh_catalog(app, runtime);
            }
        });

        ui.add_space(4.0);

        // Mode selector
        ui.horizontal(|ui| {
            ui.label(tl!("Mode:"));
            craft_ui::ComboBox::from_id_salt("diffusion-mode-select")
                .selected_text(match app.ui.diffusion.mode {
                    Mode::Generate => tl!("Generate (Text-to-Image)"),
                    Mode::Refine => tl!("Refine (Image-to-Image)"),
                    Mode::Inpaint => tl!("Inpaint (Selection Mask)"),
                    Mode::Upscale => tl!("Upscale"),
                    Mode::Custom => tl!("Custom (API Workflow)"),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.ui.diffusion.mode, Mode::Generate, tl!("Generate (Text-to-Image)"));
                    ui.selectable_value(&mut app.ui.diffusion.mode, Mode::Refine, tl!("Refine (Image-to-Image)"));
                    ui.selectable_value(&mut app.ui.diffusion.mode, Mode::Inpaint, tl!("Inpaint (Selection Mask)"));
                    ui.selectable_value(&mut app.ui.diffusion.mode, Mode::Upscale, tl!("Upscale"));
                    ui.selectable_value(&mut app.ui.diffusion.mode, Mode::Custom, tl!("Custom (API Workflow)"));
                });
        });

        ui.add_space(4.0);

        // Explicit upload consent for modes transmitting pixels
        let sends_pixels = matches!(app.ui.diffusion.mode, Mode::Refine | Mode::Inpaint | Mode::Upscale | Mode::Custom);

        if sends_pixels {
            ui.group(|ui| {
                ui.checkbox(&mut app.ui.diffusion.consent, tl!("I consent to sending the visible canvas and active selection to the chosen server"));
                if !app.ui.diffusion.consent {
                    ui.colored_label(t.warning, tl!("Consent required: uploads document pixels or selection mask to the selected server."));
                }
            });
            ui.add_space(4.0);
        }

        // Prompts
        if app.ui.diffusion.mode != Mode::Upscale && app.ui.diffusion.mode != Mode::Custom {
            ui.label(tl!("Prompt:"));
            ui.add(egui::TextEdit::multiline(&mut app.ui.diffusion.prompt).desired_rows(3).desired_width(f32::INFINITY));

            ui.add_space(2.0);
            ui.label(tl!("Negative Prompt:"));
            ui.add(egui::TextEdit::multiline(&mut app.ui.diffusion.negative).desired_rows(2).desired_width(f32::INFINITY));
            ui.add_space(4.0);
        }

        // Checkpoint selector
        if app.ui.diffusion.mode != Mode::Upscale && app.ui.diffusion.mode != Mode::Custom {
            ui.horizontal(|ui| {
                ui.label(tl!("Checkpoint:"));
                if !runtime.catalog_checkpoints.is_empty() {
                    craft_ui::ComboBox::from_id_salt("diffusion-ckpt-picker")
                        .selected_text(if app.ui.diffusion.checkpoint.is_empty() { tl!("Select checkpoint…") } else { &app.ui.diffusion.checkpoint })
                        .show_ui(ui, |ui| {
                            for ckpt in &runtime.catalog_checkpoints {
                                ui.selectable_value(&mut app.ui.diffusion.checkpoint, ckpt.clone(), ckpt);
                            }
                        });
                } else {
                    ui.add(egui::TextEdit::singleline(&mut app.ui.diffusion.checkpoint).desired_width(220.0).hint_text(tl!("model.safetensors")));
                }
            });
            ui.add_space(4.0);
        }

        // Dimensions
        if matches!(app.ui.diffusion.mode, Mode::Generate | Mode::Refine | Mode::Inpaint) {
            ui.horizontal(|ui| {
                ui.label(tl!("Size:"));
                ui.add(egui::DragValue::new(&mut app.ui.diffusion.width).speed(8.0).range(64..=8192).prefix("W: "));
                ui.add(egui::DragValue::new(&mut app.ui.diffusion.height).speed(8.0).range(64..=8192).prefix("H: "));

                if ui.small_button("512").clicked() {
                    app.ui.diffusion.width = 512;
                    app.ui.diffusion.height = 512;
                }
                if ui.small_button("768").clicked() {
                    app.ui.diffusion.width = 768;
                    app.ui.diffusion.height = 768;
                }
                if ui.small_button("1024").clicked() {
                    app.ui.diffusion.width = 1024;
                    app.ui.diffusion.height = 1024;
                }
            });
            ui.add_space(4.0);
        }

        // Sliders (Steps, CFG, Denoise, Batch)
        if app.ui.diffusion.mode != Mode::Upscale && app.ui.diffusion.mode != Mode::Custom {
            ui.horizontal(|ui| {
                ui.label(tl!("Steps:"));
                ui.add(egui::Slider::new(&mut app.ui.diffusion.steps, 1..=150));
            });

            ui.horizontal(|ui| {
                ui.label(tl!("CFG:"));
                ui.add(egui::Slider::new(&mut app.ui.diffusion.cfg, 0.0..=30.0));
            });

            if app.ui.diffusion.mode != Mode::Generate {
                ui.horizontal(|ui| {
                    ui.label(tl!("Denoise:"));
                    ui.add(egui::Slider::new(&mut app.ui.diffusion.denoise, 0.0..=1.0));
                });
            }

            ui.horizontal(|ui| {
                ui.label(tl!("Batch:"));
                ui.add(egui::Slider::new(&mut app.ui.diffusion.batch, 1..=8));
            });

            ui.add_space(4.0);

            // Sampler & Scheduler
            ui.horizontal(|ui| {
                ui.label(tl!("Sampler:"));
                if !runtime.catalog_samplers.is_empty() {
                    craft_ui::ComboBox::from_id_salt("diffusion-sampler-picker").selected_text(&app.ui.diffusion.sampler).show_ui(ui, |ui| {
                        for s in &runtime.catalog_samplers {
                            ui.selectable_value(&mut app.ui.diffusion.sampler, s.clone(), s);
                        }
                    });
                } else {
                    ui.add(egui::TextEdit::singleline(&mut app.ui.diffusion.sampler).desired_width(100.0));
                }

                ui.label(tl!("Scheduler:"));
                if !runtime.catalog_schedulers.is_empty() {
                    craft_ui::ComboBox::from_id_salt("diffusion-scheduler-picker").selected_text(&app.ui.diffusion.scheduler).show_ui(ui, |ui| {
                        for s in &runtime.catalog_schedulers {
                            ui.selectable_value(&mut app.ui.diffusion.scheduler, s.clone(), s);
                        }
                    });
                } else {
                    ui.add(egui::TextEdit::singleline(&mut app.ui.diffusion.scheduler).desired_width(100.0));
                }
            });

            ui.add_space(4.0);

            // Seed
            ui.horizontal(|ui| {
                ui.label(tl!("Seed:"));
                ui.add_enabled(!app.ui.diffusion.random_seed, egui::DragValue::new(&mut app.ui.diffusion.seed));
                ui.checkbox(&mut app.ui.diffusion.random_seed, tl!("Randomize"));
            });

            ui.add_space(4.0);

            // LoRAs collapsible
            ui.collapsing(format!("{} ({})", tl!("LoRAs"), app.ui.diffusion.loras.len()), |ui| {
                let mut remove_idx = None;
                for (idx, lora) in app.ui.diffusion.loras.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}:", idx + 1));
                        ui.add(egui::TextEdit::singleline(&mut lora.name).desired_width(120.0));
                        ui.add(egui::Slider::new(&mut lora.strength, -4.0..=4.0).text(tl!("Strength")));
                        if ui.small_button("×").clicked() {
                            remove_idx = Some(idx);
                        }
                    });
                }
                if let Some(i) = remove_idx
                    && i < app.ui.diffusion.loras.len()
                {
                    app.ui.diffusion.loras.remove(i);
                }
                if app.ui.diffusion.loras.len() < 16 {
                    ui.horizontal(|ui| {
                        if !runtime.catalog_loras.is_empty() {
                            craft_ui::ComboBox::from_id_salt("diffusion-add-lora-picker")
                                .selected_text(if runtime.new_lora_name.is_empty() { tl!("Pick LoRA…") } else { &runtime.new_lora_name })
                                .show_ui(ui, |ui| {
                                    for l in &runtime.catalog_loras {
                                        ui.selectable_value(&mut runtime.new_lora_name, l.clone(), l);
                                    }
                                });
                        } else {
                            ui.add(egui::TextEdit::singleline(&mut runtime.new_lora_name).hint_text(tl!("LoRA name")));
                        }
                        if ui.button(tl!("Add LoRA")).clicked() && !runtime.new_lora_name.trim().is_empty() {
                            app.ui.diffusion.loras.push(DiffusionLoraUi::new(runtime.new_lora_name.trim(), 1.0));
                            runtime.new_lora_name.clear();
                        }
                    });
                }
            });
        }

        // Upscale parameters
        if app.ui.diffusion.mode == Mode::Upscale {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(tl!("Upscaler:"));
                    if !runtime.catalog_upscalers.is_empty() {
                        craft_ui::ComboBox::from_id_salt("diffusion-upscaler-picker")
                            .selected_text(if app.ui.diffusion.upscaler.is_empty() { tl!("Select upscaler…") } else { &app.ui.diffusion.upscaler })
                            .show_ui(ui, |ui| {
                                for u_name in &runtime.catalog_upscalers {
                                    ui.selectable_value(&mut app.ui.diffusion.upscaler, u_name.clone(), u_name);
                                }
                            });
                    } else {
                        ui.add(egui::TextEdit::singleline(&mut app.ui.diffusion.upscaler).desired_width(200.0));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(tl!("Scale:"));
                    ui.add(egui::Slider::new(&mut app.ui.diffusion.scale, 0.125..=8.0));
                });
            });
            ui.add_space(4.0);
        }

        // Custom workflow
        if app.ui.diffusion.mode == Mode::Custom {
            ui.label(tl!("API Workflow JSON:"));
            ui.add(egui::TextEdit::multiline(&mut app.ui.diffusion.custom_workflow).code_editor().desired_rows(8).desired_width(f32::INFINITY));
            ui.add_space(4.0);
        }

        ui.checkbox(&mut app.ui.diffusion.apply_on_finish, tl!("Apply to document immediately after generation"));

        ui.add_space(8.0);

        // Action buttons
        let is_running = app.session.jobs().iter().any(|j| j.label.contains("Diffusion") && j.state == "running");
        let can_generate = !is_running && (!sends_pixels || app.ui.diffusion.consent);

        ui.horizontal(|ui| {
            let gen_btn = ui.add_enabled(can_generate, egui::Button::new(RichText::new(tl!("Generate")).strong()).min_size(vec2(100.0, 26.0)));
            if gen_btn.clicked() {
                trigger_generate(app, runtime);
            }

            if is_running {
                ui.spinner();
                ui.label(tl!("Generating…"));
                if ui.button(tl!("Cancel")).clicked() {
                    trigger_cancel(app, runtime);
                }
            }
        });
    });
}

fn draw_history_tab(app: &mut PhotocraftApp, ui: &mut egui::Ui, ctx: &egui::Context, _t: &Tokens, runtime: &mut DiffusionRuntime) {
    ui.horizontal(|ui| {
        if ui.button(tl!("Refresh History")).clicked() {
            refresh_history(app, runtime);
        }
        if ui.button(tl!("Clear All")).on_hover_text(tl!("Clear all diffusion history from memory")).clicked() {
            discard_job(app, runtime, None);
        }
    });

    ui.separator();

    if runtime.history.is_empty() {
        ui.label(tl!("No diffusion generations in history."));
        return;
    }

    // Scrollable history list
    egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
        for entry in &runtime.history {
            let is_selected = runtime.selected_preview.map(|(j, _)| j) == Some(entry.job);
            let label_text = format!("#{}: {:?} - {}", entry.job, entry.mode, entry.prompt);
            if ui.selectable_label(is_selected, label_text).clicked() {
                runtime.selected_preview = Some((entry.job, 0));
            }
        }
    });

    // Preview area for selected entry
    if let Some((job, index)) = runtime.selected_preview {
        ui.separator();
        ui.label(format!("{} #{} (Image {})", tl!("Preview: Job"), job, index + 1));

        // Update preview texture if needed
        if runtime.preview_key != Some((job, index)) || runtime.preview_tex.is_none() {
            if let Some(png_bytes) = photocraft_engine::diffusion_cmds::preview(&app.session, job, index) {
                if let Some(ci) = decode_preview_png(png_bytes) {
                    let tex = ctx.load_texture("photocraft-diffusion-preview", ci, egui::TextureOptions::LINEAR);
                    runtime.preview_tex = Some(tex);
                    runtime.preview_key = Some((job, index));
                } else {
                    runtime.preview_tex = None;
                    runtime.preview_key = None;
                }
            } else {
                runtime.preview_tex = None;
                runtime.preview_key = None;
            }
        }

        if let Some(tex) = &runtime.preview_tex {
            let size = tex.size_vec2();
            if size.x > 0.0 && size.y > 0.0 {
                let max_w = 400.0;
                let max_h = 240.0;
                let scale = (max_w / size.x).min(max_h / size.y).min(1.0);
                let display_size = size * scale;
                ui.image((tex.id(), display_size));
            }

            ui.horizontal(|ui| {
                if ui.button(tl!("Apply as Layer")).on_hover_text(tl!("Insert generated image as a layer in active document")).clicked() {
                    apply_job(app, runtime, job, index, false);
                }
                if ui.button(tl!("Open as Document")).on_hover_text(tl!("Open generated image as a new document")).clicked() {
                    apply_job(app, runtime, job, index, true);
                }
                if ui.button(tl!("Discard")).on_hover_text(tl!("Remove this generation from memory")).clicked() {
                    discard_job(app, runtime, Some(job));
                }
            });
        } else {
            ui.label(tl!("No preview available for this job."));
        }
    }
}

fn draw_presets_tab(app: &mut PhotocraftApp, ui: &mut egui::Ui, runtime: &mut DiffusionRuntime) {
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut runtime.preset_name_input).hint_text(tl!("Preset name…")));
        if ui.button(tl!("Save Preset")).clicked() {
            let name = runtime.preset_name_input.trim().to_string();
            if !name.is_empty() {
                let preset = DiffusionPreset {
                    name: name.clone(),
                    mode: app.ui.diffusion.mode,
                    checkpoint: app.ui.diffusion.checkpoint.clone(),
                    prompt: app.ui.diffusion.prompt.clone(),
                    negative: app.ui.diffusion.negative.clone(),
                    width: app.ui.diffusion.width,
                    height: app.ui.diffusion.height,
                    steps: app.ui.diffusion.steps,
                    cfg: app.ui.diffusion.cfg,
                    denoise: app.ui.diffusion.denoise,
                    sampler: app.ui.diffusion.sampler.clone(),
                    scheduler: app.ui.diffusion.scheduler.clone(),
                    loras: app.ui.diffusion.loras.clone(),
                };
                if let Some(pos) = app.ui.diffusion.presets.iter().position(|p| p.name == name) {
                    app.ui.diffusion.presets[pos] = preset;
                } else {
                    app.ui.diffusion.presets.push(preset);
                }
                runtime.preset_name_input.clear();
                runtime.status = Some(format!("Saved preset '{name}'"));
                runtime.status_is_error = false;
            }
        }
    });

    ui.separator();

    if app.ui.diffusion.presets.is_empty() {
        ui.label(tl!("No presets saved."));
        return;
    }

    egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
        let mut delete_preset_name = None;
        for preset in &app.ui.diffusion.presets {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&preset.name).strong());
                    ui.label(format!("({:?}, {}×{}, steps: {})", preset.mode, preset.width, preset.height, preset.steps));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button(tl!("Delete")).clicked() {
                            delete_preset_name = Some(preset.name.clone());
                        }
                        if ui.small_button(tl!("Load")).clicked() {
                            app.ui.diffusion.mode = preset.mode;
                            app.ui.diffusion.checkpoint = preset.checkpoint.clone();
                            app.ui.diffusion.prompt = preset.prompt.clone();
                            app.ui.diffusion.negative = preset.negative.clone();
                            app.ui.diffusion.width = preset.width;
                            app.ui.diffusion.height = preset.height;
                            app.ui.diffusion.steps = preset.steps;
                            app.ui.diffusion.cfg = preset.cfg;
                            app.ui.diffusion.denoise = preset.denoise;
                            app.ui.diffusion.sampler = preset.sampler.clone();
                            app.ui.diffusion.scheduler = preset.scheduler.clone();
                            app.ui.diffusion.loras = preset.loras.clone();
                            runtime.status = Some(format!("Loaded preset '{}'", preset.name));
                            runtime.status_is_error = false;
                        }
                    });
                });
            });
        }
        if let Some(name) = delete_preset_name {
            app.ui.diffusion.presets.retain(|p| p.name != name);
        }
    });
}

fn trigger_generate(app: &mut PhotocraftApp, runtime: &mut DiffusionRuntime) {
    if app.ui.diffusion.random_seed {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        app.ui.diffusion.seed = (now as u64) & 0x001F_FFFF_FFFF_FFFF;
    }

    match app.ui.diffusion.to_request() {
        Ok(request) => {
            let p = json!({
                "endpoint": &app.ui.diffusion.endpoint,
                "consent": app.ui.diffusion.consent,
                "apply": app.ui.diffusion.apply_on_finish,
                "timeoutSecs": app.ui.diffusion.timeout_secs,
                "request": serde_json::to_value(&request).unwrap_or(Value::Null),
            });
            match app.run("ai.diffusion.generate", p) {
                Ok(res) => {
                    if let Some(job) = res.get("job").and_then(Value::as_u64) {
                        runtime.selected_preview = Some((job, 0));
                    }
                    runtime.status = Some("Diffusion generation started".into());
                    runtime.status_is_error = false;
                    refresh_history(app, runtime);
                }
                Err(e) => {
                    runtime.status = Some(format!("Generation failed: {e}"));
                    runtime.status_is_error = true;
                }
            }
        }
        Err(e) => {
            runtime.status = Some(format!("Invalid parameters: {e}"));
            runtime.status_is_error = true;
        }
    }
}

fn trigger_cancel(app: &mut PhotocraftApp, runtime: &mut DiffusionRuntime) {
    let job_id = app.session.jobs().iter().find(|j| j.label.contains("Diffusion") && j.state == "running").map(|j| j.id);
    if let Some(id) = job_id {
        let _ = crate::jobs_ui::cancel(app, id);
        let _ = app.run("ai.diffusion.cancel", json!({ "job": id.0 }));
        runtime.status = Some("Diffusion generation cancelled".into());
        runtime.status_is_error = false;
    }
}

fn refresh_catalog(app: &mut PhotocraftApp, runtime: &mut DiffusionRuntime) {
    let p = json!({ "endpoint": &app.ui.diffusion.endpoint });
    match app.run("ai.diffusion.catalog", p) {
        Ok(val) => {
            if let Some(cat) = val.get("catalog") {
                runtime.catalog_checkpoints =
                    photocraft_diffusion::model_choices(cat, "CheckpointLoaderSimple", "ckpt_name").into_iter().map(String::from).collect();
                runtime.catalog_samplers = photocraft_diffusion::model_choices(cat, "KSampler", "sampler_name").into_iter().map(String::from).collect();
                runtime.catalog_schedulers = photocraft_diffusion::model_choices(cat, "KSampler", "scheduler").into_iter().map(String::from).collect();
                runtime.catalog_loras = photocraft_diffusion::model_choices(cat, "LoraLoader", "lora_name").into_iter().map(String::from).collect();
                runtime.catalog_upscalers =
                    photocraft_diffusion::model_choices(cat, "UpscaleModelLoader", "model_name").into_iter().map(String::from).collect();
                runtime.catalog_fetched = true;
                runtime.status = Some("Catalog refreshed successfully".into());
                runtime.status_is_error = false;
            }
        }
        Err(e) => {
            runtime.status = Some(format!("Catalog request failed: {e}"));
            runtime.status_is_error = true;
        }
    }
}

fn refresh_history(app: &mut PhotocraftApp, runtime: &mut DiffusionRuntime) {
    if let Ok(val) = app.run("ai.diffusion.history", json!({}))
        && let Some(arr) = val.get("history").and_then(Value::as_array)
    {
        runtime.history = arr
            .iter()
            .filter_map(|item| {
                Some(DiffusionHistoryEntry {
                    job: item.get("job")?.as_u64()?,
                    mode: serde_json::from_value(item.get("mode")?.clone()).ok()?,
                    prompt: item.get("prompt").and_then(Value::as_str).unwrap_or("").to_string(),
                    images: item.get("images").and_then(Value::as_u64).unwrap_or(0) as usize,
                    bytes: item.get("bytes").and_then(Value::as_u64).unwrap_or(0) as usize,
                    created_ms: item.get("createdMs").and_then(Value::as_f64).unwrap_or(0.0),
                })
            })
            .collect();
        if runtime.selected_preview.is_none()
            && let Some(last) = runtime.history.last()
        {
            runtime.selected_preview = Some((last.job, 0));
        }
    }
}

fn apply_job(app: &mut PhotocraftApp, runtime: &mut DiffusionRuntime, job: u64, index: usize, new_doc: bool) {
    let p = json!({
        "job": job,
        "index": index,
        "newDocument": new_doc,
    });
    match app.run("ai.diffusion.apply", p) {
        Ok(_) => {
            runtime.status =
                Some(if new_doc { format!("Opened job {job} image {index} as new document") } else { format!("Applied job {job} image {index} as new layer") });
            runtime.status_is_error = false;
        }
        Err(e) => {
            runtime.status = Some(format!("Apply failed: {e}"));
            runtime.status_is_error = true;
        }
    }
}

fn discard_job(app: &mut PhotocraftApp, runtime: &mut DiffusionRuntime, job: Option<u64>) {
    let p = if let Some(j) = job { json!({ "job": j }) } else { json!({}) };
    match app.run("ai.diffusion.discard", p) {
        Ok(_) => {
            if let Some(j) = job {
                if runtime.preview_key.map(|(k, _)| k) == Some(j) {
                    runtime.preview_tex = None;
                    runtime.preview_key = None;
                    runtime.selected_preview = None;
                }
            } else {
                runtime.preview_tex = None;
                runtime.preview_key = None;
                runtime.selected_preview = None;
            }
            refresh_history(app, runtime);
        }
        Err(e) => {
            runtime.status = Some(format!("Discard failed: {e}"));
            runtime.status_is_error = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffusion_ui_state_defaults() {
        let state = DiffusionUiState::default();
        assert_eq!(state.width, 512);
        assert_eq!(state.height, 512);
        assert_eq!(state.steps, 20);
        assert_eq!(state.cfg, 7.0);
        assert_eq!(state.denoise, 0.5);
        assert_eq!(state.batch, 1);
        assert_eq!(state.sampler, "euler");
        assert_eq!(state.scheduler, "normal");
        assert_eq!(state.endpoint, "http://127.0.0.1:8188");
        assert!(!state.consent, "Consent must always default to false");
        assert_eq!(state.mode, Mode::Generate);
        assert!(state.loras.is_empty());
        assert!(state.presets.is_empty());
    }

    #[test]
    fn test_diffusion_ui_state_serde_roundtrip() {
        let mut state = DiffusionUiState {
            checkpoint: "v1-5-pruned-emaonly.safetensors".into(),
            prompt: "A picturesque mountain lake at sunset".into(),
            negative: "ugly, blurry".into(),
            steps: 25,
            cfg: 8.5,
            consent: true,
            ..Default::default()
        };
        state.loras.push(DiffusionLoraUi::new("add_detail", 0.8));

        let json = serde_json::to_string(&state).expect("serialize");
        let parsed: DiffusionUiState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(state, parsed);
    }

    #[test]
    fn test_to_request_validation_and_clamping() {
        let state = DiffusionUiState {
            checkpoint: "model.safetensors".into(),
            width: 515,   // not multiple of 8
            height: 70,   // 70 / 8 * 8 = 64
            steps: 500,   // clamp to 150
            cfg: 50.0,    // clamp to 30.0
            denoise: 2.0, // clamp to 1.0
            batch: 12,    // clamp to 8
            ..Default::default()
        };

        let req = state.to_request().expect("valid request");
        assert_eq!(req.width, 512);
        assert_eq!(req.height, 64);
        assert_eq!(req.steps, 150);
        assert_eq!(req.cfg, 30.0);
        assert_eq!(req.denoise, 1.0);
        assert_eq!(req.batch, 8);
    }

    #[test]
    fn test_to_request_truncates_long_strings_safely() {
        let state = DiffusionUiState {
            checkpoint: "m".repeat(2000),
            prompt: "p".repeat(50000),
            negative: "n".repeat(50000),
            sampler: "s".repeat(300),
            scheduler: "c".repeat(300),
            ..Default::default()
        };

        let req = state.to_request().expect("safe truncation");
        assert_eq!(req.checkpoint.len(), 1024);
        assert_eq!(req.prompt.len(), 32768);
        assert_eq!(req.negative.len(), 32768);
        assert_eq!(req.sampler.len(), 128);
        assert_eq!(req.scheduler.len(), 128);
    }

    #[test]
    fn test_to_request_loras_bounded() {
        let mut state = DiffusionUiState { checkpoint: "model.safetensors".into(), ..Default::default() };
        for i in 0..25 {
            state.loras.push(DiffusionLoraUi::new(format!("lora_{i}"), 10.0));
        }
        let req = state.to_request().expect("valid request");
        assert_eq!(req.loras.len(), 16);
        assert_eq!(req.loras[0].strength, 4.0);
    }

    #[test]
    fn test_custom_mode_requires_workflow_json() {
        let mut state = DiffusionUiState { mode: Mode::Custom, ..Default::default() };
        assert!(state.to_request().is_err());

        state.custom_workflow = "not a valid json".into();
        assert!(state.to_request().is_err());

        state.custom_workflow = serde_json::json!({
            "1": {
                "class_type": "SaveImage",
                "inputs": {}
            }
        })
        .to_string();
        assert!(state.to_request().is_ok());
    }

    #[test]
    fn test_presets_save_load_delete() {
        let mut state = DiffusionUiState { checkpoint: "test.safetensors".into(), prompt: "sample prompt".into(), steps: 35, cfg: 9.0, ..Default::default() };

        let preset = DiffusionPreset {
            name: "Anime".into(),
            mode: state.mode,
            checkpoint: state.checkpoint.clone(),
            prompt: state.prompt.clone(),
            negative: state.negative.clone(),
            width: state.width,
            height: state.height,
            steps: state.steps,
            cfg: state.cfg,
            denoise: state.denoise,
            sampler: state.sampler.clone(),
            scheduler: state.scheduler.clone(),
            loras: state.loras.clone(),
        };

        state.presets.push(preset.clone());
        assert_eq!(state.presets.len(), 1);

        state.steps = 10;
        assert_ne!(state.steps, preset.steps);

        let loaded = &state.presets[0];
        state.steps = loaded.steps;
        assert_eq!(state.steps, 35);

        state.presets.retain(|p| p.name != "Anime");
        assert!(state.presets.is_empty());
    }

    #[test]
    fn test_decode_preview_png_invalid_bytes() {
        assert!(decode_preview_png(&[]).is_none());
        assert!(decode_preview_png(&[1, 2, 3, 4, 5]).is_none());
    }

    #[test]
    fn test_decode_preview_png_valid_image() {
        let mut png_bytes = Vec::new();
        let dyn_img = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255])));
        dyn_img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png).expect("write png");

        let decoded = decode_preview_png(&png_bytes).expect("decode preview");
        assert_eq!(decoded.size, [2, 2]);
        assert_eq!(decoded.pixels.len(), 4);
    }
}
