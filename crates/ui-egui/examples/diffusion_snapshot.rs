//! Render an offscreen snapshot of the PhotoCraft AI Diffusion floating panel.
//!
//! Usage:
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example diffusion_snapshot -- --out /tmp/diffusion_snapshot.png
//! ```

use egui_kittest::Harness;
use photocraft_ui_egui::PhotocraftApp;
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1).cloned()).unwrap_or_else(|| "/tmp/diffusion_snapshot.png".into());

    let mut session = photocraft_engine::Session::new();
    let _ = session.execute(
        "file.new",
        json!({
            "width": 1000,
            "height": 700,
            "background": "white",
        }),
    );

    let mut harness = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_pixels_per_point(1.0).with_max_steps(32).wgpu().build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(session, Default::default());
        // Open the AI Diffusion panel and set illustrative defaults
        app.ui.panels.diffusion = true;
        app.ui.diffusion.checkpoint = "v1-5-pruned-emaonly.safetensors".into();
        app.ui.diffusion.prompt = "A picturesque mountain valley with serene lake at sunset, cinematic lighting".into();
        app.ui.diffusion.negative = "ugly, blurry, distorted, low quality".into();
        app.ui.diffusion.steps = 25;
        app.ui.diffusion.cfg = 7.5;
        app
    });

    harness.run_steps(8);

    match harness.render() {
        Ok(img) => match img.save(&out_path) {
            Ok(_) => println!("Successfully saved diffusion panel snapshot to {out_path}"),
            Err(e) => eprintln!("Failed to save snapshot image to {out_path}: {e}"),
        },
        Err(e) => eprintln!("Harness render error: {e}"),
    }
}
