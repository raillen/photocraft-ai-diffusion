# Generative Diffusion Architecture

PhotoCraft integrates generative AI diffusion natively via clean-room communication with stock ComfyUI instances (local or remote), avoiding proprietary code, Electron, Python runtime requirements, or licensing contamination from GPL plugins.

## 1. Design & Boundaries

- **`photocraft-diffusion` (L0 standalone crate):**
  - Defines clean-room API data types (`Request`, `Mode`, `Output`, `Error`).
  - Serializes stock ComfyUI API graph workflows for `Generate` (Text-to-Image), `Refine` (Image-to-Image), `Inpaint` (selection-masked inpainting), `Upscale` (model-based latent upscale), and `Custom` (custom workflow injection).
  - Implements bounded native HTTP transport (`ureq`) with strict timeouts, size caps, and SSRF guardrails.
  - Compiles on both native targets and `wasm32-unknown-unknown` (graceful unsupported runtime error for network calls on wasm).

- **`photocraft-engine` (L5 orchestrator):**
  - Manages background inference jobs via `photocraft_engine::jobs::run`.
  - Dispatches `ai.diffusion.catalog`, `ai.diffusion.generate`, `ai.diffusion.history`, `ai.diffusion.apply`, and `ai.diffusion.discard`.
  - Renders input documents and selection masks with ICC color management before sending.
  - Applies generated results non-destructively as new raster layers with full undo/redo history.
  - Enforces explicit consent for outbound document data transmission.

- **`photocraft-ui-egui` (L6 interface):**
  - Provides the dockable/floating **AI Diffusion** panel (`Window › AI Diffusion`).
  - Fetches server checkpoints, samplers, schedulers, and upscalers asynchronously without blocking UI interactions.
  - Presents interactive controls for prompt, negative prompt, steps, CFG, denoise, seed, and batch size.
  - Displays thumbnail previews of completed generations with one-click layer/document application.
