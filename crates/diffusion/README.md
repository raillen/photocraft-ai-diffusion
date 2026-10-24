# photocraft-diffusion

Clean-room stock ComfyUI workflow builder and bounded native HTTP client. No upstream GPL code.

## Scope and Architecture Limitations

This crate targets automated integration with local and remote ComfyUI HTTP endpoints. It is **not** a universal graph synthesizer for every model architecture or custom node ecosystem.

### Supported Stock Modes
- **Generate (txt2img)**: `CheckpointLoaderSimple` + optional `LoraLoader` chain + `CLIPTextEncode` (pos/neg) + `EmptyLatentImage` + `KSampler` + `VAEDecode` + `SaveImage`.
- **Refine (img2img)**: `LoadImage` + `VAEEncode` + `KSampler` (with configured denoise) + `VAEDecode` + `SaveImage`.
- **Inpaint**: `LoadImage` (source) + `LoadImage` (mask) + `ImageToMask` (red channel) + `VAEEncode` + `SetLatentNoiseMask` + `KSampler` + `VAEDecode` + `SaveImage`.
- **Upscale**: `LoadImage` + `UpscaleModelLoader` + `ImageUpscaleWithModel` + `ImageScaleBy` (lanczos post-scale) + `SaveImage`. Requires an explicit upscaler model name; rejected if empty.
- **Custom**: Validated API-format graph (JSON node map with bounded node count, depth, and typed link arrays).

### Unsupported Architectures
- Advanced multi-text-encoder pipelines (Flux dual CLIP, SD3 triple CLIP, Wan 2.x, Qwen) are not synthesized by stock modes. Use `Mode::Custom` with an authored API graph.
- ControlNet, IP-Adapter, AnimateDiff, and third-party custom node packs are not automatically wired by stock modes.

### Inpainting Mask Convention
The stock inpainting workflow extracts mask opacity from the red channel (`ImageToMask` with `channel: "red"`). Red represents editable pixels, black represents preserved pixels.

### Shared-Server Cancellation & Race Conditions
ComfyUI's native API provides:
1. `POST /queue` with `{"delete": [id]}` to remove pending jobs from `queue_pending`.
2. `POST /interrupt` to abort the currently running execution. **`/interrupt` takes no prompt ID** and unconditionally halts whatever job is executing on the server.

`Client::cancel(prompt_id)` inspects `/queue`:
- If the ID is found in `queue_pending`, it deletes only that queued job.
- If the ID is already executing in `queue_running`, `cancel()` refuses to send a global `/interrupt` and returns an error explaining that interrupting would race with other clients on a shared server.

### Bounded Limits and Security
- **Timeouts**: 20 seconds total per request, redirects disabled.
- **Payload Caps**: 16 MiB for JSON graphs and API responses; 64 MiB for PNG uploads and downloads.
- **URL Policy**: Cleartext HTTP is restricted strictly to loopback interfaces (`127.0.0.1`, `localhost`, `[::1]`). Remote hosts require HTTPS. URLs with credentials, queries, or fragments are rejected.
- **Request Bounds**:
  - Width and height: 64..=8192 (must be multiples of 8).
  - Total pixels: at most 32 MiPixels across all batch items.
  - Steps: 1..=150.
  - Batch: 1..=8.
  - Seed: 0..=9,007,199,254,740,991 (IEEE 754 exact integer limit).
  - LoRAs: maximum 16 entries, strengths bounded within -4.0..=4.0.
- **Validation**: No default model or sampler names are assumed. `Client::submit` fetches live `/object_info` from the server and validates all node classes, required fields, and choices (checkpoint, sampler, scheduler, LoRAs, upscaler) prior to submission.
