use std::io::{Read, Write};
use std::net::TcpListener;

use serde_json::json;

use crate::{Client, Lora, Mode, Output, Request, build_workflow, model_choices, validate_catalog};

fn valid_req() -> Request {
    Request {
        checkpoint: "v1-5-pruned.safetensors".into(),
        prompt: "oil painting of a mountain landscape".into(),
        negative: "blurry, low quality".into(),
        ..Request::default()
    }
}

fn mock_catalog() -> serde_json::Value {
    json!({
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["v1-5-pruned.safetensors", "sd_xl_base_1.0.safetensors"]]}},
            "output": ["MODEL", "CLIP", "VAE"]
        },
        "CLIPTextEncode": {
            "input": {"required": {"clip": ["CLIP"], "text": ["STRING", {"multiline": true}]}},
            "output": ["CONDITIONING"]
        },
        "EmptyLatentImage": {
            "input": {"required": {"width": ["INT", {"min": 16, "max": 8192}], "height": ["INT", {"min": 16, "max": 8192}], "batch_size": ["INT", {"min": 1, "max": 64}]}},
            "output": ["LATENT"]
        },
        "KSampler": {
            "input": {"required": {
                "model": ["MODEL"], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"], "seed": ["INT", {"min": 0, "max": 9007199254740991i64}],
                "steps": ["INT", {"min": 1, "max": 1000}], "cfg": ["FLOAT", {"min": 0.0, "max": 100.0}],
                "sampler_name": [["euler", "dpmpp_2m"]], "scheduler": [["normal", "karras"]],
                "denoise": ["FLOAT", {"min": 0.0, "max": 1.0}]
            }},
            "output": ["LATENT"]
        },
        "VAEDecode": {
            "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
            "output": ["IMAGE"]
        },
        "VAEEncode": {
            "input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}},
            "output": ["LATENT"]
        },
        "LoadImage": {
            "input": {"required": {"image": ["STRING"]}},
            "output": ["IMAGE", "MASK"]
        },
        "ImageToMask": {
            "input": {"required": {"image": ["IMAGE"], "channel": [["red", "green", "blue", "alpha"]]}},
            "output": ["MASK"]
        },
        "SetLatentNoiseMask": {
            "input": {"required": {"samples": ["LATENT"], "mask": ["MASK"]}},
            "output": ["LATENT"]
        },
        "RepeatLatentBatch": {
            "input": {"required": {"samples": ["LATENT"], "amount": ["INT", {"min": 1, "max": 64}]}},
            "output": ["LATENT"]
        },
        "UpscaleModelLoader": {
            "input": {"required": {"model_name": [["4x_NMKD-Superscale-SP_178000_G.pth", "4x-UltraSharp.pth"]]}},
            "output": ["UPSCALE_MODEL"]
        },
        "ImageUpscaleWithModel": {
            "input": {"required": {"upscale_model": ["UPSCALE_MODEL"], "image": ["IMAGE"]}},
            "output": ["IMAGE"]
        },
        "ImageScaleBy": {
            "input": {"required": {"image": ["IMAGE"], "upscale_method": [["lanczos", "bicubic"]], "scale_by": ["FLOAT", {"min": 0.01, "max": 8.0}]}},
            "output": ["IMAGE"]
        },
        "LoraLoader": {
            "input": {"required": {
                "model": ["MODEL"], "clip": ["CLIP"],
                "lora_name": [["detail_tweaker.safetensors"]],
                "strength_model": ["FLOAT", {"min": -10.0, "max": 10.0}],
                "strength_clip": ["FLOAT", {"min": -10.0, "max": 10.0}]
            }},
            "output": ["MODEL", "CLIP"]
        },
        "SaveImage": {
            "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}},
            "output": []
        }
    })
}

#[test]
fn test_request_validation_defaults() {
    let req = valid_req();
    assert!(req.validate().is_ok());
}

#[test]
fn test_request_adversarial_rejections() {
    let mut req = valid_req();
    req.width = 510; // not multiple of 8
    assert!(req.validate().is_err());

    req = valid_req();
    req.width = 32; // < 64
    assert!(req.validate().is_err());

    req = valid_req();
    req.height = 9000; // > 8192
    assert!(req.validate().is_err());

    req = valid_req();
    req.batch = 0;
    assert!(req.validate().is_err());

    req = valid_req();
    req.batch = 9; // > 8
    assert!(req.validate().is_err());

    req = valid_req();
    req.width = 8192;
    req.height = 8192;
    req.batch = 2; // > 33_554_432 pixels
    assert!(req.validate().is_err());

    req = valid_req();
    req.steps = 0;
    assert!(req.validate().is_err());

    req = valid_req();
    req.steps = 151;
    assert!(req.validate().is_err());

    req = valid_req();
    req.cfg = -1.0;
    assert!(req.validate().is_err());

    req = valid_req();
    req.cfg = f64::NAN;
    assert!(req.validate().is_err());

    req = valid_req();
    req.cfg = f64::INFINITY;
    assert!(req.validate().is_err());

    req = valid_req();
    req.denoise = 1.01;
    assert!(req.validate().is_err());

    req = valid_req();
    req.scale = 0.05;
    assert!(req.validate().is_err());

    req = valid_req();
    req.seed = 9_007_199_254_740_992; // > JSON safe int
    assert!(req.validate().is_err());

    req = valid_req();
    req.checkpoint = "".into();
    assert!(req.validate().is_err());

    req = valid_req();
    req.checkpoint = "bad\0name".into();
    assert!(req.validate().is_err());

    req = valid_req();
    req.loras = vec![Lora { name: "lora1".into(), strength: 5.0 }]; // strength > 4.0
    assert!(req.validate().is_err());

    req = valid_req();
    req.mode = Mode::Custom;
    req.workflow = Some(json!({
        "1": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}}
    }));
    assert!(req.validate().is_err()); // cycle detected

    req = valid_req();
    req.workflow = Some(json!({"not": "custom"}));
    assert!(req.validate().is_err());
}

#[test]
fn test_workflow_generation_stock_modes() {
    // Generate
    let req = valid_req();
    let wf = build_workflow(&req, None, None).expect("generate workflow");
    assert!(wf.get("checkpoint").is_some());
    assert!(wf.get("positive").is_some());
    assert!(wf.get("negative").is_some());
    assert!(wf.get("latent").is_some());
    assert!(wf.get("sample").is_some());
    assert!(wf.get("decode").is_some());
    assert!(wf.get("save").is_some());

    // Generate with LoRAs
    let mut req_lora = valid_req();
    req_lora.loras = vec![Lora { name: "detail_tweaker.safetensors".into(), strength: 0.8 }];
    let wf_lora = build_workflow(&req_lora, None, None).expect("lora workflow");
    assert!(wf_lora.get("lora_0").is_some());

    // Refine
    let mut req_refine = valid_req();
    req_refine.mode = Mode::Refine;
    assert!(build_workflow(&req_refine, None, None).is_err()); // missing input
    let wf_refine = build_workflow(&req_refine, Some("photo.png"), None).expect("refine workflow");
    assert!(wf_refine.get("image").is_some());
    assert!(wf_refine.get("encode").is_some());
    assert!(wf_refine.get("latent").is_none());

    // Inpaint
    let mut req_inpaint = valid_req();
    req_inpaint.mode = Mode::Inpaint;
    assert!(build_workflow(&req_inpaint, Some("photo.png"), None).is_err()); // missing mask
    let wf_inpaint = build_workflow(&req_inpaint, Some("photo.png"), Some("mask.png")).expect("inpaint workflow");
    assert!(wf_inpaint.get("mask_image").is_some());
    assert!(wf_inpaint.get("mask").is_some());
    assert!(wf_inpaint.get("masked").is_some());

    // Upscale
    let mut req_upscale = valid_req();
    req_upscale.mode = Mode::Upscale;
    req_upscale.upscaler = "4x-UltraSharp.pth".into();
    let wf_upscale = build_workflow(&req_upscale, Some("small.png"), None).expect("upscale workflow");
    assert!(wf_upscale.get("upscaler").is_some());
    assert!(wf_upscale.get("upscale").is_some());
    assert!(wf_upscale.get("scale").is_some());

    // Upscale rejected when no upscaler
    let mut req_no_up = valid_req();
    req_no_up.mode = Mode::Upscale;
    req_no_up.upscaler = "".into();
    assert!(build_workflow(&req_no_up, Some("small.png"), None).is_err());
}

#[test]
fn test_catalog_validation() {
    let catalog = mock_catalog();

    let choices = model_choices(&catalog, "CheckpointLoaderSimple", "ckpt_name");
    assert_eq!(choices, vec!["v1-5-pruned.safetensors", "sd_xl_base_1.0.safetensors"]);

    let req = valid_req();
    let wf = build_workflow(&req, None, None).unwrap();
    assert!(validate_catalog(&wf, &catalog).is_ok());

    // Unlisted checkpoint fails
    let mut req_bad = valid_req();
    req_bad.checkpoint = "nonexistent_model.safetensors".into();
    let wf_bad = build_workflow(&req_bad, None, None).unwrap();
    assert!(validate_catalog(&wf_bad, &catalog).is_err());

    // Unlisted sampler fails
    let mut req_sampler = valid_req();
    req_sampler.sampler = "unknown_sampler".into();
    let wf_sampler = build_workflow(&req_sampler, None, None).unwrap();
    assert!(validate_catalog(&wf_sampler, &catalog).is_err());
}

#[test]
fn test_client_url_validation() {
    assert!(Client::new("http://127.0.0.1:8188").is_ok());
    assert!(Client::new("http://localhost:8188").is_ok());
    assert!(Client::new("https://remote-comfy.example.com").is_ok());

    // Unencrypted non-loopback rejected
    assert!(Client::new("http://remote-comfy.example.com").is_err());

    // Forbidden credentials, query, fragment
    assert!(Client::new("http://user:pass@127.0.0.1:8188").is_err());
    assert!(Client::new("http://127.0.0.1:8188?auth=token").is_err());
    assert!(Client::new("http://127.0.0.1:8188#fragment").is_err());
    assert!(Client::new("ftp://127.0.0.1:8188").is_err());
}

fn read_http_request(stream: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut content_length = None;
    let mut header_end = None;

    loop {
        let n = stream.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);

        if header_end.is_none()
            && let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n")
        {
            header_end = Some(pos + 4);
            let headers_str = String::from_utf8_lossy(&buf[..pos]);
            for line in headers_str.lines() {
                if line.to_ascii_lowercase().starts_with("content-length:") {
                    let val = line.split(':').nth(1).unwrap_or("0").trim();
                    content_length = val.parse::<usize>().ok();
                }
            }
        }

        if let Some(h_end) = header_end {
            let cl = content_length.unwrap_or(0);
            if buf.len() >= h_end + cl {
                break;
            }
        }
    }

    String::from_utf8_lossy(&buf).to_string()
}

#[test]
fn test_fake_http_server_end_to_end() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let addr = listener.local_addr().expect("local addr");
    let endpoint = format!("http://127.0.0.1:{}", addr.port());

    let server_thread = std::thread::spawn(move || {
        let catalog_bytes = serde_json::to_vec(&mock_catalog()).unwrap();

        for _ in 0..8 {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let req_str = read_http_request(&mut stream);

            if req_str.starts_with("GET /object_info ") {
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", catalog_bytes.len());
                stream.write_all(resp.as_bytes()).unwrap();
                stream.write_all(&catalog_bytes).unwrap();
            } else if req_str.starts_with("POST /upload/image ") {
                let body = json!({"name": "uploaded_test.png", "subfolder": "sub", "type": "input"});
                let body_bytes = serde_json::to_vec(&body).unwrap();
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body_bytes.len());
                stream.write_all(resp.as_bytes()).unwrap();
                stream.write_all(&body_bytes).unwrap();
            } else if req_str.starts_with("POST /prompt ") {
                let body = json!({"prompt_id": "test-prompt-uuid-123"});
                let body_bytes = serde_json::to_vec(&body).unwrap();
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body_bytes.len());
                stream.write_all(resp.as_bytes()).unwrap();
                stream.write_all(&body_bytes).unwrap();
            } else if req_str.starts_with("GET /history/test-prompt-uuid-123 ") {
                let body = json!({
                    "test-prompt-uuid-123": {
                        "status": {"status_str": "success", "completed": true, "messages": []},
                        "outputs": {
                            "save": {
                                "images": [
                                    {"filename": "out_0001.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    }
                });
                let body_bytes = serde_json::to_vec(&body).unwrap();
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body_bytes.len());
                stream.write_all(resp.as_bytes()).unwrap();
                stream.write_all(&body_bytes).unwrap();
            } else if req_str.starts_with("GET /view?") {
                let png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...";
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", png_bytes.len());
                stream.write_all(resp.as_bytes()).unwrap();
                stream.write_all(png_bytes).unwrap();
            } else if req_str.starts_with("GET /queue ") {
                let body = json!({
                    "queue_running": [],
                    "queue_pending": [
                        [0, "test-prompt-uuid-123", {}, {}, []]
                    ]
                });
                let body_bytes = serde_json::to_vec(&body).unwrap();
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body_bytes.len());
                stream.write_all(resp.as_bytes()).unwrap();
                stream.write_all(&body_bytes).unwrap();
            } else if req_str.starts_with("POST /queue ") {
                let resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                stream.write_all(resp.as_bytes()).unwrap();
            }
        }
    });

    let client = Client::new(&endpoint).expect("create client");

    // 1. Catalog
    let cat = client.catalog().expect("get catalog");
    assert!(cat.is_object());

    // 2. Upload PNG
    let fake_png = b"\x89PNG\r\n\x1a\nfake_pixel_payload";
    let remote_name = client.upload_png(fake_png, "sample.png").expect("upload png");
    assert_eq!(remote_name, "sub/uploaded_test.png");

    // 3. Submit
    let req = valid_req();
    let prompt_id = client.submit(&req, None, None).expect("submit request");
    assert_eq!(prompt_id, "test-prompt-uuid-123");

    // 4. History
    let history = client.history(&prompt_id).expect("get history");
    assert_eq!(history, Some(vec![Output { filename: "out_0001.png".into(), subfolder: "".into(), kind: "output".into() }]));

    // 5. Download
    let image_bytes = client.download(&Output { filename: "out_0001.png".into(), subfolder: "".into(), kind: "output".into() }).expect("download image");
    assert!(image_bytes.starts_with(b"\x89PNG"));

    // 6. Cancel queued ID succeeds
    assert!(client.cancel(&prompt_id).is_ok());

    let _ = server_thread.join();

    // 7. Cancel running job returns Err (refuses global interrupt)
    let running_listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let running_addr = running_listener.local_addr().expect("local addr");
    let running_endpoint = format!("http://127.0.0.1:{}", running_addr.port());
    let running_server = std::thread::spawn(move || {
        let (mut stream, _) = running_listener.accept().expect("accept connection");
        let _ = read_http_request(&mut stream);
        let body = json!({
            "queue_running": [
                [0, "running-prompt-id", {}, {}, []]
            ],
            "queue_pending": []
        });
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body_bytes.len());
        stream.write_all(resp.as_bytes()).unwrap();
        stream.write_all(&body_bytes).unwrap();
    });

    let running_client = Client::new(&running_endpoint).expect("create client");
    let cancel_res = running_client.cancel("running-prompt-id");
    assert!(cancel_res.is_err());
    let _ = running_server.join();
}
