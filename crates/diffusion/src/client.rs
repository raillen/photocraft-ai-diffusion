use std::time::Duration;

use serde_json::{Value, json};
use url::{Host, Url};

use crate::{Error, MAX_IMAGE_BYTES, MAX_JSON_BYTES, Output, Request, bounded_text, build_workflow, image_name, invalid, validate_catalog};

/// Blocking native client. Run network calls off the UI thread.
#[derive(Clone)]
pub struct Client {
    endpoint: Url,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(endpoint: &str) -> Result<Self, Error> {
        bounded_text(endpoint, "endpoint", 4096, true)?;
        let mut endpoint = Url::parse(endpoint).map_err(|_| invalid("invalid endpoint URL"))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(invalid("endpoint requires HTTP(S), a host, and no credentials, query or fragment"));
        }
        if endpoint.scheme() == "http" {
            let loopback = match endpoint.host() {
                Some(Host::Ipv4(ip)) => ip == std::net::Ipv4Addr::LOCALHOST,
                Some(Host::Ipv6(ip)) => ip == std::net::Ipv6Addr::LOCALHOST,
                Some(Host::Domain(name)) => name.eq_ignore_ascii_case("localhost"),
                None => false,
            };
            if !loopback {
                return Err(invalid("remote servers require HTTPS"));
            }
            // Avoid trusting DNS for the only permitted cleartext hostname.
            if endpoint.host_str().is_some_and(|h| h.eq_ignore_ascii_case("localhost")) {
                endpoint.set_host(Some("127.0.0.1")).map_err(|_| invalid("invalid loopback host"))?;
            }
        }
        let path = format!("{}/", endpoint.path().trim_end_matches('/'));
        endpoint.set_path(&path);
        let agent = ureq::Agent::config_builder()
            .tls_config(ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build())
            .timeout_global(Some(Duration::from_secs(20)))
            .max_redirects(0)
            .build()
            .new_agent();
        Ok(Self { endpoint, agent })
    }

    fn route(&self, path: &str) -> Result<Url, Error> {
        self.endpoint.join(path).map_err(|_| invalid("invalid server route"))
    }

    fn read(mut response: ureq::http::Response<ureq::Body>, cap: usize) -> Result<Vec<u8>, Error> {
        if !response.status().is_success() {
            return Err(Error::Protocol(format!("unexpected HTTP status {} (redirects disabled)", response.status())));
        }
        Ok(response.body_mut().with_config().limit(cap as u64).read_to_vec()?)
    }

    fn get_json(&self, url: Url) -> Result<Value, Error> {
        let response = self.agent.get(url.as_str()).header("User-Agent", "PhotoCraft-diffusion").call()?;
        Ok(serde_json::from_slice(&Self::read(response, MAX_JSON_BYTES)?)?)
    }

    fn post_json(&self, route: &str, value: &Value) -> Result<Vec<u8>, Error> {
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() > MAX_JSON_BYTES {
            return Err(invalid("request exceeds JSON limit"));
        }
        let response = self
            .agent
            .post(self.route(route)?.as_str())
            .header("User-Agent", "PhotoCraft-diffusion")
            .header("Content-Type", "application/json")
            .send(bytes.as_slice())?;
        Self::read(response, MAX_JSON_BYTES)
    }

    /// Raw live object_info. No model names are inferred or selected automatically.
    pub fn catalog(&self) -> Result<Value, Error> {
        let result = self.get_json(self.route("object_info")?)?;
        if !result.is_object() {
            return Err(Error::Protocol("object_info is not an object".into()));
        }
        Ok(result)
    }

    /// Upload a PNG without overwriting existing server inputs. Returns its relative remote name.
    pub fn upload_png(&self, bytes: &[u8], filename: &str) -> Result<String, Error> {
        if bytes.len() > MAX_IMAGE_BYTES || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err(invalid("upload must have a PNG signature and be at most 64 MiB"));
        }
        if filename.is_empty()
            || filename.len() > 200
            || !filename.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
            || !filename.to_ascii_lowercase().ends_with(".png")
            || filename.starts_with('.')
        {
            return Err(invalid("filename must be a simple ASCII .png basename"));
        }
        // Choose a delimiter absent from the payload; never let PNG bytes inject another part.
        let mut boundary = "PhotoCraftMultipart".to_owned();
        while bytes.windows(boundary.len()).any(|part| part == boundary.as_bytes()) {
            boundary.push('x');
        }
        let mut body =
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").into_bytes();
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\nfalse\r\n--{boundary}--\r\n").as_bytes());
        let response = self
            .agent
            .post(self.route("upload/image")?.as_str())
            .header("User-Agent", "PhotoCraft-diffusion")
            .header("Content-Type", format!("multipart/form-data; boundary={boundary}"))
            .send(body.as_slice())?;
        let result: Value = serde_json::from_slice(&Self::read(response, MAX_JSON_BYTES)?)?;
        let name = result.get("name").and_then(Value::as_str).ok_or_else(|| Error::Protocol("upload response has no name".into()))?;
        let folder = result.get("subfolder").and_then(Value::as_str).unwrap_or("");
        let remote = if folder.is_empty() { name.into() } else { format!("{folder}/{name}") };
        image_name(Some(&remote), "uploaded")
    }

    /// Re-fetches object_info and checks every used node, enum, model and connection before POST.
    pub fn submit(&self, request: &Request, input: Option<&str>, mask: Option<&str>) -> Result<String, Error> {
        let graph = build_workflow(request, input, mask)?;
        validate_catalog(&graph, &self.catalog()?)?;
        let bytes = self.post_json("prompt", &json!({"prompt": graph}))?;
        let result: Value = serde_json::from_slice(&bytes)?;
        if result.get("error").is_some_and(|v| !v.is_null()) || result.get("node_errors").is_some_and(|v| v.as_object().is_none_or(|v| !v.is_empty())) {
            return Err(Error::Job("server rejected the workflow".into()));
        }
        let id = result.get("prompt_id").and_then(Value::as_str).ok_or_else(|| Error::Protocol("prompt response has no prompt_id".into()))?;
        prompt_id(id)?;
        Ok(id.into())
    }

    /// None means absent or still executing. Completed jobs may have zero image outputs.
    pub fn history(&self, id: &str) -> Result<Option<Vec<Output>>, Error> {
        prompt_id(id)?;
        let mut url = self.route("history/")?;
        url.path_segments_mut().map_err(|_| invalid("invalid history URL"))?.pop_if_empty().push(id);
        let history = self.get_json(url)?;
        let map = history.as_object().ok_or_else(|| Error::Protocol("history is not an object".into()))?;
        let Some(entry) = map.get(id) else {
            return Ok(None);
        };
        let status = entry.get("status").ok_or_else(|| Error::Protocol("history entry missing status".into()))?;
        if status.get("status_str").and_then(Value::as_str).is_some_and(|s| matches!(s, "error" | "failed" | "interrupted"))
            || status.get("messages").and_then(Value::as_array).is_some_and(|messages| {
                messages.iter().any(|m| m.get(0).and_then(Value::as_str).is_some_and(|s| matches!(s, "execution_error" | "execution_interrupted")))
            })
        {
            return Err(Error::Job("execution failed or was interrupted; inspect server history for details".into()));
        }
        if status.get("completed").and_then(Value::as_bool) != Some(true) {
            return Ok(None);
        }
        let outputs = entry.get("outputs").and_then(Value::as_object).ok_or_else(|| Error::Protocol("completed history missing outputs".into()))?;
        let mut images = Vec::new();
        for output in outputs.values() {
            if let Some(items) = output.get("images") {
                let items = items.as_array().ok_or_else(|| Error::Protocol("images is not an array".into()))?;
                for item in items {
                    let image: Output = serde_json::from_value(item.clone())?;
                    validate_output(&image)?;
                    images.push(image);
                    if images.len() > 1024 {
                        return Err(Error::Protocol("too many image outputs".into()));
                    }
                }
            }
        }
        Ok(Some(images))
    }

    pub fn download(&self, output: &Output) -> Result<Vec<u8>, Error> {
        validate_output(output)?;
        let mut url = self.route("view")?;
        url.query_pairs_mut().append_pair("filename", &output.filename).append_pair("subfolder", &output.subfolder).append_pair("type", &output.kind);
        Self::read(self.agent.get(url.as_str()).header("User-Agent", "PhotoCraft-diffusion").call()?, MAX_IMAGE_BYTES)
    }

    /// Deletes only this queued ID. Never calls the global /interrupt endpoint.
    /// A running job returns Err: ComfyUI cannot atomically interrupt a particular ID.
    pub fn cancel(&self, id: &str) -> Result<(), Error> {
        prompt_id(id)?;
        let queue = self.get_json(self.route("queue")?)?;
        let running = queue.get("queue_running").and_then(Value::as_array).ok_or_else(|| Error::Protocol("queue_running missing".into()))?;
        if running.iter().any(|item| item.get(1).and_then(Value::as_str) == Some(id)) {
            return Err(Error::Job("job is running; refusing global interrupt on a shared server".into()));
        }
        self.post_json("queue", &json!({"delete": [id]}))?;
        Ok(())
    }
}

fn prompt_id(id: &str) -> Result<(), Error> {
    if id.is_empty() || id.len() > 128 || !id.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_')) {
        return Err(invalid("invalid prompt id"));
    }
    Ok(())
}

fn validate_output(output: &Output) -> Result<(), Error> {
    image_name(Some(&output.filename), "output filename")?;
    bounded_text(&output.subfolder, "output subfolder", 2048, false)?;
    if !output.subfolder.is_empty() {
        image_name(Some(&output.subfolder), "output subfolder")?;
    }
    if !matches!(output.kind.as_str(), "input" | "output" | "temp") {
        return Err(Error::Protocol("invalid output type".into()));
    }
    Ok(())
}
