// gavel-server (M5-lite): minimal HTTP server exposing decide-core as JSON.
//
// Endpoints:
//   POST /define    {"name","inputSchema","outputSchema","policy"?,"engine"?} -> {"ok":true}
//                   engine: "logistic" (default) or
//                   {"onnx":{"model":"...onnx","tokenizer":"...json"?,
//                            "labels":[...],"maxLength":512?,"temperature":1.0?}}
//   POST /train     {"question","examples":[{"input","label"}]}      -> {"ok":true}
//   POST /calibrate {"question","validation":[{"input","label"}]}    -> {"ok":true}
//   POST /ask       {"question","input"}                            -> Decision JSON
//   GET  /metrics?question=<name>                                    -> EngineMetrics JSON
//   GET  /questions                                                  -> {"questions":[...]}
//
// All errors are JSON {"error":"..."} with HTTP 400. Requests are handled
// sequentially; shared state is guarded by a Mutex. Binds 0.0.0.0:7575.

use decide_core::{extract_text, Engine, Policy, Registry};
use decide_onnx::{OnnxConfig, OnnxEngine};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::sync::Mutex;
use tiny_http::{Header, Request, Response, Server, StatusCode};

const BIND_ADDR: &str = "0.0.0.0:7575";
const MAX_BODY_BYTES: u64 = 16 * 1024 * 1024;

fn json_response(req: Request, status: u16, value: &Value) {
    let body =
        serde_json::to_string(value).unwrap_or_else(|_| r#"{"error":"encode failed"}"#.into());
    let header = Header::from_bytes(b"Content-Type", b"application/json").unwrap();
    let resp = Response::from_string(body)
        .with_header(header)
        .with_status_code(StatusCode(status));
    let _ = req.respond(resp);
}

fn ok(req: Request) {
    json_response(req, 200, &json!({"ok": true}));
}

fn bad(req: Request, msg: impl Into<String>) {
    json_response(req, 400, &json!({"error": msg.into()}));
}

fn read_json_body(req: &mut Request) -> Result<Value, String> {
    let mut body = String::new();
    req.as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_string(&mut body)
        .map_err(|e| format!("failed to read request body: {e}"))?;
    if body.len() as u64 > MAX_BODY_BYTES {
        return Err(format!("request body exceeds {MAX_BODY_BYTES} bytes"));
    }
    serde_json::from_str(&body).map_err(|e| format!("invalid JSON: {e}"))
}

fn field_str<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing or invalid string field \"{key}\""))
}

fn field_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or invalid numeric field \"{key}\""))
}

/// Parse an optional `policy` object into a `Policy`, defaulting when absent/null.
fn parse_policy(v: &Value) -> Result<Policy, String> {
    match v.get("policy") {
        None | Some(Value::Null) => Ok(Policy::default_policy()),
        Some(p) => {
            let act_above = field_num(p, "act_above")?;
            let review_low = field_num(p, "review_low")?;
            let review_high = field_num(p, "review_high")?;
            Policy::new(act_above, review_low, review_high)
                .map_err(|e| format!("invalid policy: {e}"))
        }
    }
}

/// Parse `{"model": "...onnx", "tokenizer": "...json"?, "labels": [...],
/// "maxLength": n?, "temperature": t?}` into an [`OnnxConfig`].
fn parse_onnx_config(v: &Value) -> Result<OnnxConfig, String> {
    let model = field_str(v, "model")
        .map_err(|_| "engine.onnx: missing required field \"model\" (path to .onnx file)")?
        .to_string();
    let labels: Vec<String> = v
        .get("labels")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "engine.onnx: missing required field \"labels\" (array of strings)".to_string()
        })?
        .iter()
        .map(|l| {
            l.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "engine.onnx: every label must be a string".to_string())
        })
        .collect::<Result<_, _>>()?;
    let mut cfg = OnnxConfig::new(model, labels).map_err(|e| e.to_string())?;
    if let Some(tok) = v.get("tokenizer").and_then(Value::as_str) {
        cfg.tokenizer_path = Some(tok.into());
    }
    if let Some(n) = v.get("maxLength").and_then(Value::as_u64) {
        cfg.max_length = n.max(1) as usize;
    }
    if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
        cfg.temperature = t;
    }
    Ok(cfg)
}

/// Convert `[{"input": <any>, "label": "<str>"}]` into (text, label) pairs.
fn parse_examples(v: &Value, key: &str) -> Result<Vec<(String, String)>, String> {
    let arr = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field \"{key}\""))?;
    if arr.is_empty() {
        return Err(format!("\"{key}\" must not be empty"));
    }
    arr.iter()
        .enumerate()
        .map(|(i, ex)| {
            let input = ex
                .get("input")
                .ok_or_else(|| format!("{key}[{i}]: missing field \"input\""))?;
            let label = ex
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{key}[{i}]: \"label\" must be a string"))?;
            Ok((extract_text(input), label.to_string()))
        })
        .collect()
}

/// Minimal percent-decoding for query parameter values.
fn percent_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut bytes = s.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let hi = bytes.next().unwrap_or(b'0');
            let lo = bytes.next().unwrap_or(b'0');
            let hex = |c: u8| (c as char).to_digit(16).unwrap_or(0) as u8;
            out.push((hex(hi) * 16 + hex(lo)) as char);
        } else if b == b'+' {
            out.push(' ');
        } else {
            out.push(b as char);
        }
    }
    out
}

fn query_param(url: &str, name: &str) -> Option<String> {
    url.split('?').nth(1).and_then(|q| {
        q.split('&').find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            let k = kv.next()?;
            let v = kv.next().unwrap_or("");
            (k == name).then(|| percent_decode(v))
        })
    })
}

fn handle(mut req: Request, registry: &Mutex<Registry>) {
    let method = req.method().as_str().to_string();
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();

    // Log every request method + path to stdout.
    println!("{method} {path}");
    let _ = std::io::stdout().flush();

    match (method.as_str(), path.as_str()) {
        ("POST", "/define") => match read_json_body(&mut req) {
            Err(e) => bad(req, e),
            Ok(body) => {
                let result = (|| -> Result<(), String> {
                    let name = field_str(&body, "name")?.to_string();
                    let input_schema = body
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let output_schema = body
                        .get("outputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let policy = parse_policy(&body)?;
                    // Engine selection. Logistic is the default; an "onnx"
                    // object loads an ONNX model via decide-onnx. If the model
                    // fails to load, warn on stderr and fall back to logistic
                    // — the server never refuses to serve a question.
                    let engine: Option<Box<dyn Engine>> = match body.get("engine") {
                        None | Some(serde_json::Value::String(_)) => match body.get("engine") {
                            Some(serde_json::Value::String(s)) if s != "logistic" => {
                                return Err(format!(
                                    "unknown engine \"{s}\"; expected \"logistic\" or {{\"onnx\": ...}}"
                                ))
                            }
                            _ => None,
                        },
                        Some(obj) => {
                            let onnx = obj.get("onnx").ok_or_else(|| {
                                "engine must be \"logistic\" or {\"onnx\": {...}}".to_string()
                            })?;
                            let cfg = parse_onnx_config(onnx)?;
                            match OnnxEngine::load(&cfg) {
                                Ok(e) => Some(Box::new(e)),
                                Err(err) => {
                                    eprintln!(
                                        "warn: onnx engine failed to load ({err}); \
                                         falling back to logistic for question \"{name}\""
                                    );
                                    None
                                }
                            }
                        }
                    };
                    let mut registry = registry
                        .lock()
                        .map_err(|e| format!("state lock poisoned: {e}"))?;
                    match engine {
                        Some(e) => registry.define_question_with_engine(
                            name,
                            input_schema,
                            output_schema,
                            policy,
                            e,
                        ),
                        None => registry.define_question(name, input_schema, output_schema, policy),
                    }
                    Ok(())
                })();
                match result {
                    Ok(()) => ok(req),
                    Err(e) => bad(req, e),
                }
            }
        },
        ("POST", "/train") => match read_json_body(&mut req) {
            Err(e) => bad(req, e),
            Ok(body) => {
                let result = (|| -> Result<(), String> {
                    let question = field_str(&body, "question")?.to_string();
                    let examples = parse_examples(&body, "examples")?;
                    registry
                        .lock()
                        .map_err(|e| format!("state lock poisoned: {e}"))?
                        .train(&question, &examples)
                        .map_err(|e| e.to_string())
                })();
                match result {
                    Ok(()) => ok(req),
                    Err(e) => bad(req, e),
                }
            }
        },
        ("POST", "/calibrate") => match read_json_body(&mut req) {
            Err(e) => bad(req, e),
            Ok(body) => {
                let result = (|| -> Result<(), String> {
                    let question = field_str(&body, "question")?.to_string();
                    let validation = parse_examples(&body, "validation")?;
                    registry
                        .lock()
                        .map_err(|e| format!("state lock poisoned: {e}"))?
                        .calibrate(&question, &validation)
                        .map_err(|e| e.to_string())
                })();
                match result {
                    Ok(()) => ok(req),
                    Err(e) => bad(req, e),
                }
            }
        },
        ("POST", "/ask") => match read_json_body(&mut req) {
            Err(e) => bad(req, e),
            Ok(body) => {
                let result = (|| -> Result<Value, String> {
                    let question = field_str(&body, "question")?.to_string();
                    let input = body.get("input").cloned().unwrap_or_else(|| json!({}));
                    let decision = registry
                        .lock()
                        .map_err(|e| format!("state lock poisoned: {e}"))?
                        .ask(&question, &input)
                        .map_err(|e| e.to_string())?;
                    serde_json::to_value(&decision).map_err(|e| format!("encode failed: {e}"))
                })();
                match result {
                    Ok(v) => json_response(req, 200, &v),
                    Err(e) => bad(req, e),
                }
            }
        },
        ("GET", "/metrics") => match query_param(&url, "question") {
            None => bad(req, "missing query parameter \"question\""),
            Some(question) => {
                let result = registry
                    .lock()
                    .map_err(|e| format!("state lock poisoned: {e}"))
                    .and_then(|reg| {
                        reg.metrics(&question)
                            .map_err(|e| e.to_string())
                            .and_then(|m| {
                                serde_json::to_value(&m).map_err(|e| format!("encode failed: {e}"))
                            })
                    });
                match result {
                    Ok(v) => json_response(req, 200, &v),
                    Err(e) => bad(req, e),
                }
            }
        },
        ("GET", "/questions") => {
            let names = registry
                .lock()
                .map(|reg| reg.question_names())
                .unwrap_or_default();
            json_response(req, 200, &json!({"questions": names}));
        }
        _ => bad(req, format!("unknown route: {method} {path}")),
    }
}

fn main() {
    let registry = Registry::new(json!({}), 0.0).expect("mock fallback registry");
    let registry = Mutex::new(registry);

    let server = Server::http(BIND_ADDR).unwrap_or_else(|e| {
        eprintln!("gavel-server: failed to bind {BIND_ADDR}: {e}");
        std::process::exit(1);
    });
    println!("gavel-server listening on {BIND_ADDR}");
    let _ = std::io::stdout().flush();

    for req in server.incoming_requests() {
        handle(req, &registry);
    }
}
