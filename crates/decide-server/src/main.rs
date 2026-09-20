// gavel-server (M5-lite): minimal HTTP server exposing decide-core as JSON.
//
// Endpoints:
//   POST /define    {"name","inputSchema","outputSchema","policy"?,"engine"?} -> {"ok":true}
//                   engine: "logistic" (default),
//                   {"onnx":{"model":"...onnx","tokenizer":"...json"?,
//                            "labels":[...],"maxLength":512?,"temperature":1.0?}}, or
//                   {"tfidf":{"model":"...json"}} (native TF-IDF + logistic artifact)
//                   policy may include "abstain_energy_above": abstain when the
//                   engine's energy score exceeds it (TF-IDF engine only).
//   POST /train     {"question","examples":[{"input","label"}]}      -> {"ok":true}
//   POST /calibrate {"question","validation":[{"input","label"}]}    -> {"ok":true}
//   POST /ask       {"question","input"?,"state"?}                    -> Decision JSON
//                   {"question","input"?,"state"?,"questions":[...]}  -> {"answers":[...]} (batch)
//                   "state" is an object of string fields rendered as
//                   "key: value" lines ahead of the text input; "questions"
//                   items are {"kind":"choice","id"} (default kind) or
//                   {"kind":"noul","id","statement","class"} answered
//                   independently in order against the same question.
//   GET  /metrics?question=<name>                                    -> EngineMetrics JSON
//   GET  /questions                                                  -> {"questions":[...]}
//
// All errors are JSON {"error":"..."} with HTTP 400. Requests are handled
// sequentially; shared state is guarded by a Mutex. Binds 0.0.0.0:7575.

use decide_core::{extract_text, render_state, Engine, NoulQuestion, Policy, Registry};
use decide_onnx::{OnnxConfig, OnnxEngine};
use decide_tfidf::TfidfEngine;
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
/// Accepts the confidence thresholds plus the optional energy-based
/// abstention gate `abstain_energy_above` (any finite value; abstain when the
/// engine's energy score exceeds it).
fn parse_policy(v: &Value) -> Result<Policy, String> {
    match v.get("policy") {
        None | Some(Value::Null) => Ok(Policy::default_policy()),
        Some(p) => {
            let act_above = field_num(p, "act_above")?;
            let review_low = field_num(p, "review_low")?;
            let review_high = field_num(p, "review_high")?;
            let mut policy = Policy::new(act_above, review_low, review_high)
                .map_err(|e| format!("invalid policy: {e}"))?;
            if let Some(t) = p.get("abstain_energy_above") {
                let t = t.as_f64().filter(|t| t.is_finite()).ok_or_else(|| {
                    "\"abstain_energy_above\" must be a finite number".to_string()
                })?;
                policy = policy.with_abstain_energy(t);
            }
            Ok(policy)
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

/// Compose the effective engine input for /ask.
///
/// Accepts the legacy `"input"` (string, object, anything [`extract_text`]
/// handles) plus an optional top-level `"state"` object of string fields:
///
/// - state renders as `"key: value"` lines (keys sorted, via
///   [`render_state`]), then the text input follows after a newline;
/// - state alone (no text) renders just the lines;
/// - no state behaves exactly as before.
///
/// The engine receives a single string either way, so every engine
/// (logistic, tfidf, onnx, mock) gets structured-state support uniformly.
/// Field names become ordinary tokens in that string: this is a transparent
/// encoding, not schema-aware featurization, and very short values let the
/// field names dominate the token stream — callers should keep the
/// substantive content in the values.
fn compose_input(body: &Value) -> Result<Value, String> {
    let raw = body.get("input").cloned().unwrap_or_else(|| json!({}));
    match body.get("state") {
        None | Some(Value::Null) => Ok(raw),
        Some(state) => {
            if !state.is_object() {
                return Err("\"state\" must be an object of string fields".to_string());
            }
            let rendered = render_state(state);
            let text = extract_text(&raw);
            let combined = match (rendered.is_empty(), text.is_empty()) {
                (true, _) => text,
                (false, true) => rendered,
                (false, false) => format!("{rendered}\n{text}"),
            };
            Ok(Value::String(combined))
        }
    }
}

/// Prepend "id"/"kind" to a serialized answer so both audiences see which
/// question an answer belongs to before the payload.
fn tagged_answer(id: &str, kind: &str, v: Value) -> Value {
    let mut map = serde_json::Map::with_capacity(8);
    map.insert("id".to_string(), json!(id));
    map.insert("kind".to_string(), json!(kind));
    if let Value::Object(fields) = v {
        for (k, val) in fields {
            map.insert(k, val);
        }
    }
    Value::Object(map)
}

/// Answer one item of a batch `"questions"` array against the registered
/// question `name`. Items share the question's engine, policy, energy gate,
/// and input; each item is an independent *view* of that engine's output:
///
/// - `{"kind": "choice", "id": "..."}` — the standard choice decision
///   (`kind` may be omitted; it defaults to `"choice"`).
/// - `{"kind": "noul", "id": "...", "statement": "...", "class": "..."}` —
///   yes/no answer as a one-vs-rest projection of the class distribution.
///
/// Answers are returned in array order (see [`answer_batch`]).
fn answer_one(
    registry: &Registry,
    name: &str,
    item: &Value,
    input: &Value,
    idx: usize,
) -> Result<Value, String> {
    let tag = format!("questions[{idx}]");
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{tag}: missing non-empty \"id\""))?;
    let kind = item.get("kind").and_then(Value::as_str).unwrap_or("choice");
    match kind {
        "choice" => {
            let d = registry
                .ask(name, input)
                .map_err(|e| format!("{tag}: {e}"))?;
            let v = serde_json::to_value(&d).map_err(|e| format!("encode failed: {e}"))?;
            Ok(tagged_answer(id, "choice", v))
        }
        "noul" => {
            let nq: NoulQuestion = serde_json::from_value(item.clone())
                .map_err(|e| format!("{tag}: invalid noul item: {e}"))?;
            if nq.statement.trim().is_empty() {
                return Err(format!("{tag}: noul \"statement\" must not be empty"));
            }
            if nq.class.trim().is_empty() {
                return Err(format!("{tag}: noul \"class\" must not be empty"));
            }
            let d = registry
                .noul_answer(name, &nq, input)
                .map_err(|e| format!("{tag}: {e}"))?;
            let v = serde_json::to_value(&d).map_err(|e| format!("encode failed: {e}"))?;
            Ok(tagged_answer(id, "noul", v))
        }
        other => Err(format!(
            "{tag}: unknown kind {other:?}; expected \"choice\" or \"noul\""
        )),
    }
}

/// Answer a batch `"questions"` array, returning `{"answers": [...]}` in
/// order. All-or-nothing: any item that fails to parse or answer fails the
/// whole request with a 400, so callers never have to guess which answers
/// are missing.
fn answer_batch(
    registry: &Registry,
    name: &str,
    items: &Value,
    input: &Value,
) -> Result<Value, String> {
    let arr = items
        .as_array()
        .ok_or_else(|| "\"questions\" must be an array of question items".to_string())?;
    if arr.is_empty() {
        return Err("\"questions\" must not be empty".to_string());
    }
    let mut answers = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        answers.push(answer_one(registry, name, item, input, i)?);
    }
    Ok(json!({"answers": answers}))
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
                    // object loads an ONNX model via decide-onnx, and a "tfidf"
                    // object loads a native TF-IDF + logistic artifact via
                    // decide-tfidf. If a model fails to load, warn on stderr
                    // and fall back to logistic — the server never refuses to
                    // serve a question.
                    let engine: Option<Box<dyn Engine>> = match body.get("engine") {
                        None | Some(serde_json::Value::String(_)) => match body.get("engine") {
                            Some(serde_json::Value::String(s)) if s != "logistic" => {
                                return Err(format!(
                                    "unknown engine \"{s}\"; expected \"logistic\", {{\"onnx\": ...}} or {{\"tfidf\": ...}}"
                                ))
                            }
                            _ => None,
                        },
                        Some(obj) => {
                            if let Some(onnx) = obj.get("onnx") {
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
                            } else if let Some(tfidf) = obj.get("tfidf") {
                                let model = tfidf
                                    .get("model")
                                    .and_then(Value::as_str)
                                    .filter(|s| !s.is_empty())
                                    .ok_or_else(|| {
                                        "engine.tfidf: missing required field \"model\" \
                                         (path to .json artifact)"
                                            .to_string()
                                    })?;
                                match TfidfEngine::load(model) {
                                    Ok(e) => Some(Box::new(e)),
                                    Err(err) => {
                                        eprintln!(
                                            "warn: tfidf engine failed to load ({err}); \
                                             falling back to logistic for question \"{name}\""
                                        );
                                        None
                                    }
                                }
                            } else {
                                return Err(
                                    "engine must be \"logistic\", {\"onnx\": {...}} or {\"tfidf\": {...}}"
                                        .to_string(),
                                );
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
                    let input = compose_input(&body)?;
                    let registry = registry
                        .lock()
                        .map_err(|e| format!("state lock poisoned: {e}"))?;
                    match body.get("questions") {
                        // Single-question ask: unchanged legacy behavior.
                        None | Some(Value::Null) => {
                            let decision =
                                registry.ask(&question, &input).map_err(|e| e.to_string())?;
                            serde_json::to_value(&decision)
                                .map_err(|e| format!("encode failed: {e}"))
                        }
                        // Batch: independent views of the same engine, in order.
                        Some(items) => answer_batch(&registry, &question, items, &input),
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn trained_registry() -> Registry {
        let mut reg = Registry::new(json!({}), 0.0).unwrap();
        reg.define_question(
            "triage".into(),
            json!({"type": "string"}),
            json!({"type": "string"}),
            Policy::new(0.85, 0.5, 0.85).unwrap(),
        );
        let examples: Vec<(String, String)> = vec![
            ("crash error exception stacktrace".into(), "bug".into()),
            ("null pointer panic segfault".into(), "bug".into()),
            ("add feature request dark mode".into(), "feature".into()),
            ("please add export button".into(), "feature".into()),
        ];
        reg.train("triage", &examples).unwrap();
        reg
    }

    #[test]
    fn compose_input_without_state_is_legacy() {
        let body = json!({"question": "q", "input": "hello"});
        assert_eq!(compose_input(&body).unwrap(), json!("hello"));
        let body = json!({"question": "q"});
        assert_eq!(compose_input(&body).unwrap(), json!({}));
    }

    #[test]
    fn compose_input_renders_state_first() {
        let body = json!({
            "question": "q",
            "state": {"title": "Crash on start", "repo": "k8s"},
            "input": "stacktrace here",
        });
        assert_eq!(
            compose_input(&body).unwrap(),
            json!("repo: k8s\ntitle: Crash on start\nstacktrace here")
        );
        // State alone.
        let body = json!({"question": "q", "state": {"a": "b"}});
        assert_eq!(compose_input(&body).unwrap(), json!("a: b"));
        // Non-object state is a client bug.
        let body = json!({"question": "q", "state": "nope"});
        assert!(compose_input(&body).is_err());
    }

    #[test]
    fn batch_answers_choice_and_noul_in_order() {
        let reg = trained_registry();
        let items = json!([
            {"kind": "choice", "id": "triage"},
            {"kind": "noul", "id": "is_bug", "statement": "This is a bug", "class": "bug"},
            {"kind": "noul", "id": "is_feature", "statement": "This is a feature", "class": "feature"},
        ]);
        let v = answer_batch(&reg, "triage", &items, &json!("crash error panic")).unwrap();
        let answers = v.get("answers").unwrap().as_array().unwrap();
        assert_eq!(answers.len(), 3);
        assert_eq!(answers[0].get("id").unwrap(), "triage");
        assert_eq!(answers[0].get("kind").unwrap(), "choice");
        assert_eq!(answers[1].get("kind").unwrap(), "noul");
        assert_eq!(answers[1].get("prediction").unwrap(), "yes");
        let p_true: f64 = answers[1].get("p_true").unwrap().as_f64().unwrap();
        assert!((0.0..=1.0).contains(&p_true));
        // One-vs-rest: the two noul p_trues come from one distribution.
        let p2: f64 = answers[2].get("p_true").unwrap().as_f64().unwrap();
        assert!((p_true + p2 - 1.0).abs() < 1e-9);
        assert_eq!(answers[2].get("prediction").unwrap(), "no");
    }

    #[test]
    fn batch_rejects_bad_items() {
        let reg = trained_registry();
        // Unknown kind.
        let items = json!([{"kind": "score", "id": "x"}]);
        assert!(answer_batch(&reg, "triage", &items, &json!("hi")).is_err());
        // Unknown class.
        let items = json!([{"kind": "noul", "id": "x", "statement": "s", "class": "nope"}]);
        assert!(answer_batch(&reg, "triage", &items, &json!("hi")).is_err());
        // Missing id.
        let items = json!([{"kind": "choice"}]);
        assert!(answer_batch(&reg, "triage", &items, &json!("hi")).is_err());
        // Empty array and non-array.
        assert!(answer_batch(&reg, "triage", &json!([]), &json!("hi")).is_err());
        assert!(answer_batch(&reg, "triage", &json!({}), &json!("hi")).is_err());
    }
}
