//! Integration tests for decide-onnx against a tiny synthetic ONNX model.
//!
//! Fixture `fixtures/bow_3label.onnx`: input `bow` [1,16384] float32, one
//! Gemm node with W[i][c] = 3.0 iff i % 3 == c (zero bias), output `logits`
//! [1,3] raw logits (no softmax — the engine applies softmax + temperature).
//! Regenerate with the script documented in crates/decide-onnx/README.md.

use decide_core::{hashed_bow_dense, softmax_scaled, Action, Decision, Engine, Policy, Question};
use decide_onnx::{OnnxConfig, OnnxEngine};
use serde_json::json;
use std::time::Instant;

const LABELS: [&str; 3] = ["billing", "technical", "account"];

fn fixture_path() -> String {
    format!(
        "{}/tests/fixtures/bow_3label.onnx",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn test_config() -> OnnxConfig {
    OnnxConfig::new(
        fixture_path(),
        LABELS.iter().map(|s| s.to_string()).collect(),
    )
    .unwrap()
}

fn test_question() -> Question {
    Question {
        name: "route_ticket".into(),
        input_schema: json!({"type": "object"}),
        output_schema: json!({"type": "object"}),
        policy: Policy::default_policy(),
    }
}

/// Independent expectation, computed from the public featurizer and the
/// fixture's documented weight structure — not by re-running the engine.
fn expected(text: &str, temperature: f64) -> (usize, f64) {
    let feats = hashed_bow_dense(text);
    assert_eq!(feats.len(), 16384);
    let mut logits = vec![0.0f64; 3];
    for (b, x) in feats.iter().enumerate() {
        logits[b % 3] += 3.0 * f64::from(*x);
    }
    let probs = softmax_scaled(&logits, temperature);
    let best = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    (best, probs[best])
}

#[test]
fn ask_selects_argmax_label_with_calibrated_confidence() {
    let engine = OnnxEngine::load(&test_config()).unwrap();
    let q = test_question();
    let text = "please refund my invoice for last month";

    let (exp_idx, exp_conf) = expected(text, 1.0);
    let d: Decision = engine.ask(&q, &json!({ "text": text })).unwrap();

    assert_eq!(d.output, json!({ "label": LABELS[exp_idx] }));
    assert!(
        (d.confidence - exp_conf).abs() < 1e-5,
        "confidence {} != expected {exp_conf}",
        d.confidence
    );
    assert!((0.0..=1.0).contains(&d.confidence));
    // Action always follows the question's policy.
    assert_eq!(d.action, q.policy.decide(d.confidence));
}

#[test]
fn temperature_sharpens_confidence() {
    let warm = OnnxEngine::load(&test_config()).unwrap();
    let mut cold_cfg = test_config();
    cold_cfg.temperature = 0.05;
    let cold = OnnxEngine::load(&cold_cfg).unwrap();
    let q = test_question();
    let input = json!({ "text": "please refund my invoice for last month" });

    let c_warm = warm.ask(&q, &input).unwrap().confidence;
    let c_cold = cold.ask(&q, &input).unwrap().confidence;
    // Same winning label, sharper distribution at low temperature.
    assert_eq!(
        warm.ask(&q, &input).unwrap().output,
        cold.ask(&q, &input).unwrap().output
    );
    assert!(
        c_cold > c_warm,
        "cold confidence {c_cold} should exceed warm {c_warm}"
    );
    let (_, exp_cold) = expected("please refund my invoice for last month", 0.05);
    assert!((c_cold - exp_cold).abs() < 1e-5);
}

#[test]
fn policy_action_follows_thresholds() {
    let engine = OnnxEngine::load(&test_config()).unwrap();
    let input = json!({ "text": "my card was charged twice" });
    let strict = Question {
        policy: Policy::new(1.0, 0.6, 1.0).unwrap(),
        ..test_question()
    };
    let d = engine.ask(&strict, &input).unwrap();
    assert!(d.confidence < 1.0);
    assert!(matches!(d.action, Action::Review | Action::Escalate));

    let lax = Question {
        policy: Policy::new(0.0, 0.0, 0.0).unwrap(),
        ..test_question()
    };
    assert_eq!(engine.ask(&lax, &input).unwrap().action, Action::Act);
}

#[test]
fn missing_model_file_is_a_clear_error() {
    let cfg = OnnxConfig::new("/does/not/exist/model.onnx", vec!["a".into()]).unwrap();
    let err = OnnxEngine::load(&cfg).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not found"), "unexpected message: {msg}");
    assert!(
        msg.contains("/does/not/exist/model.onnx"),
        "unexpected message: {msg}"
    );
}

#[test]
fn empty_labels_rejected_at_config_time() {
    assert!(OnnxConfig::new(fixture_path(), vec![]).is_err());
}

#[test]
fn train_refuses_with_inference_only_error() {
    let mut engine = OnnxEngine::load(&test_config()).unwrap();
    let err = engine
        .train(&[("text".into(), "billing".into())])
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("inference-only"), "unexpected message: {msg}");
}

#[test]
fn logit_count_mismatch_is_a_clear_error() {
    // Fixture emits 3 logits; configure 2 labels.
    let cfg = OnnxConfig::new(fixture_path(), vec!["a".into(), "b".into()]).unwrap();
    let engine = OnnxEngine::load(&cfg).unwrap();
    let err = engine
        .ask(&test_question(), &json!({ "text": "hello" }))
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("3 logits"), "unexpected message: {msg}");
    assert!(msg.contains("2 labels"), "unexpected message: {msg}");
}

#[test]
fn calibrate_fits_temperature_on_validation() {
    let mut engine = OnnxEngine::load(&test_config()).unwrap();
    assert_eq!(engine.temperature(), 1.0);
    // Build validation pairs whose true labels match the fixture's argmax.
    let texts = [
        "refund my invoice please",
        "the server keeps crashing on upload",
        "cannot log into my account",
        "charged twice on my credit card",
        "api returns 500 on every request",
        "password reset email never arrived",
    ];
    let validation: Vec<(String, String)> = texts
        .iter()
        .map(|t| {
            let (idx, _) = expected(t, 1.0);
            (t.to_string(), LABELS[idx].to_string())
        })
        .collect();
    engine.calibrate(&validation).unwrap();
    let t = engine.temperature();
    assert!(
        (0.05..=10.0).contains(&t),
        "temperature {t} outside the M2 grid range"
    );
    // Engine still answers after calibration.
    let d = engine
        .ask(&test_question(), &json!({ "text": texts[0] }))
        .unwrap();
    assert!((0.0..=1.0).contains(&d.confidence));
}

#[test]
fn metrics_report_label_set_and_temperature() {
    let engine = OnnxEngine::load(&test_config()).unwrap();
    let m = engine.engine_metrics();
    assert!(m.trained);
    assert_eq!(m.classes, LABELS);
    assert_eq!(m.temperature, 1.0);
}

#[test]
fn ask_latency_is_well_within_budget() {
    let engine = OnnxEngine::load(&test_config()).unwrap();
    let q = test_question();
    let input = json!({ "text": "please refund my invoice for last month" });
    for _ in 0..10 {
        engine.ask(&q, &input).unwrap(); // warmup
    }
    let n = 200;
    let mut micros = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        engine.ask(&q, &input).unwrap();
        micros.push(t.elapsed().as_secs_f64() * 1e6);
    }
    micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = micros[n / 2];
    let p99 = micros[(n as f64 * 0.99) as usize];
    eprintln!("onnx ask latency (tiny Gemm fixture): p50={p50:.1}µs p99={p99:.1}µs n={n}");
    assert!(p50 < 50_000.0, "p50 {p50:.1}µs exceeds the 50ms budget");
}
