//! decide-core: types and policy engine for the open typed-decision layer.
//!
//! M0 scope: questions, policies, decisions, and a stub `MockEngine`.
//! M1 adds a real inference engine: [`LogisticEngine`], multinomial logistic
//! regression on hashed bag-of-words features (pure Rust).
//! M2 adds temperature scaling ([`LogisticEngine::calibrate`]) and expected
//! calibration error ([`expected_calibration_error`]).
//! [`Registry`] owns per-question engines for the Node bindings and HTTP server.

use serde::{Deserialize, Serialize};

pub mod logistic;
pub mod registry;

pub use logistic::{
    expected_calibration_error, extract_text, fit_temperature, hashed_bow_dense, softmax_scaled,
    LogisticEngine, TrainConfig, N_BUCKETS,
};
pub use registry::{EngineMetrics, Registry};

/// What the engine recommends the caller do with a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    /// Confidence is high: act on the decision autonomously.
    Act,
    /// Confidence is middling: flag for human review.
    Review,
    /// Confidence is low: escalate / do not act.
    Escalate,
    /// The input does not belong to the question's task (out-of-distribution):
    /// the engine declined to answer. Checked before the confidence policy —
    /// abstention is about *task fit*, not confidence.
    Abstain,
}

/// Threshold policy mapping a confidence in [0, 1] to an [`Action`].
///
/// - `confidence >= act_above`            -> [`Action::Act`]
/// - `review_range.0 <= confidence < act_above` (when within range) -> [`Action::Review`]
/// - otherwise                            -> [`Action::Escalate`]
///
/// When `abstain_energy_above` is set and the engine reports an energy score,
/// [`Policy::decide_with_energy`] checks abstention *first*: an input whose
/// energy exceeds the threshold yields [`Action::Abstain`] regardless of
/// confidence. Energy is task- and model-specific (fit it as a percentile of
/// calibration-set energies per question); `None` disables the gate and the
/// policy behaves exactly as before.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub act_above: f64,
    pub review_range: (f64, f64),
    #[serde(default)]
    pub abstain_energy_above: Option<f64>,
}

impl Policy {
    /// Build a policy, validating the thresholds.
    pub fn new(act_above: f64, review_low: f64, review_high: f64) -> Result<Self, DecideError> {
        for (name, v) in [
            ("act_above", act_above),
            ("review_low", review_low),
            ("review_high", review_high),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(DecideError::InvalidPolicy(format!(
                    "{name} must be in [0,1], got {v}"
                )));
            }
        }
        if review_low > review_high {
            return Err(DecideError::InvalidPolicy(format!(
                "review_low ({review_low}) must be <= review_high ({review_high})"
            )));
        }
        if review_high > act_above {
            return Err(DecideError::InvalidPolicy(format!(
                "review_high ({review_high}) must be <= act_above ({act_above})"
            )));
        }
        Ok(Policy {
            act_above,
            review_range: (review_low, review_high),
            abstain_energy_above: None,
        })
    }

    /// Set an energy-based abstention threshold: inputs whose energy score
    /// exceeds `threshold` map to [`Action::Abstain`] before the confidence
    /// policy is consulted. `None` disables the gate (the default).
    pub fn with_abstain_energy(mut self, threshold: f64) -> Self {
        self.abstain_energy_above = Some(threshold);
        self
    }

    /// A sensible default: act >= 0.9, review in [0.6, 0.9), escalate below 0.6.
    pub fn default_policy() -> Self {
        Policy {
            act_above: 0.9,
            review_range: (0.6, 0.9),
            abstain_energy_above: None,
        }
    }

    /// Map a confidence score to an action.
    pub fn decide(&self, confidence: f64) -> Action {
        if confidence >= self.act_above {
            Action::Act
        } else if confidence >= self.review_range.0 && confidence < self.act_above {
            Action::Review
        } else {
            Action::Escalate
        }
    }

    /// Map a (confidence, energy) pair to an action. Abstention is evaluated
    /// FIRST: when the policy carries an energy threshold and the engine
    /// supplies an energy score above it, the answer is [`Action::Abstain`]
    /// no matter how confident the classifier is — a confident answer to the
    /// wrong task is the failure mode this gate exists to prevent. With no
    /// threshold or no energy score, this is exactly [`Policy::decide`].
    pub fn decide_with_energy(&self, confidence: f64, energy: Option<f64>) -> Action {
        if let (Some(threshold), Some(e)) = (self.abstain_energy_above, energy) {
            if e > threshold {
                return Action::Abstain;
            }
        }
        self.decide(confidence)
    }
}

/// A typed question the engine can answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub name: String,
    /// JSON Schema for the input.
    pub input_schema: serde_json::Value,
    /// JSON Schema for the output (the decision shape).
    pub output_schema: serde_json::Value,
    pub policy: Policy,
}

/// A typed decision returned by an engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// The decision payload; must conform to the question's output schema.
    pub output: serde_json::Value,
    /// Calibrated confidence in [0, 1].
    pub confidence: f64,
    /// Recommended handling, derived from the policy.
    pub action: Action,
    /// Energy score of the input when the engine computes one (higher means
    /// less like the training task). `None` for engines without OOD scoring.
    /// Omitted from JSON when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
}

impl Decision {
    pub fn new(
        output: serde_json::Value,
        confidence: f64,
        policy: &Policy,
    ) -> Result<Self, DecideError> {
        Self::with_energy(output, confidence, None, policy)
    }

    /// Build a decision carrying an energy score. The policy's abstention
    /// gate (if configured) is evaluated before the confidence thresholds.
    pub fn with_energy(
        output: serde_json::Value,
        confidence: f64,
        energy: Option<f64>,
        policy: &Policy,
    ) -> Result<Self, DecideError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(DecideError::InvalidConfidence(confidence));
        }
        Ok(Decision {
            output,
            action: policy.decide_with_energy(confidence, energy),
            confidence,
            energy,
        })
    }
}

/// Render a structured state object as `"key: value"` lines, one per line,
/// for engines that consume plain text.
///
/// Keys are sorted alphabetically so the rendering is deterministic.
/// String values render as-is; numbers and booleans render as JSON;
/// nulls are skipped; nested objects/arrays render as compact JSON.
/// A non-object input renders as an empty string.
///
/// Rationale: field names become ordinary tokens in the text the model was
/// trained on. This is a cheap, transparent encoding — not a schema-aware
/// featurization. If field names dominate the vocabulary (e.g. very short
/// values), they can sway the decision; prefer putting the substantive
/// content in the values.
pub fn render_state(state: &serde_json::Value) -> String {
    let map = match state.as_object() {
        Some(m) => m,
        None => return String::new(),
    };
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    keys.iter()
        .filter_map(|k| {
            let rendered = match &map[*k] {
                serde_json::Value::Null => return None,
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Some(format!("{k}: {rendered}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Calibrated per-class probabilities for one input, plus an optional
/// energy score. Returned by [`Engine::scored_distribution`]; the server
/// builds Noul (yes/no) answers from this.
#[derive(Debug, Clone)]
pub struct ScoredDistribution {
    /// Class names, aligned with `probabilities`.
    pub classes: Vec<String>,
    /// Calibrated probabilities; sum to 1.
    pub probabilities: Vec<f64>,
    /// Energy score when the engine computes one (higher = less like the
    /// training task). Drives the policy's abstention gate for Noul answers
    /// exactly as it does for Choice answers.
    pub energy: Option<f64>,
}

impl ScoredDistribution {
    /// Calibrated probability of `class`, or `None` when the model has no
    /// such class.
    pub fn prob_of(&self, class: &str) -> Option<f64> {
        self.classes
            .iter()
            .position(|c| c == class)
            .map(|i| self.probabilities[i])
    }
}

/// A yes/no question: "is this statement true?" answered as a probability.
///
/// `statement` is carried through for the caller — the engine does **not**
/// read it. What the engine scores is `class`: `p_true` is the calibrated
/// multiclass probability of that class (see [`NoulDecision`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulQuestion {
    /// Caller-supplied id, echoed back in the answer.
    pub id: String,
    /// The statement being judged, e.g. "This issue reports a software defect".
    pub statement: String,
    /// Which model class counts as "yes", e.g. "bug".
    pub class: String,
}

/// A yes/no answer: the probability that the statement is true.
///
/// Built as a **one-vs-rest projection** of the choice distribution:
/// `p_true` is the calibrated `P(class)`, `prediction` is `"yes"` when
/// `p_true >= 0.5`, and `confidence = max(p_true, 1 - p_true)` — the
/// confidence *in the stated prediction*, so a confident "no" can still
/// `act`. The policy's energy abstention gate (when configured) is evaluated
/// first, exactly as for Choice decisions.
///
/// Honest limits, stated plainly:
/// - This is **not** a dedicated binary classifier for the statement. A
///   model trained to distinguish bug/feature/docs/support answers a
///   different question than one trained on "is this a bug, yes or no".
/// - The engine never sees `statement`; a mismatch between the statement
///   and `class` (e.g. statement "is this about docs" with class "bug")
///   produces a confidently wrong-looking answer. The caller owns that mapping.
/// - `p_true` values across classes sum to 1: asking one Noul per class and
///   treating them as independent probabilities double-counts the same
///   distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulDecision {
    /// Caller-supplied question id, echoed back.
    pub id: String,
    /// Always `"noul"`.
    #[serde(default = "noul_kind")]
    pub kind: String,
    /// The judged statement, echoed back.
    pub statement: String,
    /// Calibrated P(class): probability the statement is true.
    pub p_true: f64,
    /// `"yes"` when `p_true >= 0.5`, else `"no"`.
    pub prediction: String,
    /// `max(p_true, 1 - p_true)`: confidence in `prediction`, in [0.5, 1].
    pub confidence: f64,
    /// Recommended handling, derived from the policy (abstention gate first).
    pub action: Action,
    /// Energy score of the input when the engine computes one.
    /// Omitted from JSON when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
}

fn noul_kind() -> String {
    "noul".to_string()
}

impl NoulDecision {
    /// Build from a one-vs-rest projection: `p_true` is the calibrated
    /// probability of the "yes" class. See the struct docs for the limits
    /// of this projection.
    pub fn from_class_probability(
        id: String,
        statement: String,
        p_true: f64,
        energy: Option<f64>,
        policy: &Policy,
    ) -> Result<Self, DecideError> {
        if !(0.0..=1.0).contains(&p_true) {
            return Err(DecideError::InvalidConfidence(p_true));
        }
        let prediction = if p_true >= 0.5 { "yes" } else { "no" };
        let confidence = p_true.max(1.0 - p_true);
        Ok(NoulDecision {
            id,
            kind: noul_kind(),
            statement,
            p_true,
            prediction: prediction.to_string(),
            confidence,
            action: policy.decide_with_energy(confidence, energy),
            energy,
        })
    }
}

/// Errors returned by decide-core.
#[derive(Debug, Clone, PartialEq)]
pub enum DecideError {
    InvalidPolicy(String),
    InvalidConfidence(f64),
    UnknownQuestion(String),
    EngineError(String),
}

impl std::fmt::Display for DecideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecideError::InvalidPolicy(m) => write!(f, "invalid policy: {m}"),
            DecideError::InvalidConfidence(c) => write!(f, "confidence must be in [0,1], got {c}"),
            DecideError::UnknownQuestion(n) => write!(f, "unknown question: {n}"),
            DecideError::EngineError(m) => write!(f, "engine error: {m}"),
        }
    }
}

impl std::error::Error for DecideError {}

/// Inference engine interface. Real engines (embed+classify, distilled LM)
/// implement this trait; M0 ships a [`MockEngine`] for wiring and tests.
///
/// `Send` is required so engines can live in a shared [`Registry`] behind a
/// mutex (the HTTP server) or cross into Node.js threads (the N-API binding).
///
/// Beyond [`Engine::ask`], the trait carries defaulted lifecycle methods so
/// a registry can hold `Box<dyn Engine>` without knowing the engine kind:
/// - [`Engine::train`] / [`Engine::calibrate`] — overridden by trainable
///   engines; the default refuses with a clear error, which is the correct
///   behavior for inference-only engines (e.g. a loaded ONNX model — train
///   it offline, then load the artifact).
/// - [`Engine::is_trained`] — whether `ask` will produce a real decision.
/// - [`Engine::engine_metrics`] — the `/metrics` snapshot.
pub trait Engine: Send {
    fn ask(&self, question: &Question, input: &serde_json::Value) -> Result<Decision, DecideError>;

    /// Calibrated per-class probabilities plus an optional energy score for
    /// `input`. The server builds Noul (yes/no) answers from this:
    /// `p_true` is the calibrated probability of the requested class — a
    /// one-vs-rest projection of the choice distribution, *not* a dedicated
    /// binary judgment of the statement (see [`NoulDecision`]).
    ///
    /// Default: refuse. Engines that only return a single decision keep this
    /// default; the server then rejects Noul questions for them with a clear
    /// error instead of inventing probabilities.
    fn scored_distribution(
        &self,
        question: &Question,
        input: &serde_json::Value,
    ) -> Result<ScoredDistribution, DecideError> {
        let _ = (question, input);
        Err(DecideError::EngineError(
            "scored_distribution: this engine does not expose class probabilities".into(),
        ))
    }

    /// Train on labeled (text, label) examples.
    ///
    /// Default: refuse. Inference-only engines keep this default; the error
    /// message tells the caller where training actually happens.
    fn train(&mut self, _examples: &[(String, String)]) -> Result<(), DecideError> {
        Err(DecideError::EngineError(
            "train: this engine is inference-only; train the model offline and load the resulting artifact".into(),
        ))
    }

    /// Fit calibration (temperature scaling) on held-out validation examples.
    ///
    /// Default: refuse. Engines that support post-hoc calibration override this.
    fn calibrate(&mut self, _validation: &[(String, String)]) -> Result<(), DecideError> {
        Err(DecideError::EngineError(
            "calibrate: this engine does not support calibration".into(),
        ))
    }

    /// Whether the engine is ready to answer (trained, or model loaded).
    /// Unready engines make the registry fall back to the mock stub.
    fn is_trained(&self) -> bool {
        false
    }

    /// Status snapshot for `/metrics`.
    fn engine_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            trained: self.is_trained(),
            classes: Vec::new(),
            temperature: 1.0,
            ece_before: None,
            ece_after: None,
        }
    }
}

/// Stub engine returning a fixed decision + confidence. Useful for
/// integration tests and for developing the SDK before real engines land.
#[derive(Debug, Clone)]
pub struct MockEngine {
    pub output: serde_json::Value,
    pub confidence: f64,
}

impl MockEngine {
    pub fn new(output: serde_json::Value, confidence: f64) -> Self {
        MockEngine { output, confidence }
    }
}

impl Engine for MockEngine {
    fn ask(
        &self,
        question: &Question,
        _input: &serde_json::Value,
    ) -> Result<Decision, DecideError> {
        Decision::new(self.output.clone(), self.confidence, &question.policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy() -> Policy {
        Policy::new(0.9, 0.6, 0.9).unwrap()
    }

    #[test]
    fn policy_maps_confidence_to_action() {
        let p = policy();
        assert_eq!(p.decide(0.95), Action::Act);
        assert_eq!(p.decide(0.90), Action::Act); // boundary: act_above is inclusive
        assert_eq!(p.decide(0.89), Action::Review);
        assert_eq!(p.decide(0.60), Action::Review); // boundary: review_low is inclusive
        assert_eq!(p.decide(0.59), Action::Escalate);
        assert_eq!(p.decide(0.0), Action::Escalate);
        assert_eq!(p.decide(1.0), Action::Act);
    }

    #[test]
    fn policy_rejects_invalid_thresholds() {
        assert!(Policy::new(1.5, 0.6, 0.9).is_err());
        assert!(Policy::new(0.9, 0.9, 0.6).is_err()); // review_low > review_high
        assert!(Policy::new(0.5, 0.6, 0.9).is_err()); // review_high > act_above
        assert!(Policy::new(0.9, 0.6, 0.9).is_ok());
    }

    #[test]
    fn decision_rejects_out_of_range_confidence() {
        let p = policy();
        assert!(Decision::new(json!({"a": 1}), 1.2, &p).is_err());
        assert!(Decision::new(json!({"a": 1}), -0.1, &p).is_err());
        let d = Decision::new(json!({"a": 1}), 0.95, &p).unwrap();
        assert_eq!(d.action, Action::Act);
        assert_eq!(d.confidence, 0.95);
    }

    fn question() -> Question {
        Question {
            name: "route_ticket".to_string(),
            input_schema: json!({"type": "object"}),
            output_schema: json!({"type": "object"}),
            policy: policy(),
        }
    }

    #[test]
    fn mock_engine_returns_fixed_decision_and_policy_action() {
        let engine = MockEngine::new(json!({"route": "billing"}), 0.94);
        let d = engine
            .ask(&question(), &json!({"subject": "invoice"}))
            .unwrap();
        assert_eq!(d.output, json!({"route": "billing"}));
        assert_eq!(d.confidence, 0.94);
        assert_eq!(d.action, Action::Act);

        let low = MockEngine::new(json!({"route": "escalate"}), 0.3);
        let d2 = low.ask(&question(), &json!({})).unwrap();
        assert_eq!(d2.action, Action::Escalate);
    }

    #[test]
    fn action_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Action::Act).unwrap(), "\"act\"");
        assert_eq!(
            serde_json::to_string(&Action::Review).unwrap(),
            "\"review\""
        );
        assert_eq!(
            serde_json::to_string(&Action::Escalate).unwrap(),
            "\"escalate\""
        );
        assert_eq!(
            serde_json::to_string(&Action::Abstain).unwrap(),
            "\"abstain\""
        );
    }

    #[test]
    fn abstain_gate_evaluated_before_confidence_policy() {
        let p = policy().with_abstain_energy(-1.5);
        // High energy abstains even at max confidence.
        assert_eq!(p.decide_with_energy(0.99, Some(-1.2)), Action::Abstain);
        // Below-threshold energy falls through to the confidence policy.
        assert_eq!(p.decide_with_energy(0.99, Some(-1.8)), Action::Act);
        assert_eq!(p.decide_with_energy(0.7, Some(-1.8)), Action::Review);
        assert_eq!(p.decide_with_energy(0.3, Some(-1.8)), Action::Escalate);
        // Boundary: energy exactly at the threshold does not abstain.
        assert_eq!(p.decide_with_energy(0.99, Some(-1.5)), Action::Act);
        // No threshold or no energy => plain confidence policy.
        let plain = policy();
        assert_eq!(plain.decide_with_energy(0.99, Some(-1.0)), Action::Act);
        assert_eq!(p.decide_with_energy(0.99, None), Action::Act);
    }

    #[test]
    fn decision_with_energy_carries_it_and_applies_gate() {
        let p = policy().with_abstain_energy(-1.5);
        let d = Decision::with_energy(json!({"label": "bug"}), 0.95, Some(-1.2), &p).unwrap();
        assert_eq!(d.action, Action::Abstain);
        assert_eq!(d.energy, Some(-1.2));
        let d2 = Decision::with_energy(json!({"label": "bug"}), 0.95, Some(-1.9), &p).unwrap();
        assert_eq!(d2.action, Action::Act);
        // Decision::new leaves energy absent and the JSON omits the field.
        let d3 = Decision::new(json!({"label": "bug"}), 0.95, &policy()).unwrap();
        assert_eq!(d3.energy, None);
        let v = serde_json::to_value(&d3).unwrap();
        assert!(v.get("energy").is_none());
        assert_eq!(
            serde_json::to_value(&d).unwrap().get("energy").unwrap(),
            &json!(-1.2)
        );
    }

    #[test]
    fn render_state_produces_sorted_key_value_lines() {
        let v = json!({"title": "Crash on startup", "priority": 1, "open": true});
        assert_eq!(
            render_state(&v),
            "open: true\npriority: 1\ntitle: Crash on startup"
        );
        // Nulls are skipped; nested values render as compact JSON.
        let v2 = json!({"a": null, "b": {"x": 1}, "c": [1, 2]});
        assert_eq!(render_state(&v2), "b: {\"x\":1}\nc: [1,2]");
        assert_eq!(render_state(&json!({})), "");
        assert_eq!(render_state(&json!("nope")), "");
        assert_eq!(render_state(&json!(null)), "");
    }

    #[test]
    fn scored_distribution_prob_of() {
        let d = ScoredDistribution {
            classes: vec!["bug".into(), "docs".into()],
            probabilities: vec![0.7, 0.3],
            energy: None,
        };
        assert_eq!(d.prob_of("bug"), Some(0.7));
        assert_eq!(d.prob_of("docs"), Some(0.3));
        assert_eq!(d.prob_of("nope"), None);
    }

    #[test]
    fn noul_decision_projects_class_probability() {
        let p = Policy::new(0.85, 0.5, 0.85).unwrap();
        // Confident yes.
        let d = NoulDecision::from_class_probability(
            "is_bug".into(),
            "This is a bug".into(),
            0.92,
            None,
            &p,
        )
        .unwrap();
        assert_eq!(d.prediction, "yes");
        assert!((d.confidence - 0.92).abs() < 1e-12);
        assert_eq!(d.action, Action::Act);
        assert_eq!(d.kind, "noul");
        // Confident no: confidence is in the *prediction*, so it can still act.
        let d2 = NoulDecision::from_class_probability(
            "is_bug".into(),
            "This is a bug".into(),
            0.08,
            None,
            &p,
        )
        .unwrap();
        assert_eq!(d2.prediction, "no");
        assert!((d2.confidence - 0.92).abs() < 1e-12);
        assert_eq!(d2.action, Action::Act);
        // Fence-sitter: confidence 0.5 -> review (review_low is inclusive).
        // Note: noul confidence is max(p,1-p) >= 0.5, so with review_low=0.5
        // a noul answer can never reach escalate — the floor is review.
        let d3 =
            NoulDecision::from_class_probability("q".into(), "s".into(), 0.5, None, &p).unwrap();
        assert_eq!(d3.prediction, "yes"); // boundary: >= 0.5 is yes
        assert_eq!(d3.confidence, 0.5);
        assert_eq!(d3.action, Action::Review);
        // p_true outside [0,1] is rejected.
        assert!(
            NoulDecision::from_class_probability("q".into(), "s".into(), 1.5, None, &p).is_err()
        );
    }

    #[test]
    fn noul_energy_gate_overrides_confidence() {
        let p = Policy::new(0.85, 0.5, 0.85)
            .unwrap()
            .with_abstain_energy(-1.5);
        let d = NoulDecision::from_class_probability("q".into(), "s".into(), 0.99, Some(-1.2), &p)
            .unwrap();
        assert_eq!(d.action, Action::Abstain);
        assert_eq!(d.energy, Some(-1.2));
        let d2 = NoulDecision::from_class_probability("q".into(), "s".into(), 0.99, Some(-1.9), &p)
            .unwrap();
        assert_eq!(d2.action, Action::Act);
    }

    #[test]
    fn engine_scored_distribution_default_refuses() {
        let engine = MockEngine::new(json!({"label": "x"}), 0.9);
        let q = question();
        let err = engine
            .scored_distribution(&q, &json!("hi"))
            .expect_err("mock has no distribution");
        assert!(err
            .to_string()
            .contains("does not expose class probabilities"));
    }
}
