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
}

/// Threshold policy mapping a confidence in [0, 1] to an [`Action`].
///
/// - `confidence >= act_above`            -> [`Action::Act`]
/// - `review_range.0 <= confidence < act_above` (when within range) -> [`Action::Review`]
/// - otherwise                            -> [`Action::Escalate`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub act_above: f64,
    pub review_range: (f64, f64),
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
        })
    }

    /// A sensible default: act >= 0.9, review in [0.6, 0.9), escalate below 0.6.
    pub fn default_policy() -> Self {
        Policy {
            act_above: 0.9,
            review_range: (0.6, 0.9),
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
}

impl Decision {
    pub fn new(
        output: serde_json::Value,
        confidence: f64,
        policy: &Policy,
    ) -> Result<Self, DecideError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(DecideError::InvalidConfidence(confidence));
        }
        Ok(Decision {
            output,
            action: policy.decide(confidence),
            confidence,
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
    }
}
