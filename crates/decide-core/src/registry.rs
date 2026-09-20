//! registry.rs — per-question engine registry.
//!
//! Shared by the Node bindings and the HTTP server: holds registered
//! questions plus one [`Engine`] per question (boxed, so logistic and ONNX
//! engines coexist). A question whose engine is not ready (never trained,
//! model failed to load) falls back to the M0 [`MockEngine`] stub so the
//! original define/ask behavior is preserved.

use crate::{
    DecideError, Decision, Engine, LogisticEngine, MockEngine, NoulDecision, NoulQuestion, Policy,
    Question,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Calibration + training status for one question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub trained: bool,
    pub classes: Vec<String>,
    pub temperature: f64,
    /// Uncalibrated ECE (train set at train time; validation set after calibrate).
    pub ece_before: Option<f64>,
    /// Calibrated ECE on the validation set (None until calibrated).
    pub ece_after: Option<f64>,
}

/// Owns questions and their engines.
///
/// `Box<dyn Engine>` (rather than a concrete engine type) so that M1's
/// logistic engine and M3's ONNX engine — and any future engine — can serve
/// side by side. `Send` is required by the N-API binding.
pub struct Registry {
    questions: HashMap<String, Question>,
    engines: HashMap<String, Box<dyn Engine>>,
    mock: MockEngine,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("questions", &self.question_names())
            .field("mock", &self.mock)
            .finish_non_exhaustive()
    }
}

impl Registry {
    pub fn new(mock_output: serde_json::Value, mock_confidence: f64) -> Result<Self, DecideError> {
        if !(0.0..=1.0).contains(&mock_confidence) {
            return Err(DecideError::InvalidConfidence(mock_confidence));
        }
        Ok(Registry {
            questions: HashMap::new(),
            engines: HashMap::new(),
            mock: MockEngine::new(mock_output, mock_confidence),
        })
    }

    fn question(&self, name: &str) -> Result<&Question, DecideError> {
        self.questions
            .get(name)
            .ok_or_else(|| DecideError::UnknownQuestion(name.to_string()))
    }

    pub fn define_question(
        &mut self,
        name: String,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
        policy: Policy,
    ) {
        self.define_question_with_engine(
            name,
            input_schema,
            output_schema,
            policy,
            Box::new(LogisticEngine::new()),
        );
    }

    /// Register a question served by a caller-supplied engine
    /// (e.g. an [`Engine`] implementation from decide-onnx).
    pub fn define_question_with_engine(
        &mut self,
        name: String,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
        policy: Policy,
        engine: Box<dyn Engine>,
    ) {
        self.engines.insert(name.clone(), engine);
        self.questions.insert(
            name.clone(),
            Question {
                name,
                input_schema,
                output_schema,
                policy,
            },
        );
    }

    pub fn train(&mut self, name: &str, examples: &[(String, String)]) -> Result<(), DecideError> {
        let engine = self
            .engines
            .get_mut(name)
            .ok_or_else(|| DecideError::UnknownQuestion(name.to_string()))?;
        engine.train(examples)
    }

    pub fn calibrate(
        &mut self,
        name: &str,
        validation: &[(String, String)],
    ) -> Result<(), DecideError> {
        let engine = self
            .engines
            .get_mut(name)
            .ok_or_else(|| DecideError::UnknownQuestion(name.to_string()))?;
        engine.calibrate(validation)
    }

    pub fn metrics(&self, name: &str) -> Result<EngineMetrics, DecideError> {
        let engine = self
            .engines
            .get(name)
            .ok_or_else(|| DecideError::UnknownQuestion(name.to_string()))?;
        Ok(engine.engine_metrics())
    }

    /// Ask a question. Uses the question's engine when it is ready
    /// (trained / model loaded), otherwise the M0 mock stub.
    pub fn ask(&self, name: &str, input: &serde_json::Value) -> Result<Decision, DecideError> {
        let question = self.question(name)?;
        match self.engines.get(name) {
            Some(engine) if engine.is_trained() => engine.ask(question, input),
            _ => self.mock.ask(question, input),
        }
    }

    /// Answer a Noul (yes/no) question as a one-vs-rest projection of the
    /// choice distribution: `p_true` is the calibrated probability of
    /// `noul.class` (see [`NoulDecision`] for the honest limits of this).
    ///
    /// Errors when the question is unknown, when no trained engine serves
    /// it (the mock stub has no distribution), when the engine cannot expose
    /// class probabilities, or when `noul.class` is not one of the model's
    /// classes. The question's policy — including the energy abstention
    /// gate — applies exactly as for Choice answers.
    pub fn noul_answer(
        &self,
        name: &str,
        noul: &NoulQuestion,
        input: &serde_json::Value,
    ) -> Result<NoulDecision, DecideError> {
        let question = self.question(name)?;
        let dist = match self.engines.get(name) {
            Some(engine) if engine.is_trained() => engine.scored_distribution(question, input)?,
            _ => {
                return Err(DecideError::EngineError(format!(
                    "noul: question \"{name}\" has no trained engine exposing \
                     class probabilities"
                )))
            }
        };
        let p_true = dist.prob_of(&noul.class).ok_or_else(|| {
            DecideError::EngineError(format!(
                "noul: class {:?} is not one of the model classes {:?}",
                noul.class, dist.classes
            ))
        })?;
        NoulDecision::from_class_probability(
            noul.id.clone(),
            noul.statement.clone(),
            p_true,
            dist.energy,
            &question.policy,
        )
    }

    pub fn question_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.questions.keys().cloned().collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn registry() -> Registry {
        Registry::new(json!({"route": "billing"}), 0.94).unwrap()
    }

    fn define(reg: &mut Registry) {
        reg.define_question(
            "q".into(),
            json!({"type": "object"}),
            json!({"type": "object"}),
            Policy::default_policy(),
        );
    }

    #[test]
    fn untrained_question_falls_back_to_mock() {
        let mut reg = registry();
        define(&mut reg);
        let d = reg.ask("q", &json!({"subject": "x"})).unwrap();
        assert_eq!(d.output, json!({"route": "billing"}));
        assert_eq!(d.confidence, 0.94);
        let m = reg.metrics("q").unwrap();
        assert!(!m.trained);
    }

    #[test]
    fn train_then_ask_uses_logistic_engine() {
        let mut reg = registry();
        define(&mut reg);
        let examples: Vec<(String, String)> = vec![
            ("invoice payment refund".into(), "billing".into()),
            ("invoice charged twice".into(), "billing".into()),
            ("login password reset".into(), "account".into()),
            ("cannot sign in account".into(), "account".into()),
        ];
        reg.train("q", &examples).unwrap();
        let d = reg
            .ask("q", &json!({"text": "refund my invoice payment"}))
            .unwrap();
        assert_eq!(d.output, json!({"label": "billing"}));
        assert!((0.0..=1.0).contains(&d.confidence));
        let m = reg.metrics("q").unwrap();
        assert!(m.trained);
        assert_eq!(m.classes, vec!["account", "billing"]);
        assert!(m.ece_before.is_some());
    }

    #[test]
    fn unknown_question_errors() {
        let mut reg = registry();
        assert!(reg.ask("nope", &json!({})).is_err());
        assert!(reg.train("nope", &[]).is_err());
        assert!(reg.metrics("nope").is_err());
    }

    fn trained_registry() -> Registry {
        let mut reg = registry();
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

    fn noul(id: &str, class: &str) -> NoulQuestion {
        NoulQuestion {
            id: id.into(),
            statement: format!("this is {class}"),
            class: class.into(),
        }
    }

    #[test]
    fn noul_answer_projects_trained_distribution() {
        let reg = trained_registry();
        let d = reg
            .noul_answer(
                "triage",
                &noul("is_bug", "bug"),
                &json!("crash error panic"),
            )
            .unwrap();
        assert_eq!(d.id, "is_bug");
        assert_eq!(d.kind, "noul");
        assert!((0.0..=1.0).contains(&d.p_true));
        assert_eq!(d.prediction, "yes");
        // One-vs-rest honesty: p_true for bug + p_true for feature = 1.
        let d2 = reg
            .noul_answer(
                "triage",
                &noul("is_feature", "feature"),
                &json!("crash error panic"),
            )
            .unwrap();
        assert!((d.p_true + d2.p_true - 1.0).abs() < 1e-9);
        assert_eq!(d2.prediction, "no");
    }

    #[test]
    fn noul_answer_rejects_unknown_class_and_untrained() {
        let reg = trained_registry();
        let err = reg
            .noul_answer("triage", &noul("x", "nope"), &json!("hi"))
            .expect_err("unknown class must fail");
        assert!(err.to_string().contains("not one of the model classes"));
        // Untrained question falls back to the mock, which has no distribution.
        let mut reg2 = registry();
        reg2.define_question(
            "q".into(),
            json!({"type": "object"}),
            json!({"type": "object"}),
            Policy::default_policy(),
        );
        assert!(reg2
            .noul_answer("q", &noul("x", "bug"), &json!("hi"))
            .is_err());
        assert!(reg2
            .noul_answer("missing", &noul("x", "bug"), &json!("hi"))
            .is_err());
    }
}
