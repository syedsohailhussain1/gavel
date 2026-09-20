//! registry.rs — per-question engine registry.
//!
//! Shared by the Node bindings and the HTTP server: holds registered
//! questions plus one [`LogisticEngine`] per question. A question that has
//! never been trained falls back to the M0 [`MockEngine`] stub so the
//! original define/ask behavior is preserved.

use crate::{DecideError, Decision, Engine, LogisticEngine, MockEngine, Policy, Question};
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

/// Owns questions and their trained engines.
#[derive(Debug)]
pub struct Registry {
    questions: HashMap<String, Question>,
    engines: HashMap<String, LogisticEngine>,
    mock: MockEngine,
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
        self.engines.insert(name.clone(), LogisticEngine::new());
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
        Ok(EngineMetrics {
            trained: engine.is_trained(),
            classes: engine.classes().to_vec(),
            temperature: engine.temperature(),
            ece_before: engine.ece_before(),
            ece_after: engine.ece_after(),
        })
    }

    /// Ask a question. Uses the trained [`LogisticEngine`] when available,
    /// otherwise the M0 mock stub.
    pub fn ask(&self, name: &str, input: &serde_json::Value) -> Result<Decision, DecideError> {
        let question = self.question(name)?;
        match self.engines.get(name) {
            Some(engine) if engine.is_trained() => engine.ask(question, input),
            _ => self.mock.ask(question, input),
        }
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
}
