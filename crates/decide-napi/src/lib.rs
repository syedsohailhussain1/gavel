use decide_core::{extract_text, Action, Decision, Policy, Registry};
use napi_derive::napi;

/// Thin Node.js wrapper over decide-core.
///
/// Holds registered questions and one trainable engine per question.
/// `train` / `calibrate` / `metrics` were added in M1/M2; the original
/// `defineQuestion` / `ask` / `questions` API is unchanged.
#[napi]
pub struct DecideEngine {
    registry: Registry,
}

fn parse_policy(policy_json: Option<String>) -> napi::Result<Policy> {
    match policy_json {
        Some(s) => {
            let v: serde_json::Value = serde_json::from_str(&s)
                .map_err(|e| napi::Error::from_reason(format!("invalid policy JSON: {e}")))?;
            let get = |k: &str| {
                v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| {
                    napi::Error::from_reason(format!("policy JSON must contain numeric {k}"))
                })
            };
            Policy::new(get("act_above")?, get("review_low")?, get("review_high")?)
                .map_err(|e| napi::Error::from_reason(e.to_string()))
        }
        None => Ok(Policy::default_policy()),
    }
}

/// Parse `[{"input": <any JSON>, "label": "..."}]` into (text, label) pairs.
fn parse_examples(examples_json: &str, what: &str) -> napi::Result<Vec<(String, String)>> {
    let v: serde_json::Value = serde_json::from_str(examples_json)
        .map_err(|e| napi::Error::from_reason(format!("invalid {what} JSON: {e}")))?;
    let arr = v.as_array().ok_or_else(|| {
        napi::Error::from_reason(format!("{what} must be a JSON array of {{\"input\",\"label\"}}"))
    })?;
    arr.iter()
        .enumerate()
        .map(|(i, item)| {
            let label = item
                .get("label")
                .and_then(|l| l.as_str())
                .ok_or_else(|| {
                    napi::Error::from_reason(format!("{what}[{i}] must contain string \"label\""))
                })?
                .to_string();
            let input = item.get("input").ok_or_else(|| {
                napi::Error::from_reason(format!("{what}[{i}] must contain \"input\""))
            })?;
            Ok((extract_text(input), label))
        })
        .collect()
}

#[napi]
impl DecideEngine {
    /// Create an engine. Before M1 training lands on a question, `ask`
    /// returns the M0 stub decision: fixed `mock_output` at `mock_confidence`.
    /// Pass e.g. `new DecideEngine('{"route":"billing"}', 0.94)`.
    #[napi(constructor)]
    pub fn new(mock_output_json: Option<String>, mock_confidence: Option<f64>) -> napi::Result<Self> {
        let output: serde_json::Value = match mock_output_json {
            Some(s) => serde_json::from_str(&s)
                .map_err(|e| napi::Error::from_reason(format!("invalid mock_output JSON: {e}")))?,
            None => serde_json::json!({}),
        };
        let confidence = mock_confidence.unwrap_or(0.9);
        let registry = Registry::new(output, confidence)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(DecideEngine { registry })
    }

    /// Register a question. `policy_json` is optional; when omitted the
    /// default policy (act >= 0.9, review in [0.6, 0.9), escalate below 0.6) is used.
    /// Expected shape: `{"act_above": 0.9, "review_low": 0.6, "review_high": 0.9}`.
    #[napi]
    pub fn define_question(
        &mut self,
        name: String,
        input_schema_json: String,
        output_schema_json: String,
        policy_json: Option<String>,
    ) -> napi::Result<()> {
        let input_schema: serde_json::Value = serde_json::from_str(&input_schema_json)
            .map_err(|e| napi::Error::from_reason(format!("invalid input_schema JSON: {e}")))?;
        let output_schema: serde_json::Value = serde_json::from_str(&output_schema_json)
            .map_err(|e| napi::Error::from_reason(format!("invalid output_schema JSON: {e}")))?;
        let policy = parse_policy(policy_json)?;
        self.registry
            .define_question(name, input_schema, output_schema, policy);
        Ok(())
    }

    /// Train a question's engine. `examples_json` is
    /// `[{"input": <string|object>, "label": "..."}]`; object inputs have
    /// their string values concatenated, same as `ask`.
    #[napi]
    pub fn train(&mut self, question_name: String, examples_json: String) -> napi::Result<()> {
        let examples = parse_examples(&examples_json, "examples")?;
        self.registry
            .train(&question_name, &examples)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Fit temperature scaling on held-out validation examples (same shape
    /// as `train`). Reduces overconfidence; see `metrics`.
    #[napi]
    pub fn calibrate(
        &mut self,
        question_name: String,
        validation_json: String,
    ) -> napi::Result<()> {
        let validation = parse_examples(&validation_json, "validation")?;
        self.registry
            .calibrate(&question_name, &validation)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Calibration/training status as JSON:
    /// `{"trained":true,"classes":[...],"temperature":1.0,
    ///   "ece_before":0.12,"ece_after":0.04}` (`ece_after` is null until calibrated).
    #[napi]
    pub fn metrics(&self, question_name: String) -> napi::Result<String> {
        let m = self
            .registry
            .metrics(&question_name)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        serde_json::to_string(&m).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Ask a registered question. Returns the decision as a JSON string:
    /// `{"output": {"label": "..."}, "confidence": 0.94, "action": "act"}`.
    /// Uses the trained engine when the question has been trained, otherwise
    /// the M0 mock stub from the constructor.
    #[napi]
    pub fn ask(&self, question_name: String, input_json: String) -> napi::Result<String> {
        let input: serde_json::Value = serde_json::from_str(&input_json)
            .map_err(|e| napi::Error::from_reason(format!("invalid input JSON: {e}")))?;
        let decision: Decision = self
            .registry
            .ask(&question_name, &input)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let action_str = match decision.action {
            Action::Act => "act",
            Action::Review => "review",
            Action::Escalate => "escalate",
        };
        let out = serde_json::json!({
            "output": decision.output,
            "confidence": decision.confidence,
            "action": action_str,
        });
        serde_json::to_string(&out).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// List registered question names.
    #[napi]
    pub fn questions(&self) -> Vec<String> {
        self.registry.question_names()
    }
}
