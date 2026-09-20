//! decide-onnx: ONNX Runtime inference engine for gavel (M3).
//!
//! A distilled small-LM (or any classifier exported to ONNX) behind the same
//! [`Engine`](decide_core::Engine) trait as the logistic engine. Swap the
//! brain, keep everything else: typed questions, policy actions,
//! temperature-scaling calibration, the HTTP server, the SDKs.
//!
//! ## Schema-constrained output (the core invariant)
//!
//! For classification questions the label set **is** the output schema.
//! [`OnnxEngine::ask`] runs the session, applies softmax over the model's
//! logits, and takes the argmax **over the configured labels** — the engine
//! is structurally incapable of emitting anything else. There is no
//! free-form generation step, so there is nothing to validate or retry:
//! every output is one of `labels` by construction.
//!
//! ## Input modes
//!
//! - [`InputMode::HashedBow`] (default): text is hashed with decide-core's
//!   FNV-1a bag-of-words featurizer into one `[1, 16384]` float32 tensor.
//!   The model must accept a single float32 input of that shape.
//! - [`InputMode::TokenIds`] (cargo feature `tokenizers`): a HuggingFace
//!   `tokenizer.json` produces `input_ids` + `attention_mask` int64 tensors
//!   of shape `[1, max_length]`.

use decide_core::{
    extract_text, fit_temperature, hashed_bow_dense, softmax_scaled, DecideError, Decision, Engine,
    EngineMetrics, Question, N_BUCKETS,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// How raw input text becomes model input tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// FNV-1a hashed bag-of-words, hashed identically to decide-core:
    /// one input tensor `[1, 16384]` float32 (TF counts).
    HashedBow,
    /// `tokenizer.json` → `input_ids` + `attention_mask`, int64 `[1, max_length]`.
    /// Requires the `tokenizers` cargo feature.
    #[cfg(feature = "tokenizers")]
    TokenIds,
}

/// Configuration for loading an [`OnnxEngine`].
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Path to the `.onnx` model file.
    pub model_path: PathBuf,
    /// Optional path to a HuggingFace `tokenizer.json`. When set, the engine
    /// uses [`InputMode::TokenIds`] (requires the `tokenizers` cargo feature);
    /// when absent it uses [`InputMode::HashedBow`].
    pub tokenizer_path: Option<PathBuf>,
    /// The label set. This is the output schema: `ask` can only ever return
    /// one of these labels. Must be non-empty, and the model must emit
    /// exactly `labels.len()` logits.
    pub labels: Vec<String>,
    /// Sequence length for [`InputMode::TokenIds`] (pad/truncate target).
    pub max_length: usize,
    /// Temperature for softmax calibration. `1.0` = raw model probabilities;
    /// use [`OnnxEngine::calibrate`] to fit it on held-out data (M2 recipe).
    pub temperature: f64,
    /// Model input names, in feed order. `None` (default) uses the session's
    /// own input names in order: one name for [`InputMode::HashedBow`], two
    /// (`input_ids`, `attention_mask`) for [`InputMode::TokenIds`].
    pub input_names: Option<Vec<String>>,
}

impl OnnxConfig {
    /// Minimal config: hashed-BoW model with a label set.
    pub fn new(model_path: impl Into<PathBuf>, labels: Vec<String>) -> Result<Self, DecideError> {
        if labels.is_empty() {
            return Err(DecideError::EngineError(
                "onnx: `labels` must not be empty".into(),
            ));
        }
        Ok(OnnxConfig {
            model_path: model_path.into(),
            tokenizer_path: None,
            labels,
            max_length: 512,
            temperature: 1.0,
            input_names: None,
        })
    }
}

/// ONNX Runtime classification engine: distilled-LM inference for gavel.
///
/// Inference-only: [`Engine::train`] refuses with a clear error. Train the
/// model offline (see `crates/decide-onnx/README.md` for the distillation
/// recipe), export to ONNX, and point a question at the artifact.
pub struct OnnxEngine {
    session: Mutex<ort::session::Session>,
    input_names: Vec<String>,
    output_name: String,
    labels: Vec<String>,
    temperature: Mutex<f64>,
    mode: InputMode,
    #[cfg(feature = "tokenizers")]
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Pad/truncate target for [`InputMode::TokenIds`]. Read by the
    /// feature-gated encoder; kept in all builds so configs stay portable.
    #[cfg_attr(not(feature = "tokenizers"), allow(dead_code))]
    max_length: usize,
}

impl std::fmt::Debug for OnnxEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEngine")
            .field("labels", &self.labels)
            .field("mode", &self.mode)
            .field("input_names", &self.input_names)
            .field("output_name", &self.output_name)
            .finish_non_exhaustive()
    }
}

fn engine_error(context: &str, err: impl std::fmt::Display) -> DecideError {
    DecideError::EngineError(format!("onnx: {context}: {err}"))
}

/// A softmax temperature must be positive and finite (NaN fails both).
fn valid_temperature(t: f64) -> bool {
    t > 0.0 && t.is_finite()
}

impl OnnxEngine {
    /// Load an ONNX model and prepare an inference engine.
    ///
    /// Fails with a descriptive [`DecideError::EngineError`] when the model
    /// file is missing, the session cannot be built, or the configured
    /// input mode is unavailable.
    pub fn load(config: &OnnxConfig) -> Result<Self, DecideError> {
        if !Path::new(&config.model_path).exists() {
            return Err(DecideError::EngineError(format!(
                "onnx: model file not found: {}",
                config.model_path.display()
            )));
        }
        if !valid_temperature(config.temperature) {
            return Err(DecideError::EngineError(format!(
                "onnx: temperature must be positive and finite, got {}",
                config.temperature
            )));
        }

        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&config.model_path))
            .map_err(|e| engine_error("failed to build session", e))?;

        let session_input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| DecideError::EngineError("onnx: model has no outputs".into()))?;

        // Resolve the input mode and validate the plumbing up front.
        #[cfg(feature = "tokenizers")]
        let (mode, tokenizer): (InputMode, Option<tokenizers::Tokenizer>) = match &config
            .tokenizer_path
        {
            Some(tok_path) => {
                if session_input_names.len() < 2 {
                    return Err(DecideError::EngineError(format!(
                            "onnx: TokenIds mode needs a model with >= 2 inputs (input_ids, attention_mask); model has {}",
                            session_input_names.len()
                        )));
                }
                let tokenizer = tokenizers::Tokenizer::from_file(tok_path).map_err(|e| {
                    engine_error(
                        &format!("failed to load tokenizer {}", tok_path.display()),
                        e,
                    )
                })?;
                (InputMode::TokenIds, Some(tokenizer))
            }
            None => (InputMode::HashedBow, None),
        };
        #[cfg(not(feature = "tokenizers"))]
        let mode = match &config.tokenizer_path {
            Some(p) => {
                return Err(DecideError::EngineError(format!(
                "onnx: tokenizer {} requested but the `tokenizers` cargo feature is not enabled",
                p.display()
            )))
            }
            None => InputMode::HashedBow,
        };

        let input_names = match &config.input_names {
            Some(names) => names.clone(),
            None => session_input_names,
        };
        let need = match mode {
            InputMode::HashedBow => 1,
            #[cfg(feature = "tokenizers")]
            InputMode::TokenIds => 2,
        };
        if input_names.len() < need {
            return Err(DecideError::EngineError(format!(
                "onnx: input mode needs {need} input name(s), got {}",
                input_names.len()
            )));
        }

        Ok(OnnxEngine {
            session: Mutex::new(session),
            input_names,
            output_name,
            labels: config.labels.clone(),
            temperature: Mutex::new(config.temperature),
            mode,
            #[cfg(feature = "tokenizers")]
            tokenizer,
            max_length: config.max_length.max(1),
        })
    }

    /// Labels this engine can emit (the output schema).
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Current softmax temperature.
    pub fn temperature(&self) -> f64 {
        *self.temperature.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Override the softmax temperature directly.
    pub fn set_temperature(&self, temperature: f64) -> Result<(), DecideError> {
        if !valid_temperature(temperature) {
            return Err(DecideError::EngineError(format!(
                "onnx: temperature must be positive and finite, got {temperature}"
            )));
        }
        *self
            .temperature
            .lock()
            .map_err(|e| DecideError::EngineError(format!("onnx: lock poisoned: {e}")))? =
            temperature;
        Ok(())
    }

    /// Raw (pre-softmax) logits for one input text.
    fn run_logits(&self, text: &str) -> Result<Vec<f64>, DecideError> {
        let mut session = self
            .session
            .lock()
            .map_err(|e| DecideError::EngineError(format!("onnx: session lock poisoned: {e}")))?;

        let input_name = self.input_names[0].clone();
        // Feature builds only: hoisted so the `inputs!` borrow lives long
        // enough. Defaults to "" in HashedBow mode (unused there).
        #[cfg(feature = "tokenizers")]
        let mask_name = self.input_names.get(1).cloned().unwrap_or_default();
        let inputs = match self.mode {
            InputMode::HashedBow => {
                let feats = hashed_bow_dense(text);
                debug_assert_eq!(feats.len(), N_BUCKETS);
                let tensor = ort::value::Tensor::from_array((vec![1i64, N_BUCKETS as i64], feats))
                    .map_err(|e| engine_error("failed to build input tensor", e))?;
                ort::inputs![input_name.as_str() => tensor]
            }
            #[cfg(feature = "tokenizers")]
            InputMode::TokenIds => {
                let (ids, mask) = self.encode_token_ids(text)?;
                let l = ids.len() as i64;
                let ids_t = ort::value::Tensor::from_array((vec![1i64, l], ids))
                    .map_err(|e| engine_error("failed to build input_ids tensor", e))?;
                let mask_t = ort::value::Tensor::from_array((vec![1i64, l], mask))
                    .map_err(|e| engine_error("failed to build attention_mask tensor", e))?;
                ort::inputs![input_name.as_str() => ids_t, mask_name.as_str() => mask_t]
            }
        };

        let outputs = session
            .run(inputs)
            .map_err(|e| engine_error("session run failed", e))?;
        let (_shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| engine_error("failed to extract output tensor as f32", e))?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    /// Tokenize text into padded/truncated `(input_ids, attention_mask)`.
    #[cfg(feature = "tokenizers")]
    fn encode_token_ids(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), DecideError> {
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            DecideError::EngineError("onnx: TokenIds mode has no tokenizer loaded".into())
        })?;
        let enc = tok
            .encode(text, true)
            .map_err(|e| engine_error("tokenization failed", e))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        ids.truncate(self.max_length);
        mask.truncate(self.max_length);
        // Pad with 0 (documented limitation: assumes 0 is the pad id; pass a
        // tokenizer whose pad token is 0, or extend OnnxConfig).
        while ids.len() < self.max_length {
            ids.push(0);
            mask.push(0);
        }
        Ok((ids, mask))
    }
}

impl Engine for OnnxEngine {
    fn ask(&self, question: &Question, input: &serde_json::Value) -> Result<Decision, DecideError> {
        let logits = self.run_logits(&extract_text(input))?;
        if logits.len() != self.labels.len() {
            return Err(DecideError::EngineError(format!(
                "onnx: model emitted {} logits but the question configures {} labels",
                logits.len(),
                self.labels.len()
            )));
        }
        let temperature = self.temperature();
        let probs = softmax_scaled(&logits, temperature);
        let (best, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("logits non-empty: labels non-empty");
        let confidence = probs[best].clamp(0.0, 1.0);
        // Schema constraint: the output is always one of the configured
        // labels — argmax over the label set, never free-form text.
        Decision::new(
            json!({ "label": self.labels[best].clone() }),
            confidence,
            &question.policy,
        )
    }

    fn train(&mut self, _examples: &[(String, String)]) -> Result<(), DecideError> {
        Err(DecideError::EngineError(
            "train: decide-onnx is inference-only — train the model offline (distillation recipe in crates/decide-onnx/README.md), export to ONNX, and load the artifact".into(),
        ))
    }

    fn calibrate(&mut self, validation: &[(String, String)]) -> Result<(), DecideError> {
        if validation.is_empty() {
            return Err(DecideError::EngineError(
                "calibrate: no validation examples provided".into(),
            ));
        }
        let mut logit_sets = Vec::with_capacity(validation.len());
        let mut targets = Vec::with_capacity(validation.len());
        for (text, label) in validation {
            let target = self.labels.iter().position(|l| l == label).ok_or_else(|| {
                DecideError::EngineError(format!(
                    "calibrate: unknown label '{label}' (not in the configured label set)"
                ))
            })?;
            logit_sets.push(self.run_logits(text)?);
            targets.push(target);
        }
        let t = fit_temperature(&logit_sets, &targets);
        self.set_temperature(t)
    }

    fn is_trained(&self) -> bool {
        // A loaded model is always ready: there is no separate training step.
        true
    }

    fn engine_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            trained: true,
            classes: self.labels.clone(),
            temperature: self.temperature(),
            ece_before: None,
            ece_after: None,
        }
    }
}

#[cfg(all(test, feature = "tokenizers"))]
mod tokenizer_tests {
    use super::*;

    fn token_ids_engine() -> OnnxEngine {
        // The fixture model has a single float32 input, so TokenIds *load*
        // is rejected by the >= 2 input check — but HashedBow load succeeds
        // and lets us exercise encode_token_ids after swapping the mode.
        let mut cfg = OnnxConfig::new(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/bow_3label.onnx"
            ),
            vec!["a".into(), "b".into(), "c".into()],
        )
        .unwrap();
        cfg.max_length = 8;
        let mut engine = OnnxEngine::load(&cfg).unwrap();
        let tok = tokenizers::Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mini_tokenizer.json"
        ))
        .unwrap();
        engine.mode = InputMode::TokenIds;
        engine.tokenizer = Some(tok);
        engine
    }

    #[test]
    fn encode_token_ids_pads_and_masks() {
        let engine = token_ids_engine();
        // "hello world refund" -> ids [1, 2, 3], padded to 8 with 0s.
        let (ids, mask) = engine.encode_token_ids("hello world refund").unwrap();
        assert_eq!(ids, vec![1, 2, 3, 0, 0, 0, 0, 0]);
        assert_eq!(mask, vec![1, 1, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn encode_token_ids_truncates() {
        let engine = token_ids_engine();
        // 10 tokens, truncated to max_length 8.
        let (ids, mask) = engine
            .encode_token_ids("hello world refund hello world refund hello world refund hello")
            .unwrap();
        assert_eq!(ids.len(), 8);
        assert_eq!(mask.iter().sum::<i64>(), 8); // all real tokens
        assert_eq!(ids[..6], [1, 2, 3, 1, 2, 3]);
    }
}
