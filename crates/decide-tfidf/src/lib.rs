//! decide-tfidf: native-Rust TF-IDF + logistic-regression inference engine.
//!
//! Loads a model artifact exported from the sklearn pipeline used by the
//! issue-triage flagship (`TfidfVectorizer` word 1-2grams + `LogisticRegression`)
//! and scores inputs with the identical recipe, so this engine reproduces
//! sklearn's logits to floating-point noise. No ONNX runtime, no Python —
//! the "small, fast, CPU-friendly" path.
//!
//! The engine also computes the energy score `E(x) = -logsumexp(logits)` from
//! the *raw* logits (Liu et al., NeurIPS 2020) and attaches it to every
//! decision. A question whose policy sets `abstain_energy_above` therefore
//! gets out-of-distribution abstention for free: inputs whose energy exceeds
//! the threshold map to [`decide_core::Action::Abstain`] before the
//! confidence policy is consulted.
//!
//! Recipe parity with sklearn (verified to 6.2e-15 max logit diff on 1,250
//! held-out examples):
//! - tokens: `token_pattern=r"(?u)\b\w\w+\b"` on lowercased text (maximal
//!   runs of unicode word chars — alphanumeric or `_` — of length >= 2)
//! - ngrams: word 1-2grams, bigrams joined with a single space
//! - tf: sublinear (`1 + ln(count)`); weight = tf * idf (idf from the artifact)
//! - row L2 normalization; logits = intercept + X·coef

use decide_core::{
    extract_text, softmax_scaled, DecideError, Decision, Engine, EngineMetrics, Question,
    ScoredDistribution,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

/// Model artifact written by `tasks/issue-triage/08_export_tfidf.py`.
#[derive(Debug, Deserialize)]
struct Artifact {
    model_id: String,
    classes: Vec<String>,
    temperature: f64,
    intercept: Vec<f64>,
    /// term -> [idf, coef_class0, coef_class1, ...]
    vocab: HashMap<String, Vec<f64>>,
    #[serde(default)]
    eval: Option<EvalStats>,
}

/// Offline evaluation numbers baked into the artifact (for `/metrics`).
#[derive(Debug, Clone, Deserialize)]
struct EvalStats {
    #[serde(default)]
    ece_before: Option<f64>,
    #[serde(default)]
    ece_after: Option<f64>,
}

/// A loaded TF-IDF + logistic model: vocabulary with idf and per-class
/// coefficients, intercepts, class names, and the calibration temperature.
pub struct TfidfModel {
    model_id: String,
    classes: Vec<String>,
    temperature: f64,
    intercept: Vec<f64>,
    /// term -> (idf, per-class coefficients)
    vocab: HashMap<String, (f64, Vec<f64>)>,
    eval: Option<EvalStats>,
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Tokenize like sklearn's default `token_pattern=r"(?u)\b\w\w+\b"`: maximal
/// runs of word characters (unicode-aware, `_` counts) of length >= 2,
/// on the lowercased text.
fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut toks = Vec::new();
    let mut start: Option<usize> = None;
    let mut len = 0usize;
    for (i, c) in lower.char_indices() {
        if is_word_char(c) {
            if start.is_none() {
                start = Some(i);
            }
            len += 1;
        } else if let Some(s) = start.take() {
            if len >= 2 {
                toks.push(lower[s..i].to_string());
            }
            len = 0;
        }
    }
    if let Some(s) = start {
        if len >= 2 {
            toks.push(lower[s..].to_string());
        }
    }
    toks
}

impl TfidfModel {
    /// Load a model artifact exported by `08_export_tfidf.py`.
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("read model {path}: {e}"))?;
        let art: Artifact =
            serde_json::from_str(&data).map_err(|e| format!("parse model {path}: {e}"))?;
        let n = art.classes.len();
        if n == 0 {
            return Err("model artifact has no classes".into());
        }
        if art.intercept.len() != n {
            return Err(format!(
                "model artifact: {} intercepts but {} classes",
                art.intercept.len(),
                n
            ));
        }
        if !art.temperature.is_finite() || art.temperature <= 0.0 {
            return Err(format!(
                "model artifact: temperature must be positive, got {}",
                art.temperature
            ));
        }
        let mut vocab = HashMap::with_capacity(art.vocab.len());
        for (term, w) in art.vocab {
            if w.len() != n + 1 {
                return Err(format!(
                    "model artifact: vocab entry {term:?} has {} weights, expected {}",
                    w.len(),
                    n + 1
                ));
            }
            let mut w = w;
            let idf = w.remove(0);
            vocab.insert(term, (idf, w));
        }
        Ok(TfidfModel {
            model_id: art.model_id,
            classes: art.classes,
            temperature: art.temperature,
            intercept: art.intercept,
            vocab,
            eval: art.eval,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Raw logits `z = intercept + X·coef`, where X is the L2-normalized
    /// sublinear-tf * idf vector over word 1-2grams. Out-of-vocabulary terms
    /// contribute nothing; an input with no in-vocabulary terms yields the
    /// intercepts alone.
    pub fn logits(&self, text: &str) -> Vec<f64> {
        let toks = tokenize(text);
        let mut ngrams: Vec<String> = toks.clone();
        for i in 0..toks.len().saturating_sub(1) {
            ngrams.push(format!("{} {}", toks[i], toks[i + 1]));
        }
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for ng in &ngrams {
            if self.vocab.contains_key(ng.as_str()) {
                *counts.entry(ng.as_str()).or_insert(0) += 1;
            }
        }
        let mut weighted: Vec<(f64, &[f64])> = Vec::with_capacity(counts.len());
        let mut norm_sq = 0.0;
        for (ng, c) in &counts {
            let (idf, coefs) = &self.vocab[*ng];
            let x = (1.0 + (*c as f64).ln()) * idf;
            norm_sq += x * x;
            weighted.push((x, coefs));
        }
        let mut out = self.intercept.clone();
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for (x, coefs) in weighted {
                let xn = x / norm;
                for (o, w) in out.iter_mut().zip(coefs.iter()) {
                    *o += xn * w;
                }
            }
        }
        out
    }

    /// Energy score `E(x) = -logsumexp(logits)` on raw logits. Lower means
    /// the input looks more like the training task; higher means more
    /// out-of-distribution-like. Numerically stable via the max trick.
    pub fn energy(logits: &[f64]) -> f64 {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|l| (l - max).exp()).sum();
        -(max + sum.ln())
    }

    /// Temperature-scaled class probabilities.
    pub fn probabilities(&self, logits: &[f64]) -> Vec<f64> {
        softmax_scaled(logits, self.temperature)
    }
}

/// Inference engine serving a loaded [`TfidfModel`].
///
/// [`Engine::ask`] returns `{"label": "<predicted>"}` with confidence from
/// the temperature-scaled softmax, and attaches the raw-logit energy score
/// so the question's policy can abstain on out-of-distribution inputs.
pub struct TfidfEngine {
    model: TfidfModel,
}

impl TfidfEngine {
    pub fn load(path: &str) -> Result<Self, String> {
        Ok(TfidfEngine {
            model: TfidfModel::load(path)?,
        })
    }

    pub fn model(&self) -> &TfidfModel {
        &self.model
    }
}

impl Engine for TfidfEngine {
    fn ask(&self, question: &Question, input: &serde_json::Value) -> Result<Decision, DecideError> {
        let logits = self.model.logits(&extract_text(input));
        let energy = TfidfModel::energy(&logits);
        let probs = self.model.probabilities(&logits);
        let (best, conf) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, p)| (i, *p))
            .expect("model has at least one class");
        Decision::with_energy(
            json!({ "label": self.model.classes[best] }),
            conf,
            Some(energy),
            &question.policy,
        )
    }

    /// Full calibrated class distribution plus the raw-logit energy score.
    /// Powers Noul (yes/no) answers: `p_true` for a class is read straight
    /// out of this distribution (one-vs-rest projection).
    fn scored_distribution(
        &self,
        _question: &Question,
        input: &serde_json::Value,
    ) -> Result<ScoredDistribution, DecideError> {
        let logits = self.model.logits(&extract_text(input));
        let energy = TfidfModel::energy(&logits);
        Ok(ScoredDistribution {
            classes: self.model.classes().to_vec(),
            probabilities: self.model.probabilities(&logits),
            energy: Some(energy),
        })
    }

    fn is_trained(&self) -> bool {
        true
    }

    fn engine_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            trained: true,
            classes: self.model.classes.clone(),
            temperature: self.model.temperature,
            ece_before: self.model.eval.as_ref().and_then(|e| e.ece_before),
            ece_after: self.model.eval.as_ref().and_then(|e| e.ece_after),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_matches_sklearn_pattern() {
        // sklearn: re.compile(r"(?u)\b\w\w+\b").findall(text.lower())
        assert_eq!(tokenize("Hello WORLD"), vec!["hello", "world"]);
        assert_eq!(tokenize("don't stop"), vec!["don", "stop"]); // "t" is 1 char: dropped
        assert_eq!(tokenize("hello_world foo"), vec!["hello_world", "foo"]); // _ is a word char
        assert_eq!(tokenize("k8s 1.14 C++"), vec!["k8s", "14"]); // "c" and digits: "1" dropped
        assert_eq!(tokenize("café naïve"), vec!["café", "naïve"]); // unicode word chars
        assert_eq!(tokenize("a b c"), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
    }

    #[test]
    fn energy_of_flat_logits_is_minus_log_n() {
        let e = TfidfModel::energy(&[0.0, 0.0, 0.0, 0.0]);
        assert!((e - (-(4.0f64).ln())).abs() < 1e-12, "got {e}");
        // A peaked logit vector has much lower energy than a flat one.
        let e2 = TfidfModel::energy(&[10.0, 0.0, 0.0, 0.0]);
        assert!(e2 < e, "peaked {e2} should be < flat {e}");
    }

    #[test]
    fn empty_input_yields_intercepts() {
        let model = TfidfModel {
            model_id: "test".into(),
            classes: vec!["a".into(), "b".into()],
            temperature: 1.0,
            intercept: vec![0.3, -0.7],
            vocab: HashMap::new(),
            eval: None,
        };
        assert_eq!(model.logits(""), vec![0.3, -0.7]);
        assert_eq!(model.logits("zzz qqq"), vec![0.3, -0.7]); // all OOV
    }

    #[test]
    fn logits_follow_tfidf_recipe() {
        // One-term vocab: idf=2.0, coefs [1.0, -1.0]; single occurrence of "ab".
        // tf = 1 + ln(1) = 1; x = 2.0; l2 norm = 2.0 -> xn = 1.0.
        let mut vocab = HashMap::new();
        vocab.insert("ab".to_string(), (2.0, vec![1.0, -1.0]));
        let model = TfidfModel {
            model_id: "test".into(),
            classes: vec!["a".into(), "b".into()],
            temperature: 1.0,
            intercept: vec![0.0, 0.0],
            vocab,
            eval: None,
        };
        let z = model.logits("ab ab ab");
        // tf = 1 + ln(3); x = (1+ln3)*2; normalized by itself -> xn = 1.0
        assert!((z[0] - 1.0).abs() < 1e-12, "got {z:?}");
        assert!((z[1] + 1.0).abs() < 1e-12, "got {z:?}");
    }

    fn tiny_engine() -> TfidfEngine {
        // Two-term vocab; "ab" favors class a, "cd" favors class b.
        let mut vocab = HashMap::new();
        vocab.insert("ab".to_string(), (2.0, vec![2.0, -2.0]));
        vocab.insert("cd".to_string(), (2.0, vec![-2.0, 2.0]));
        TfidfEngine {
            model: TfidfModel {
                model_id: "test".into(),
                classes: vec!["a".into(), "b".into()],
                temperature: 1.0,
                intercept: vec![0.0, 0.0],
                vocab,
                eval: None,
            },
        }
    }

    #[test]
    fn scored_distribution_sums_to_one_and_carries_energy() {
        use decide_core::{Policy, Question};
        let engine = tiny_engine();
        let q = Question {
            name: "q".into(),
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
            policy: Policy::default_policy(),
        };
        let dist = engine
            .scored_distribution(&q, &serde_json::json!("ab ab"))
            .unwrap();
        assert_eq!(dist.classes, vec!["a".to_string(), "b".to_string()]);
        let sum: f64 = dist.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "probs sum to {sum}");
        assert!(dist.prob_of("a").unwrap() > 0.9);
        assert!(dist.energy.is_some());
    }
}
