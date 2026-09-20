//! logistic.rs — multinomial logistic regression on hashed bag-of-words.
//!
//! M1 engine, pure Rust (no torch/ONNX):
//! - features: FNV-1a hashed lowercase word unigrams into 2^14 buckets, TF weights
//! - training: SGD with seeded shuffling (deterministic), L2 regularization
//! - M2: temperature scaling fitted on held-out validation by minimizing NLL,
//!   plus expected calibration error (ECE) computation.
//!
//! The FNV-1a hash and the seeded xorshift PRNG are deliberate: `std`'s
//! `RandomState` is seeded per-process, which would make models
//! non-reproducible across runs.

use crate::{DecideError, Decision, Engine, Question};
use serde_json::json;
use std::collections::HashMap;

/// Number of feature-hash buckets (2^14).
pub const N_BUCKETS: usize = 16384;

/// FNV-1a 64-bit: fast, deterministic across processes and platforms.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Lowercase word tokens: maximal runs of alphanumeric characters.
fn tokenize(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
}

/// Sparse TF vector: (bucket, term count), sorted by bucket.
fn featurize(text: &str) -> Vec<(usize, f64)> {
    let mut counts: HashMap<usize, f64> = HashMap::new();
    for tok in tokenize(&text.to_lowercase()) {
        let b = (fnv1a(tok) % N_BUCKETS as u64) as usize;
        *counts.entry(b).or_insert(0.0) += 1.0;
    }
    let mut feats: Vec<(usize, f64)> = counts.into_iter().collect();
    feats.sort_unstable_by_key(|(b, _)| *b);
    feats
}

/// Dense hashed bag-of-words features: TF counts scattered into `N_BUCKETS`
/// buckets, as `f32`.
///
/// Public so other engine crates (e.g. decide-onnx's `HashedBow` input mode)
/// hash text *identically* to the logistic engine instead of reimplementing
/// the featurizer and risking divergence.
pub fn hashed_bow_dense(text: &str) -> Vec<f32> {
    let mut dense = vec![0.0f32; N_BUCKETS];
    for (b, x) in featurize(text) {
        dense[b] = x as f32;
    }
    dense
}

/// Deterministic xorshift64* PRNG (for reproducible shuffling).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn shuffle(&mut self, xs: &mut [usize]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
}

/// Hyperparameters for [`LogisticEngine::train`].
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub l2: f64,
    /// Fixed seed: training is fully deterministic for a given dataset.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            learning_rate: 0.3,
            epochs: 25,
            l2: 1e-5,
            seed: 0xC0FFEE,
        }
    }
}

/// Softmax with temperature scaling: `softmax_scaled(logits / temperature)`.
///
/// Public so other engine crates can apply the same M2 calibration math to
/// their own logits (see [`fit_temperature`]).
pub fn softmax_scaled(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Fit a temperature scalar for temperature scaling (the M2 recipe).
///
/// Grid-searches `T` in `[0.05, 10]` (80 log-spaced points), minimizing the
/// mean negative log-likelihood of `targets` under `softmax_scaled(logits / T)`.
/// Works over raw logit sets from *any* engine — the logistic engine and the
/// ONNX engine share this exact search.
pub fn fit_temperature(logit_sets: &[Vec<f64>], targets: &[usize]) -> f64 {
    debug_assert_eq!(logit_sets.len(), targets.len());
    let nll = |temperature: f64| {
        let mut total = 0.0;
        for (logits, y) in logit_sets.iter().zip(targets.iter()) {
            let probs = softmax_scaled(logits, temperature);
            total += -probs[*y].clamp(1e-15, 1.0).ln();
        }
        total / logit_sets.len() as f64
    };
    let mut best_t = 1.0;
    let mut best_nll = f64::INFINITY;
    for i in 0..80 {
        let t = 0.05 * 200f64.powf(i as f64 / 79.0);
        let n = nll(t);
        if n < best_nll {
            best_nll = n;
            best_t = t;
        }
    }
    best_t
}

/// Multinomial logistic regression classifier.
///
/// Labels are discovered from training data (sorted for determinism).
/// [`Engine::ask`] returns `{"label": "<predicted>"}` with confidence equal
/// to the max softmax probability (temperature-scaled after [`Self::calibrate`]).
#[derive(Debug, Clone)]
pub struct LogisticEngine {
    config: TrainConfig,
    classes: Vec<String>,
    /// weights[class][bucket]
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    temperature: f64,
    trained: bool,
    /// Uncalibrated ECE: measured on the train set at train time, then
    /// re-measured on the validation set at calibrate time.
    ece_before: Option<f64>,
    /// Calibrated ECE on the validation set (after temperature scaling).
    ece_after: Option<f64>,
}

/// Linear scores for one sparse feature vector: bias + W·x.
fn logits(weights: &[Vec<f64>], bias: &[f64], feats: &[(usize, f64)]) -> Vec<f64> {
    let mut out = bias.to_vec();
    for (c, w) in weights.iter().enumerate() {
        let mut s = 0.0;
        for (b, x) in feats {
            s += w[*b] * x;
        }
        out[c] += s;
    }
    out
}

impl LogisticEngine {
    pub fn new() -> Self {
        Self::with_config(TrainConfig::default())
    }

    pub fn with_config(config: TrainConfig) -> Self {
        LogisticEngine {
            config,
            classes: Vec::new(),
            weights: Vec::new(),
            bias: Vec::new(),
            temperature: 1.0,
            trained: false,
            ece_before: None,
            ece_after: None,
        }
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn ece_before(&self) -> Option<f64> {
        self.ece_before
    }

    pub fn ece_after(&self) -> Option<f64> {
        self.ece_after
    }

    /// Train on (text, label) examples. Deterministic given the dataset.
    pub fn train(&mut self, examples: &[(String, String)]) -> Result<(), DecideError> {
        if examples.is_empty() {
            return Err(DecideError::EngineError(
                "train: no examples provided".into(),
            ));
        }
        let mut classes: Vec<String> = examples.iter().map(|(_, l)| l.clone()).collect();
        classes.sort();
        classes.dedup();
        if classes.len() < 2 {
            return Err(DecideError::EngineError(
                "train: need at least 2 distinct labels".into(),
            ));
        }
        let class_idx: HashMap<&str, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();

        let n_classes = classes.len();
        let mut weights = vec![vec![0.0; N_BUCKETS]; n_classes];
        let mut bias = vec![0.0; n_classes];
        let feats: Vec<Vec<(usize, f64)>> = examples.iter().map(|(t, _)| featurize(t)).collect();
        let targets: Vec<usize> = examples
            .iter()
            .map(|(_, l)| class_idx[l.as_str()])
            .collect();

        let mut order: Vec<usize> = (0..examples.len()).collect();
        for epoch in 0..self.config.epochs {
            Rng(self.config.seed.wrapping_add(epoch as u64)).shuffle(&mut order);
            for &i in &order {
                let probs = softmax_scaled(&logits(&weights, &bias, &feats[i]), 1.0);
                let y = targets[i];
                for c in 0..n_classes {
                    let err = probs[c] - if c == y { 1.0 } else { 0.0 };
                    bias[c] -= self.config.learning_rate * err;
                    for (b, x) in &feats[i] {
                        weights[c][*b] -=
                            self.config.learning_rate * (err * x + self.config.l2 * weights[c][*b]);
                    }
                }
            }
        }

        self.classes = classes;
        self.weights = weights;
        self.bias = bias;
        self.temperature = 1.0;
        self.trained = true;
        self.ece_after = None;

        // Pre-calibration ECE on the training set (T = 1).
        let (confs, correct) = self.predict_batch(
            &examples.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>(),
            &targets,
        );
        self.ece_before = Some(expected_calibration_error(&confs, &correct, 10));
        Ok(())
    }

    /// Predict (class_index, confidence, full probability vector).
    pub fn predict(&self, text: &str) -> Result<(usize, f64, Vec<f64>), DecideError> {
        if !self.trained {
            return Err(DecideError::EngineError(
                "predict: engine has not been trained yet".into(),
            ));
        }
        let probs = softmax_scaled(
            &logits(&self.weights, &self.bias, &featurize(text)),
            self.temperature,
        );
        let (best, conf) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, p)| (i, *p))
            .unwrap();
        Ok((best, conf, probs))
    }

    fn predict_batch(&self, texts: &[String], targets: &[usize]) -> (Vec<f64>, Vec<bool>) {
        let mut confs = Vec::with_capacity(texts.len());
        let mut correct = Vec::with_capacity(texts.len());
        for (t, y) in texts.iter().zip(targets.iter()) {
            // Safe: only called when trained.
            let (pred, conf, _) = self.predict(t).expect("predict_batch on trained engine");
            confs.push(conf);
            correct.push(pred == *y);
        }
        (confs, correct)
    }

    /// Fit temperature scaling on held-out validation examples.
    ///
    /// Grid-searches T in [0.05, 10] (log-spaced) minimizing NLL, then
    /// records ECE before/after on the validation set.
    pub fn calibrate(&mut self, validation: &[(String, String)]) -> Result<(), DecideError> {
        if !self.trained {
            return Err(DecideError::EngineError(
                "calibrate: engine has not been trained yet".into(),
            ));
        }
        if validation.is_empty() {
            return Err(DecideError::EngineError(
                "calibrate: no validation examples provided".into(),
            ));
        }
        let class_idx: HashMap<&str, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        let mut texts = Vec::with_capacity(validation.len());
        let mut targets = Vec::with_capacity(validation.len());
        for (t, l) in validation {
            match class_idx.get(l.as_str()) {
                Some(&i) => {
                    texts.push(t.clone());
                    targets.push(i);
                }
                None => {
                    return Err(DecideError::EngineError(format!(
                        "calibrate: unknown label '{l}' (not seen in training)"
                    )))
                }
            }
        }

        // ECE before calibration (T = 1) on the validation set.
        let (confs, correct) = self.predict_batch(&texts, &targets);
        self.ece_before = Some(expected_calibration_error(&confs, &correct, 10));

        // Grid search for the NLL-minimizing temperature (shared M2 recipe).
        let logit_sets: Vec<Vec<f64>> = texts
            .iter()
            .map(|t| logits(&self.weights, &self.bias, &featurize(t)))
            .collect();
        self.temperature = fit_temperature(&logit_sets, &targets);

        let (confs_c, correct_c) = self.predict_batch(&texts, &targets);
        self.ece_after = Some(expected_calibration_error(&confs_c, &correct_c, 10));
        Ok(())
    }
}

impl Engine for LogisticEngine {
    fn ask(&self, question: &Question, input: &serde_json::Value) -> Result<Decision, DecideError> {
        let (best, conf, _) = self.predict(&extract_text(input))?;
        Decision::new(
            json!({ "label": self.classes[best] }),
            conf,
            &question.policy,
        )
    }

    fn train(&mut self, examples: &[(String, String)]) -> Result<(), DecideError> {
        LogisticEngine::train(self, examples)
    }

    fn calibrate(&mut self, validation: &[(String, String)]) -> Result<(), DecideError> {
        LogisticEngine::calibrate(self, validation)
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn engine_metrics(&self) -> crate::EngineMetrics {
        crate::EngineMetrics {
            trained: self.trained,
            classes: self.classes.clone(),
            temperature: self.temperature,
            ece_before: self.ece_before,
            ece_after: self.ece_after,
        }
    }
}

impl Default for LogisticEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract decision-relevant text from an arbitrary JSON input:
/// - strings are used as-is
/// - objects contribute their string values (keys sorted for determinism)
/// - anything else is stringified
pub fn extract_text(input: &serde_json::Value) -> String {
    match input {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            keys.iter()
                .filter_map(|k| map[*k].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }
        other => other.to_string(),
    }
}

/// Expected calibration error: mean |accuracy − confidence| over `n_bins`
/// equal-width confidence bins, weighted by bin occupancy.
pub fn expected_calibration_error(confidences: &[f64], correct: &[bool], n_bins: usize) -> f64 {
    assert_eq!(confidences.len(), correct.len());
    assert!(n_bins > 0);
    if confidences.is_empty() {
        return 0.0;
    }
    let n = confidences.len() as f64;
    let mut acc_sum = vec![0.0; n_bins];
    let mut conf_sum = vec![0.0; n_bins];
    let mut count = vec![0u64; n_bins];
    for (c, ok) in confidences.iter().zip(correct.iter()) {
        let mut b = (c.clamp(0.0, 1.0) * n_bins as f64).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        acc_sum[b] += if *ok { 1.0 } else { 0.0 };
        conf_sum[b] += c;
        count[b] += 1;
    }
    let mut ece = 0.0;
    for b in 0..n_bins {
        if count[b] > 0 {
            let bin_n = count[b] as f64;
            ece += (bin_n / n) * ((acc_sum[b] - conf_sum[b]).abs() / bin_n);
        }
    }
    ece
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Separable toy set: disjoint vocabularies per class.
    fn separable() -> Vec<(String, String)> {
        let mut ex = Vec::new();
        for i in 0..20 {
            ex.push((
                format!("invoice billing payment refund charge invoice{i}"),
                "billing".to_string(),
            ));
            ex.push((
                format!("password login account reset access login{i}"),
                "account".to_string(),
            ));
            ex.push((
                format!("crash bug error stack trace fails bug{i}"),
                "technical".to_string(),
            ));
        }
        ex
    }

    #[test]
    fn training_converges_on_separable_data() {
        let data = separable();
        let mut eng = LogisticEngine::new();
        eng.train(&data).unwrap();
        assert!(eng.is_trained());
        assert_eq!(eng.classes(), &["account", "billing", "technical"]);
        for (text, label) in &data {
            let (best, conf, probs) = eng.predict(text).unwrap();
            assert_eq!(eng.classes()[best], *label);
            assert!((0.0..=1.0).contains(&conf));
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "softmax must sum to 1, got {sum}");
        }
    }

    #[test]
    fn predictions_are_sane_on_unseen_inputs() {
        let mut eng = LogisticEngine::new();
        eng.train(&separable()).unwrap();
        let (best, conf, _) = eng
            .predict("my invoice was charged twice, refund please")
            .unwrap();
        assert_eq!(eng.classes()[best], "billing");
        assert!(conf > 0.5, "clear input should be confident, got {conf}");
        // Untrained engine refuses.
        let fresh = LogisticEngine::new();
        assert!(fresh.predict("hello").is_err());
    }

    #[test]
    fn train_rejects_bad_input() {
        let mut eng = LogisticEngine::new();
        assert!(eng.train(&[]).is_err());
        assert!(eng
            .train(&[("a b".to_string(), "only".to_string())])
            .is_err());
    }

    #[test]
    fn feature_hashing_is_deterministic() {
        let a = featurize("Hello WORLD hello");
        let b = featurize("Hello WORLD hello");
        assert_eq!(a, b);
        // Case-insensitive: "Hello" and "hello" land in the same bucket.
        let c = featurize("hello world hello");
        assert_eq!(a, c);
    }

    // --- M2: calibration ---

    /// Fixed-seed PRNG for synthetic data (xorshift64*, same construction as Rng).
    struct Gen(u64);
    impl Gen {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn pick<'a>(&mut self, xs: &'a [&str]) -> &'a str {
            xs[(self.next() % xs.len() as u64) as usize]
        }
    }

    /// Overlapping classes + label noise: the model gets overconfident on the
    /// separable-looking parts, so temperature scaling has something to fix.
    fn noisy_data(seed: u64, n: usize) -> Vec<(String, String)> {
        let billing = [
            "invoice", "billing", "payment", "refund", "charge", "receipt",
        ];
        let account = [
            "password", "login", "account", "username", "signin", "profile",
        ];
        let shared = ["please", "help", "issue", "problem", "urgent", "thanks"];
        let mut g = Gen(seed);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let billing_like = i % 2 == 0;
            let words: Vec<&str> = (0..12)
                .map(|_| {
                    let r = g.next() % 100;
                    if r < 45 {
                        g.pick(if billing_like { &billing } else { &account })
                    } else if r < 75 {
                        g.pick(&shared)
                    } else {
                        // 25%: words from the *other* class — overlap + noise.
                        g.pick(if billing_like { &account } else { &billing })
                    }
                })
                .collect();
            // 10% label noise.
            let mut label = if billing_like { "billing" } else { "account" };
            if g.next() % 100 < 10 {
                label = if billing_like { "account" } else { "billing" };
            }
            out.push((words.join(" "), label.to_string()));
        }
        out
    }

    #[test]
    fn calibration_reduces_ece_on_fixed_seed() {
        let train = noisy_data(42, 600);
        let valid = noisy_data(7, 400);
        let mut eng = LogisticEngine::new();
        eng.train(&train).unwrap();
        eng.calibrate(&valid).unwrap();
        let before = eng.ece_before().unwrap();
        let after = eng.ece_after().unwrap();
        assert!(
            after < before,
            "calibration should reduce ECE: before={before:.4}, after={after:.4}, T={:.3}",
            eng.temperature()
        );
        assert!(eng.temperature() > 0.0);
    }

    #[test]
    fn ece_is_zero_for_perfect_predictions() {
        let ece = expected_calibration_error(&[0.9, 0.8, 0.7], &[true, true, true], 10);
        assert!(ece < 0.25); // all correct, confidences vary
        let ece2 = expected_calibration_error(&[], &[], 10);
        assert_eq!(ece2, 0.0);
    }

    #[test]
    fn temperature_scales_confidence_monotonically() {
        let mut eng = LogisticEngine::new();
        eng.train(&separable()).unwrap();
        let (_, conf_t1, _) = eng.predict("invoice billing payment").unwrap();
        eng.temperature = 2.0; // soften
        let (_, conf_t2, _) = eng.predict("invoice billing payment").unwrap();
        assert!(conf_t2 < conf_t1, "higher T should soften confidence");
    }
}
