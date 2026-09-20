//! Reproducible benchmark for the gavel logistic decision engine.
//!
//! Generates a SYNTHETIC labeled dataset with a fixed seed, trains
//! `decide_core::LogisticEngine`, calibrates it, and measures train time,
//! per-ask latency, accuracy, and expected calibration error (ECE) before and
//! after temperature calibration.
//!
//! No external data, no vendor/LLM comparisons — everything reported is
//! measured on this synthetic dataset.

use decide_core::{expected_calibration_error, LogisticEngine};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64*); fixed seed => reproducible dataset.
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift seed must be non-zero");
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform usize in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform f64 in [0, 1).
    fn unit(&mut self) -> f64 {
        // Use the top 53 bits for a well-distributed [0,1) double.
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Synthetic dataset.
// ---------------------------------------------------------------------------

const SEED: u64 = 0x1234_5678_9ABC_DEF1;
const N_TRAIN: usize = 2000;
const N_VAL: usize = 500;
const N_TEST: usize = 1000;
/// Fraction of words drawn from a *different* class's vocabulary (noise).
const CROSS_CLASS_NOISE: f64 = 0.15;
/// Fraction of words drawn from the class's own vocabulary (rest is filler).
const OWN_CLASS_WORD: f64 = 0.60;

const CLASSES: [&str; 6] = [
    "billing",
    "shipping",
    "returns",
    "account",
    "technical",
    "feedback",
];

/// ~25 distinctive words per class.
const VOCAB: [&[&str]; 6] = [
    &[
        "invoice", "receipt", "charged", "payment", "billing", "card", "credit",
        "subscription", "overcharged", "amount", "total", "tax", "fee",
        "statement", "balance", "due", "transaction", "charge", "owed",
        "purchase", "order", "plan", "renew", "invoice_id", "prorated",
    ],
    &[
        "shipment", "delivery", "tracking", "package", "shipped", "carrier",
        "courier", "address", "freight", "parcel", "dispatch", "warehouse",
        "transit", "arrived", "delayed", "express", "priority", "logistics",
        "box", "route", "driver", "doorstep", "mailbox", "overnight",
        "waybill", "manifest",
    ],
    &[
        "return", "exchange", "defective", "broken", "damaged", "warranty",
        "replacement", "rma", "rebate", "restocking", "reimburse", "unopened",
        "faulty", "missing", "cracked", "scratched", "mislabeled", "recall",
        "swap", "voucher", "claim", "dispute", "refundable", "defect", "doa",
    ],
    &[
        "login", "password", "username", "profile", "signup", "signin",
        "reset", "verify", "locked", "suspended", "settings", "security",
        "authentication", "register", "deactivate", "reactivate", "session",
        "token", "oauth", "privacy", "rename", "twofactor", "passkey",
        "recovery", "username_taken", "sso",
    ],
    &[
        "error", "bug", "crash", "exception", "timeout", "lag", "slow",
        "freeze", "glitch", "outage", "downtime", "server", "database", "api",
        "endpoint", "latency", "memory", "cpu", "deploy", "patch", "backup",
        "restore", "failure", "issue", "stacktrace", "regression",
    ],
    &[
        "suggestion", "feedback", "idea", "feature", "request", "improvement",
        "love", "great", "awesome", "terrible", "dislike", "rating", "review",
        "survey", "comment", "opinion", "praise", "complaint", "wishlist",
        "vote", "poll", "recommend", "usability", "darkmode", "roadmap",
        "nps",
    ],
];

const FILLER: &[&str] = &[
    "the", "a", "my", "please", "could", "you", "help", "me", "with", "i",
    "need", "want", "to", "and", "is", "was", "for", "on", "it", "this",
    "that", "of", "in", "hi", "hello", "thanks", "thank", "urgent", "today",
    "now", "very", "really", "just", "so", "not", "no", "yes", "kindly",
    "someone", "having", "issue", "about", "regarding",
];

/// Generate `n` labeled examples. Class assignment cycles 0..6 so splits are
/// balanced, then examples are shuffled with the seeded RNG.
fn generate_split(rng: &mut XorShift64, n: usize) -> Vec<(String, String)> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % CLASSES.len();
        let len = 10 + rng.below(15); // 10..24 words
        let mut words = Vec::with_capacity(len);
        for _ in 0..len {
            let r = rng.unit();
            if r < CROSS_CLASS_NOISE {
                // Cross-class noise: word from a different class's vocab.
                let other = (class + 1 + rng.below(CLASSES.len() - 1)) % CLASSES.len();
                let v = VOCAB[other];
                words.push(v[rng.below(v.len())]);
            } else if r < CROSS_CLASS_NOISE + OWN_CLASS_WORD {
                let v = VOCAB[class];
                words.push(v[rng.below(v.len())]);
            } else {
                words.push(FILLER[rng.below(FILLER.len())]);
            }
        }
        out.push((words.join(" "), CLASSES[class].to_string()));
    }
    // Fisher-Yates shuffle with the same RNG (deterministic).
    for i in (1..out.len()).rev() {
        let j = rng.below(i + 1);
        out.swap(i, j);
    }
    out
}

// ---------------------------------------------------------------------------
// Evaluation.
// ---------------------------------------------------------------------------

/// Returns (predicted-class-index, confidence, correctness) for each input.
fn evaluate(
    engine: &LogisticEngine,
    test: &[(String, String)],
) -> Result<(Vec<f64>, Vec<bool>), decide_core::DecideError> {
    let classes = engine.classes();
    let mut confs = Vec::with_capacity(test.len());
    let mut correct = Vec::with_capacity(test.len());
    for (text, label) in test {
        let (idx, conf, _) = engine.predict(text)?;
        let pred_label = classes.get(idx).map(String::as_str).unwrap_or("");
        confs.push(conf);
        correct.push(pred_label == label.as_str());
    }
    Ok((confs, correct))
}

fn accuracy(correct: &[bool]) -> f64 {
    correct.iter().filter(|&&c| c).count() as f64 / correct.len() as f64
}

fn percentile(sorted_ns: &[u64], p: f64) -> u64 {
    let idx = (((p / 100.0) * sorted_ns.len() as f64).ceil() as usize).saturating_sub(1);
    sorted_ns[idx.min(sorted_ns.len() - 1)]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. Synthetic dataset (fixed seed) ---------------------------------
    let mut rng = XorShift64::new(SEED);
    let train = generate_split(&mut rng, N_TRAIN);
    let val = generate_split(&mut rng, N_VAL);
    let test = generate_split(&mut rng, N_TEST);

    // ---- 2. Train (timed) + calibrate --------------------------------------
    let t0 = Instant::now();
    let mut engine = LogisticEngine::new();
    engine.train(&train)?;
    let train_time = t0.elapsed();

    let t0 = Instant::now();
    engine.calibrate(&val)?;
    let calib_time = t0.elapsed();

    // "Before calibration" numbers come from an identically-trained engine
    // that is never calibrated (temperature stays at T=1). This is the honest
    // reading of ECE-before: same weights, no temperature scaling.
    let mut engine_uncal = LogisticEngine::new();
    engine_uncal.train(&train)?;

    // ---- 3. Evaluate on test set --------------------------------------------
    let (confs_cal, correct_cal) = evaluate(&engine, &test)?;
    let (confs_uncal, correct_uncal) = evaluate(&engine_uncal, &test)?;

    let acc = accuracy(&correct_cal);
    let ece_before = expected_calibration_error(&confs_uncal, &correct_uncal, 10);
    let ece_after = expected_calibration_error(&confs_cal, &correct_cal, 10);

    // ---- 4. Latency: warmup, then single-threaded per-ask timings ------------
    for (text, _) in test.iter().take(100) {
        let _ = engine.predict(text)?;
    }
    let mut lat_ns: Vec<u64> = Vec::with_capacity(test.len());
    let t0 = Instant::now();
    for (text, _) in &test {
        let s = Instant::now();
        let _ = engine.predict(text)?;
        lat_ns.push(s.elapsed().as_nanos() as u64);
    }
    let total = t0.elapsed();
    lat_ns.sort_unstable();

    let p50_ns = percentile(&lat_ns, 50.0);
    let p99_ns = percentile(&lat_ns, 99.0);
    let mean_ns = lat_ns.iter().sum::<u64>() as f64 / lat_ns.len() as f64;
    let qps = test.len() as f64 / total.as_secs_f64();

    // Pick an honest unit: microseconds unless p99 reaches milliseconds.
    let (unit, scale) = if p99_ns >= 1_000_000 {
        ("ms", 1_000_000.0)
    } else {
        ("µs", 1_000.0)
    };
    let p50 = p50_ns as f64 / scale;
    let p99 = p99_ns as f64 / scale;
    let mean = mean_ns / scale;

    let train_ms = train_time.as_secs_f64() * 1000.0;
    let calib_ms = calib_time.as_secs_f64() * 1000.0;

    // ---- 5. Report: Markdown table to stdout ----------------------------------
    println!("# decide-bench results (synthetic dataset)");
    println!();
    println!(
        "Synthetic text-classification benchmark: 6 classes, bag-of-words logistic model, \
         15% cross-class word noise. All numbers measured in this run."
    );
    println!();
    println!("| metric | value |");
    println!("|---|---|");
    println!("| train examples | {} |", N_TRAIN);
    println!("| validation examples | {} |", N_VAL);
    println!("| test examples | {} |", N_TEST);
    println!("| seed | {:#x} |", SEED);
    println!("| train wall-time | {:.1} ms |", train_ms);
    println!("| calibrate wall-time | {:.1} ms |", calib_ms);
    println!("| per-ask latency p50 | {:.1} {} |", p50, unit);
    println!("| per-ask latency p99 | {:.1} {} |", p99, unit);
    println!("| per-ask latency mean | {:.1} {} |", mean, unit);
    println!("| throughput (QPS) | {:.0} |", qps);
    println!("| accuracy (test) | {:.4} |", acc);
    println!("| ECE before calibration (T=1) | {:.4} |", ece_before);
    println!("| ECE after calibration | {:.4} |", ece_after);
    println!("| temperature after calibrate | {:.3} |", engine.temperature());

    // ---- 6. JSON artifact ------------------------------------------------------
    let unix_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs();
    let repo_root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let results_dir = format!("{repo_root}/benches/results");
    std::fs::create_dir_all(&results_dir)?;
    let results_dir = std::fs::canonicalize(&results_dir)?.to_string_lossy().into_owned();
    let json_path = format!("{results_dir}/{unix_ts}.json");

    let payload = serde_json::json!({
        "name": "decide-bench",
        "version": "0.1.0",
        "dataset": "synthetic",
        "timestamp_unix": unix_ts,
        "seed": format!("{:#x}", SEED),
        "classes": CLASSES,
        "sizes": { "train": N_TRAIN, "validation": N_VAL, "test": N_TEST },
        "cross_class_noise": CROSS_CLASS_NOISE,
        "metrics": {
            "train_wall_time_ms": train_ms,
            "calibrate_wall_time_ms": calib_ms,
            "latency_unit": unit,
            "latency_p50": p50,
            "latency_p99": p99,
            "latency_mean": mean,
            "qps": qps,
            "accuracy": acc,
            "ece_before_calibration": ece_before,
            "ece_after_calibration": ece_after,
            "temperature_after_calibrate": engine.temperature(),
        }
    });
    std::fs::write(&json_path, serde_json::to_string_pretty(&payload)?)?;
    println!();
    println!("JSON results written to: {}", json_path);

    Ok(())
}
