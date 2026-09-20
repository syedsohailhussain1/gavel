//! decide-phishing: reproduce the jev-phishing-bench evaluation with gavel.
//!
//! Reads pre-split JSONL files of {"input": <text>, "label": "phishing"|"legitimate"},
//! trains [`LogisticEngine`], calibrates with temperature scaling on validation,
//! and evaluates on the held-out test split.
//!
//! Data prep (download, seeded shuffle, the published A/B split, train/val split)
//! is done by benches/phishing_repro.py, which keeps third-party data in /tmp.
//! This binary never touches the network and embeds no dataset.
//!
//! Usage:
//!   decide-phishing <train.jsonl> <val.jsonl> <test.jsonl>
//!
//! Prints a single JSON object with metrics to stdout. ECE is computed with the
//! exact binning of the reference benchmark (10 equal-width bins over [0.5, 1.0]
//! on the predicted-class confidence); gavel's native ECE ([0,1] bins) is also
//! reported for reference.

use decide_core::{expected_calibration_error, LogisticEngine};
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn read_jsonl(path: &str) -> Vec<(String, String)> {
    let f = File::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    BufReader::new(f)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.unwrap_or_else(|e| panic!("read error in {path} line {i}: {e}"));
            let v: serde_json::Value =
                serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad JSON {path}:{i}: {e}"));
            let input = v["input"]
                .as_str()
                .unwrap_or_else(|| panic!("missing input {path}:{i}"));
            let label = v["label"]
                .as_str()
                .unwrap_or_else(|| panic!("missing label {path}:{i}"));
            (input.to_string(), label.to_string())
        })
        .collect()
}

/// AUROC via Mann-Whitney U with tie-averaged ranks (same method as the
/// reference benchmark's analyze.py).
fn auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let mut items: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    // Rust's sort_by is stable, matching mergesort argsort.
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = items.len() as f64;
    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = n - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    // Average ranks for ties (1-based).
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < items.len() {
        let mut j = i;
        while j + 1 < items.len() && items[j + 1].0 == items[i].0 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for item in &items[i..=j] {
            if item.1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// ECE with the reference benchmark's binning: 10 equal-width bins over
/// [0.5, 1.0] on the predicted-class confidence (max(p, 1-p)).
///
/// Edges are precomputed once from a single formula (like np.linspace), so
/// adjacent bins share exact edge values and no confidence is double-counted.
fn ece_reference(confidences: &[f64], correct: &[bool]) -> f64 {
    let n_bins = 10usize;
    let n = confidences.len() as f64;
    let edges: Vec<f64> = (0..=n_bins).map(|b| 0.5 + b as f64 * 0.05).collect();
    let mut total = 0.0;
    for b in 0..n_bins {
        let (lo, hi) = (edges[b], edges[b + 1]);
        let mut bin_n = 0u64;
        let mut acc_sum = 0.0;
        let mut conf_sum = 0.0;
        for (c, ok) in confidences.iter().zip(correct.iter()) {
            let inside = if b == n_bins - 1 {
                *c >= lo && *c <= hi
            } else {
                *c >= lo && *c < hi
            };
            if inside {
                bin_n += 1;
                acc_sum += if *ok { 1.0 } else { 0.0 };
                conf_sum += c;
            }
        }
        if bin_n > 0 {
            let bn = bin_n as f64;
            total += bn / n * ((acc_sum - conf_sum).abs() / bn);
        }
    }
    total
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: decide-phishing <train.jsonl> <val.jsonl> <test.jsonl>");
        std::process::exit(2);
    }
    let train = read_jsonl(&args[1]);
    let val = read_jsonl(&args[2]);
    let test = read_jsonl(&args[3]);
    eprintln!(
        "loaded: train={} val={} test={}",
        train.len(),
        val.len(),
        test.len()
    );

    let mut engine = LogisticEngine::new();
    let t0 = Instant::now();
    engine.train(&train).expect("train failed");
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    engine.calibrate(&val).expect("calibrate failed");
    let calibrate_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let classes = engine.classes().to_vec();
    assert_eq!(
        classes,
        vec!["legitimate".to_string(), "phishing".to_string()]
    );
    let phish_idx = 1; // classes are sorted: legitimate < phishing

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    let mut tn = 0u64;
    let mut p_phish: Vec<f64> = Vec::with_capacity(test.len());
    let mut is_phish: Vec<bool> = Vec::with_capacity(test.len());
    let mut confidences: Vec<f64> = Vec::with_capacity(test.len());
    let mut correct: Vec<bool> = Vec::with_capacity(test.len());
    let mut lat_us: Vec<f64> = Vec::with_capacity(test.len());

    for (text, label) in &test {
        let actual_phish = label == "phishing";
        let t0 = Instant::now();
        let (best, conf, probs) = engine.predict(text).expect("predict failed");
        lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
        let pred_phish = best == phish_idx;
        match (actual_phish, pred_phish) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => tn += 1,
        }
        p_phish.push(probs[phish_idx]);
        is_phish.push(actual_phish);
        confidences.push(conf);
        correct.push(pred_phish == actual_phish);
    }

    let n = test.len() as f64;
    let accuracy = (tp + tn) as f64 / n;
    let recall = tp as f64 / (tp + fn_) as f64;
    let fpr = fp as f64 / (fp + tn) as f64;
    let precision = tp as f64 / (tp + fp) as f64;
    lat_us.sort_by(|a, b| a.total_cmp(b));

    let out = json!({
        "n_test": test.len(),
        "n_train": train.len(),
        "n_val": val.len(),
        "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
        "accuracy": accuracy,
        "recall_phishing": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": 2.0 * precision * recall / (precision + recall),
        "auroc": auroc(&p_phish, &is_phish),
        "ece_reference_binning": ece_reference(&confidences, &correct),
        "ece_gavel_binning": expected_calibration_error(&confidences, &correct, 10),
        "temperature": engine.temperature(),
        "ece_before_calibration": engine.ece_before(),
        "ece_after_calibration": engine.ece_after(),
        "train_ms": train_ms,
        "calibrate_ms": calibrate_ms,
        "ask_latency_us": {
            "p50": percentile(&lat_us, 50.0),
            "p99": percentile(&lat_us, 99.0),
            "mean": lat_us.iter().sum::<f64>() / lat_us.len() as f64,
        },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auroc_matches_hand_computation() {
        // pos {0.35, 0.8}, neg {0.1, 0.4}: 3 of 4 pairs concordant -> 0.75
        let scores = vec![0.1, 0.4, 0.35, 0.8];
        let labels = vec![false, false, true, true];
        assert!((auroc(&scores, &labels) - 0.75).abs() < 1e-12);
        // perfect separation
        let labels2 = vec![false, false, true, true];
        let scores2 = vec![0.1, 0.2, 0.8, 0.9];
        assert!((auroc(&scores2, &labels2) - 1.0).abs() < 1e-12);
        // ties get half credit: all equal -> 0.5
        let scores3 = vec![0.5, 0.5, 0.5, 0.5];
        assert!((auroc(&scores3, &labels2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ece_reference_matches_binning_definition() {
        // Two predictions at confidence 0.9, one right one wrong, in bin [0.85,0.9):
        // |0.5 - 0.9| * (2/2) = 0.4
        let e = ece_reference(&[0.9, 0.9], &[true, false]);
        assert!((e - 0.4).abs() < 1e-12, "got {e}");
        // Perfectly calibrated single prediction -> 0
        let e2 = ece_reference(&[1.0], &[true]);
        assert!(e2 < 1e-12, "got {e2}");
    }
}
