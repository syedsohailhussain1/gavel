#!/usr/bin/env python3
"""Test the tiny sentiment ONNX model on CPU.

Usage:
    python test_model.py            # from the models/ directory

What it does:
  1. Loads models/tinybert-sst2/model.onnx with onnxruntime (CPU only).
  2. Classifies 8 unambiguous movie-review snippets, printing the
     predicted label and confidence for each.
  3. Exits non-zero if ANY sample is misclassified.
  4. Benchmarks 200 inferences and reports p50/p99 latency.

Needs: pip install onnxruntime tokenizers numpy
"""

import os
import statistics
import sys
import time

# Same urllib/no_proxy workaround as download.py (bracketed IPv6 entries
# in no_proxy break Python's proxy handling with "Invalid port").
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val and "[" in _val:
        os.environ[_var] = ",".join(p for p in _val.split(",") if "[" not in p)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "tinybert-sst2")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

# (text, expected_label) — kept deliberately unambiguous.
SAMPLES = [
    ("A wonderful, heartwarming film with brilliant performances.", "positive"),
    ("I absolutely loved every minute of this movie.", "positive"),
    ("The best comedy I have seen in years, hilarious from start to finish.", "positive"),
    ("Beautiful cinematography and a deeply moving story. Highly recommended.", "positive"),
    ("A boring, painful waste of two hours. I hated every minute.", "negative"),
    ("Terrible acting and a nonsensical plot. Avoid at all costs.", "negative"),
    ("I fell asleep halfway through. Completely dull and lifeless.", "negative"),
    ("The worst film of the year: unfunny, ugly, and interminable.", "negative"),
]

# SST-2 label convention: 0 = negative, 1 = positive.
ID2LABEL = {0: "negative", 1: "positive"}


def load():
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Run: python download.py")
        raise SystemExit(2)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    def classify(text):
        enc = tokenizer.encode(text)
        ids = enc.ids[:128]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)
        logits = session.run(None, feed)[0][0]
        # softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        return ID2LABEL[idx], float(probs[idx])

    return classify


def main() -> int:
    classify = load()

    print(f"Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)\n")
    failures = 0
    for text, expected in SAMPLES:
        label, conf = classify(text)
        ok = "OK  " if label == expected else "FAIL"
        if label != expected:
            failures += 1
        print(f"[{ok}] {label:>8} ({conf:.3f})  expected={expected:>8}  :: {text[:60]}")

    # Benchmark: 200 inferences of a fixed input.
    bench_text = "A wonderful, heartwarming film with brilliant performances."
    for _ in range(10):  # warmup
        classify(bench_text)
    timings = []
    for _ in range(200):
        t0 = time.perf_counter()
        classify(bench_text)
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    p50 = statistics.median(timings)
    p99 = timings[int(0.99 * len(timings))]

    print(f"\nLatency over 200 CPU inferences: p50={p50:.2f} ms  p99={p99:.2f} ms")
    if failures:
        print(f"\n{failures} sample(s) misclassified -> FAIL")
        return 1
    print("\nAll 8 samples classified correctly -> PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
