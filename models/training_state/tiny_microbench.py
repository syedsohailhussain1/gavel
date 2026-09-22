#!/usr/bin/env python3
"""Manual microbench: Google-tiny Snake model, batch-1, per-decision ms.

Usage:
  python tiny_microbench.py [onnx] [tok_dir] [threads] [reps]
Defaults: tiny_snake_gtiny_fp32.onnx tiny_snake_gtiny 4 300
Reports: tokenize ms, infer ms, end-to-end ms/decision, decisions/sec,
decisions per ms. Batch-1, single stream — the honest interactive number.
"""
import sys
import time

import numpy as np
import onnxruntime as ort

ONNX = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake_gtiny_fp32.onnx"
TOK = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake_gtiny"
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TOK)
opts = ort.SessionOptions()
opts.intra_op_num_threads = THREADS
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(ONNX, sess_options=opts, providers=["CPUExecutionProvider"])

TEXTS = [
    "head(2,2) apple(9,5) len12 heading right || up:free f8 s90 | down:body f0 s0 | left:free f12 s30 | right:tail f6 s100",
    "head(0,0) apple(11,11) len40 heading down || up:wall f0 s0 | down:free f21 s60 | left:wall f0 s0 | right:free f21 s55",
    "head(6,6) apple(6,2) len8 heading up || up:free f4 s100 | down:body f0 s0 | left:free f6 s80 | right:free f6 s80",
]
LI = ["up", "down", "left", "right"]

# Warmup.
for _ in range(30):
    e = tok(TEXTS[0], return_tensors="np", truncation=True, max_length=96)
    sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                    "attention_mask": e["attention_mask"].astype(np.int64)})

t_tok, t_inf, t_e2e, outs = [], [], [], []
for i in range(REPS):
    t = TEXTS[i % len(TEXTS)]
    a = time.perf_counter()
    e = tok(t, return_tensors="np", truncation=True, max_length=96)
    b = time.perf_counter()
    logits = sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                             "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
    c = time.perf_counter()
    t_tok.append((b - a) * 1000)
    t_inf.append((c - b) * 1000)
    t_e2e.append((c - a) * 1000)
    outs.append(LI[int(logits.argmax())])


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(len(x) * p))]


e2e = sorted(t_e2e)[len(t_e2e) // 2]
print(f"threads={THREADS} reps={REPS} batch=1")
print(f"tokenize p50 : {pct(t_tok, 0.5):.3f} ms")
print(f"infer    p50 : {pct(t_inf, 0.5):.3f} ms  p99: {pct(t_inf, 0.99):.3f} ms")
print(f"end2end  p50 : {e2e:.3f} ms  p99: {pct(t_e2e, 0.99):.3f} ms")
print(f"decisions/sec: {1000 / e2e:.1f}")
print(f"decisions/ms : {1 / e2e:.4f}")
print(f"sample moves : {' '.join(outs[:12])}")
