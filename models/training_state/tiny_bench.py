#!/usr/bin/env python3
"""Benchmark the tiny Snake ONNX models: parity, accuracy, latency.

Usage:
  python tiny_bench.py [model_dir] [fp32_onnx] [int8_onnx] [val_jsonl] [n]
Defaults: tiny_snake_modernbert tiny_snake_modernbert_fp32.onnx
          tiny_snake_modernbert_int8.onnx tiny_snake_val.jsonl 500
Reports per-file argmax agreement vs torch, val accuracy, p50/p99 ms, q/s.
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

D = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake_modernbert"
F32 = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake_modernbert_fp32.onnx"
I8 = sys.argv[3] if len(sys.argv) > 3 else "tiny_snake_modernbert_int8.onnx"
VAL = sys.argv[4] if len(sys.argv) > 4 else "tiny_snake_val.jsonl"
N = int(sys.argv[5]) if len(sys.argv) > 5 else 500
LI = {"up": 0, "down": 1, "left": 2, "right": 3}

val = json.load(open(VAL, encoding="utf-8"))[:N]
tok = AutoTokenizer.from_pretrained(D)
tm = AutoModelForSequenceClassification.from_pretrained(
    D, attn_implementation="eager").eval()
opts = ort.SessionOptions()
opts.intra_op_num_threads = 8
sess = {n: ort.InferenceSession(p, sess_options=opts, providers=["CPUExecutionProvider"])
        for n, p in (("fp32", F32), ("int8", I8))}
for s in sess.values():  # warmup
    s.run(None, {"input_ids": np.ones((1, 32), dtype=np.int64),
                 "attention_mask": np.ones((1, 32), dtype=np.int64)})

stat = {n: {"ok": 0, "agree": 0, "mx": 0.0, "lat": []} for n in sess}
for r in val:
    e = tok(r["text"], return_tensors="np", truncation=True, max_length=96)
    ii = e["input_ids"].astype(np.int64)
    am = e["attention_mask"].astype(np.int64)
    with torch.no_grad():
        ref = tm(torch.from_numpy(ii), torch.from_numpy(am)).logits.numpy()[0]
    yt = LI[r["label"]]
    for n, s in sess.items():
        t0 = time.perf_counter()
        got = s.run(None, {"input_ids": ii, "attention_mask": am})[0][0]
        stat[n]["lat"].append((time.perf_counter() - t0) * 1000)
        stat[n]["mx"] = max(stat[n]["mx"], float(np.abs(ref - got).max()))
        pred = int(got.argmax())
        stat[n]["agree"] += (pred == int(ref.argmax()))
        stat[n]["ok"] += (pred == yt)

for n, s in stat.items():
    lat = sorted(s["lat"])
    p50 = lat[len(lat) // 2]
    p99 = lat[int(len(lat) * 0.99)]
    print(json.dumps({
        "model": n, "n": N, "val_acc": round(s["ok"] / N, 4),
        "argmax_agree_torch": round(s["agree"] / N, 4),
        "max_logit_diff": round(s["mx"], 4),
        "p50_ms": round(p50, 3), "p99_ms": round(p99, 3),
        "per_sec": round(1000 / p50, 1),
    }), flush=True)
