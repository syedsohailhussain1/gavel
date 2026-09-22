#!/usr/bin/env python3
"""GPU shootout: every tiny Snake model on CUDA, batch-1 latency + accuracy.

Usage:
  python tiny_gpu_bench.py [reps] [val_n]
Defaults: 300 300
Models (dir, val file) are fixed below. Loads one at a time (4GB VRAM).
Reports per model: params, val acc, batch-1 ms (sync-timed), decisions/sec,
batch-32 throughput, VRAM MB.
"""
import json
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
VALN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LI = {"up": 0, "down": 1, "left": 2, "right": 3}

MODELS = [
    ("gtiny-4M", "models/training_state/tiny_snake_gtiny",
     "models/training_state/tiny_snake_val.jsonl"),
    ("distil-66M", "models/training_state/tiny_snake_distil",
     "models/training_state/tiny_snake_val.jsonl"),
    ("distil-v2-66M", "models/training_state/tiny_snake2_distil",
     "models/training_state/tiny_snake2_val.jsonl"),
    ("modernbert-150M", "models/training_state/tiny_snake_modernbert",
     "models/training_state/tiny_snake_val.jsonl"),
]

assert torch.cuda.is_available(), "no CUDA"
print("gpu:", torch.cuda.get_device_name(0), flush=True)
out = []
for name, d, vf in MODELS:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForSequenceClassification.from_pretrained(d).cuda().eval()
    n_params = sum(p.numel() for p in m.parameters())
    val = json.load(open(vf, encoding="utf-8"))[:VALN]

    def enc(texts):
        e = tok(texts, return_tensors="pt", truncation=True,
                max_length=96, padding=True)
        return {k: v.cuda() for k, v in e.items()}

    # Accuracy.
    ok = 0
    with torch.no_grad():
        for i in range(0, len(val), 64):
            chunk = val[i:i + 64]
            p = m(**enc([r["text"] for r in chunk])).logits.argmax(-1).cpu().tolist()
            ok += sum(int(a) == LI[r["label"]] for a, r in zip(p, chunk))
    acc = ok / len(val)

    # Batch-1 latency, properly synchronized.
    e1 = enc([val[0]["text"]])
    with torch.no_grad():
        for _ in range(50):
            m(**e1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            m(**e1)
        torch.cuda.synchronize()
        b1 = (time.perf_counter() - t0) / REPS * 1000

    # Batch-32 throughput.
    e32 = enc([r["text"] for r in val[:32]])
    with torch.no_grad():
        for _ in range(10):
            m(**e32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            m(**e32)
        torch.cuda.synchronize()
        tput = 32 / ((time.perf_counter() - t0) / 50)
    vram = torch.cuda.max_memory_allocated() / 1e6
    row = {"model": name, "params_M": round(n_params / 1e6, 1),
           "val_acc": round(acc, 4), "batch1_ms": round(b1, 3),
           "per_sec_b1": round(1000 / b1, 1),
           "per_sec_b32": round(tput, 1), "vram_MB": round(vram, 0)}
    print(json.dumps(row), flush=True)
    out.append(row)
    del m
    torch.cuda.empty_cache()
print("Jev reference: 436 ms p50 → multiples: " +
      ", ".join(f"{r['model']} {round(436 / r['batch1_ms'])}x" for r in out),
      flush=True)
