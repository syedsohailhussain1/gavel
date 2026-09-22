#!/usr/bin/env python3
"""SeqKD relabel: strong transformer teacher re-labels states for the linear
student (PACT MAIN doctrine: teacher runs once offline, knowledge flows
through data). GPU-batched for throughput.

Usage:
  python seqkd_relabel.py [teacher_dir] [data_json] [out_json] [batch]
Defaults: tiny_snake2_distil tiny_snake4_train.jsonl tiny_snake4kd_train.jsonl 512
Keeps text+phase, replaces label with teacher argmax. Reports agreement rate.
"""
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TDIR = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake2_distil"
DATA = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake4_train.jsonl"
OUT = sys.argv[3] if len(sys.argv) > 3 else "tiny_snake4kd_train.jsonl"
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 512
STRIP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
LABELS = ["up", "down", "left", "right"]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(TDIR)
tm = AutoModelForSequenceClassification.from_pretrained(TDIR).to(dev).eval()
rows = json.load(open(DATA, encoding="utf-8"))
print(f"relabeling {len(rows)} with {TDIR} on {dev}", flush=True)

agree = 0
with torch.no_grad():
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        texts = [r["text"] for r in chunk]
        if STRIP:
            # Teacher never saw phase prefixes: strip for in-distribution
            # labeling; labels attach to the full phased rows.
            texts = [t.split("phase hunt ", 1)[-1].split("phase survive ", 1)[-1]
                     if t.startswith("phase ") else t for t in texts]
        e = tok(texts, return_tensors="pt",
                truncation=True, max_length=96, padding=True)
        e = {k: v.to(dev) for k, v in e.items()}
        p = tm(**e).logits.argmax(-1).cpu().tolist()
        for r, a in zip(chunk, p):
            agree += (LABELS[a] == r["label"])
            r["label"] = LABELS[a]
json.dump(rows, open(OUT, "w"))
print(f"wrote {OUT} teacher_agreement={agree/max(len(rows),1):.4f}", flush=True)
