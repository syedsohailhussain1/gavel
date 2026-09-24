#!/usr/bin/env python3
"""NLI supplement pairs (entailment/contradiction only, neutral skipped)."""
import json
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
OUT = sys.argv[2] if len(sys.argv) > 2 else "nli_pairs.jsonl"

from datasets import load_dataset

ds = load_dataset("SetFit/mnli", split="train").shuffle(seed=7)
rows, n = [], 0
for r in ds:
    if n >= N:
        break
    if r["label"] not in (0, 2):
        continue
    truth = "true" if r["label"] == 0 else "false"
    other = "false" if truth == "true" else "true"
    base = f"State: {r['text1']}\nQuestion: Does the hypothesis follow?\n"
    rows.append({"text": base + f"Option: {truth}", "target": 1,
                 "item": f"mnli-{n}", "tier": "nli"})
    rows.append({"text": base + f"Option: {other}", "target": 0,
                 "item": f"mnli-{n}", "tier": "nli"})
    n += 1
json.dump(rows, open(OUT, "w"))
print(f"nli pairs={len(rows)} from {n} items -> {OUT}", flush=True)
