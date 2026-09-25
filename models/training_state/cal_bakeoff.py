#!/usr/bin/env python3
"""Calibration bake-off on official-run distributions (CPU, minutes).

Contenders, all fit on the same 231 records:
  A. global temperature (current: T_total=0.535, ECE 0.070)
  B. per-tier temperatures (3 params; disclosed if shipped)
  C. cross-fitted isotonic regression (5-fold, non-parametric)
Reports ECE + NLL each. Winner must beat 0.070 by margin, else keep A.
"""
import json
import math
import sys

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from m2 import expected_calibration_error

RES = r"C:\Users\Sohail\AppData\Local\Temp\opencode\JB_ALL.jsonl"
TIER_OF = {}
for p in (r"D:\jevbench\datasets\public\easy.jsonl",
          r"D:\jevbench\datasets\public\original.jsonl",
          r"D:\jevbench\datasets\public\hard.jsonl"):
    tier = p.split("\\")[-1].split(".")[0]
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        TIER_OF[r["id"]] = tier

recs = [json.loads(l) for l in open(RES, encoding="utf-8")]
items = []
for r in recs:
    probs = r.get("probs") or {}
    if not probs:
        continue
    gold = None
    # gold recovered as the expected label is NOT in results; use correct flag
    # with argmax label: correct => gold is argmax; else gold unknown.
    # So ECE-only evaluation (no NLL refit here): use recorded maxprob.
    jm = max(probs, key=lambda k: probs[k])
    items.append((TIER_OF.get(r["task_id"], "?"), float(probs[jm]),
                  bool(r.get("correct"))))


def ece(rows):
    c = [x[1] for x in rows]
    k = [x[2] for x in rows]
    return expected_calibration_error(c, k)


print(f"n={len(items)} overall ECE(current)={ece(items):.4f}", flush=True)
for t in ("easy", "original", "hard"):
    sub = [x for x in items if x[0] == t]
    print(f"  {t}: n={len(sub)} ECE={ece(sub):.4f}", flush=True)

# B: per-tier constant rescaling is meaningless for ECE without logits;
# instead report the honest decomposition: what ECE would a per-tier
# temperature need to fix? (informational; shipping per-tier T needs logits)
# C: isotonic needs (confidence, correctness) pairs — 5-fold cross-fit:
import random
rng = random.Random(7)
idx = list(range(len(items)))
rng.shuffle(idx)
folds, K = [[] for _ in range(5)], 5
for j, i in enumerate(idx):
    folds[j % K].append(i)


def isotonic_fit(pts):
    # PAVA on sorted unique confidences -> stepwise calibrator.
    pts = sorted(pts)
    blocks = [[c, k, 1] for c, k in pts]  # (sum, n) per block via [s, n]
    blocks = [[c, 1] for c, k in pts]
    vals = [k for _, k in pts]
    # pool adjacent violators on block means
    means = [float(v) for v in vals]
    wt = [1] * len(vals)
    i = 0
    seq = [[means[i], wt[i]]]
    for i in range(1, len(vals)):
        seq.append([float(vals[i]), 1])
        while len(seq) >= 2 and seq[-2][0] > seq[-1][0]:
            m2, w2 = seq.pop()
            m1, w1 = seq.pop()
            seq.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    # expand to (threshold, value) steps
    out, k = [], 0
    for m, w in seq:
        out.append((pts[k][0], m))
        k += w
    return out


def iso_apply(model, c):
    v = model[0][1]
    for th, m in model:
        if c >= th:
            v = m
    return v


oof_c, oof_k = [], []
for f in range(K):
    tr = [items[i] for j in range(K) if j != f for i in folds[j]]
    te = [items[i] for i in folds[f]]
    model = isotonic_fit([(c, int(k)) for _, c, k in tr])
    for _, c, k in te:
        oof_c.append(iso_apply(model, c))
        oof_k.append(k)
print(f"C isotonic 5-fold OOF ECE={expected_calibration_error(oof_c, oof_k):.4f}",
      flush=True)
print("CAL-BAKEOFF DONE", flush=True)
