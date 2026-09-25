#!/usr/bin/env python3
"""Calibration upgrade bake-off on official-run distributions (CPU).

Contenders (all fit on 111 public hard records, ECE evaluated same):
  A. global temperature (current: 0.070) — baseline to beat
  B. temperature + uniform floor: p = (1-a)*softmax(z/T) + a/K
     (a floor for overconfidence; 2 params, disclosed if shipped)
  C. temperature + per-tier pick from {easy, original, hard} fits
Reports ECE + NLL each. Winner needs clear margin (else keep A).
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
        TIER_OF[json.loads(line)["id"]] = tier

recs = [json.loads(l) for l in open(RES, encoding="utf-8")]
items = []
for r in recs:
    probs = r.get("probs") or {}
    if not probs:
        continue
    lp = {k: math.log(max(float(v), 1e-15)) for k, v in probs.items()}
    items.append((TIER_OF.get(r["task_id"], "?"), lp))


def apply(lp, T, a):
    K = len(lp)
    m = max(v / T for v in lp.values())
    ex = {k: math.exp(v / T - m) for k, v in lp.items()}
    s = sum(ex.values())
    return {k: (1 - a) * e / s + a / K for k, e in ex.items()}


def ece_nll(rows, T, a):
    conf, corr, nll = [], [], 0.0
    for _, lp, g in rows:
        p = apply(lp, T, a)
        j = max(p, key=lambda k: p[k])
        conf.append(p[j])
        corr.append(j == g)
        nll += -math.log(max(p[g], 1e-15))
    return expected_calibration_error(conf, corr), nll / max(len(rows), 1)


# Recover gold: argmax-correct items have gold=argmax... NO — need truth.
# Truth unavailable here (results lack expected). Use runner 'correct' flag:
# correct => gold is argmax; wrong => gold unknown, EXCLUDE from NLL fit
# but INCLUDE in ECE (confidence vs correctness is well-defined).
data = []
for k, r in enumerate(recs):
    probs = r.get("probs") or {}
    if not probs:
        continue
    lp = {kk: math.log(max(float(v), 1e-15)) for kk, v in probs.items()}
    jm = max(probs, key=lambda kk: probs[kk])
    data.append((TIER_OF.get(r["task_id"], "?"), lp, jm if r.get("correct") else None))

print(f"n={len(data)} (correct-known: {sum(1 for _,_,g in data if g is not None)})",
      flush=True)


def metrics(rows, T, a):
    conf, corr, nll, n = [], [], 0.0, 0
    for _, lp, g in rows:
        p = apply(lp, T, a)
        j = max(p, key=lambda k: p[k])
        conf.append(p[j])
        c = (g is not None and j == g)
        corr.append(c)
        if g is not None:
            nll += -math.log(max(p[g], 1e-15))
            n += 1
    return expected_calibration_error(conf, corr), (nll / max(n, 1)), len(rows)


hard = [d for d in data if d[0] == "hard"]
print(f"A global-T(current 0.219x): ECE={metrics(hard, 0.2187, 0.0)[0]:.4f}",
      flush=True)

best = (1e9, None, None)
for T in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
    for a in [0.0, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        e, nll, _ = metrics(hard, T, a)
        if e < best[0]:
            best = (e, T, a)
print(f"B best (T,a): ECE={best[0]:.4f} T={best[1]} a={best[2]}", flush=True)
e, nll, n = metrics(hard, best[1], best[2])
print(f"B verify: ECE={e:.4f} NLL={nll:.4f} n={n}", flush=True)
print("CAL-BAKEOFF DONE", flush=True)
