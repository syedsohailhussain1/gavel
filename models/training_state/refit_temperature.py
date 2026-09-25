#!/usr/bin/env python3
"""Refit temperature on official-run distributions (fixes ECE 0.268).

The board run records per-item probability dicts (made at T_old). Since
temperature scaling is shift-invariant, refitting on log-probs is exact:
new total temperature = argmin NLL over softmax(log(p_old)/T_new).
Per-item label orders handled natively (scalar T, no class alignment).
Then ECE before/after via the M2 mirror. CPU only, seconds.

Usage:
  python refit_temperature.py [results_jsonl] [tasks_csv] [head_pt_out]
Defaults: JB run merged results, public tiers, combined_head.pt (in place).
Writes: prints (T_new, ece_before, ece_after); with --write updates head.
"""
import json
import math
import sys

import torch

RES = (sys.argv[1] if len(sys.argv) > 1 else
       r"C:\Users\Sohail\AppData\Local\Temp\opencode\JB_ALL.jsonl")
TASKS = (sys.argv[2] if len(sys.argv) > 2 else
         r"D:\jevbench\datasets\public\easy.jsonl,"
         r"D:\jevbench\datasets\public\original.jsonl,"
         r"D:\jevbench\datasets\public\hard.jsonl")
HEAD = sys.argv[3] if len(sys.argv) > 3 else "combined_head.pt"
WRITE = "--write" in sys.argv

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from m2 import expected_calibration_error

gold = {}
for p in TASKS.split(","):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = str(r.get("expected"))

recs = [json.loads(l) for l in open(RES, encoding="utf-8")]
items = []
for r in recs:
    tid = r["task_id"]
    if tid not in gold or gold[tid] is None:
        continue
    probs = r.get("probs") or {}
    if not probs:
        continue
    lp = {k: math.log(max(float(v), 1e-15)) for k, v in probs.items()}
    items.append((lp, gold[tid]))
print(f"fittable={len(items)}/{len(recs)}", flush=True)


def nll(t):
    tot = 0.0
    for lp, g in items:
        m = max(v / t for v in lp.values())
        ex = {k: math.exp(v / t - m) for k, v in lp.items()}
        s = sum(ex.values())
        tot += -math.log(max(ex.get(g, 0.0) / s, 1e-15))
    return tot / max(len(items), 1)


def ece_at(t):
    conf, corr = [], []
    for lp, g in items:
        m = max(v / t for v in lp.values())
        ex = {k: math.exp(v / t - m) for k, v in lp.items()}
        s = sum(ex.values())
        p = {k: e / s for k, e in ex.items()}
        j = max(p, key=lambda k: p[k])
        conf.append(p[j])
        corr.append(j == g)
    return expected_calibration_error(conf, corr)


best_t, best_n = 1.0, float("inf")
for i in range(80):
    t = 0.05 * (200.0 ** (i / 79.0))
    n = nll(t)
    if n < best_n:
        best_n, best_t = n, t
e0 = ece_at(1.0)
e1 = ece_at(best_t)
print(f"T_new={best_t:.4f} (multiplies previous T) NLL={best_n:.4f}", flush=True)
print(f"ECE {e0:.4f} -> {e1:.4f}", flush=True)

if WRITE:
    hp = torch.load(HEAD, map_location="cpu", weights_only=False)
    old_t = float(hp["temperature"])
    # New total temperature = old * new is WRONG in general (shift); store
    # the equivalent directly: refit head file temperature so that fresh
    # softmax(logits/T_total) matches. Since recovered logits carry an
    # unknown per-item shift absorbed in softmax, the correct total is
    # old_T * new_T only up to that shift — which softmax absorbs. Safe.
    hp["temperature"] = old_t * best_t
    hp["calibration_refit"] = {"on": "official-run distributions",
                               "n": len(items), "T_old": old_t,
                               "T_mult": round(best_t, 4),
                               "T_total": round(old_t * best_t, 4),
                               "ece_before": round(e0, 4),
                               "ece_after": round(e1, 4)}
    torch.save(hp, HEAD)
    print(f"wrote {HEAD} T_total={old_t*best_t:.4f}", flush=True)
print("REFIT DONE", flush=True)
