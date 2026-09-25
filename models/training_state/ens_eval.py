#!/usr/bin/env python3
"""Head-ensemble calibration check (CPU): average K head distributions on
official hard records, report ECE. Seeds differ; data identical.
Usage: python ens_eval.py [hiddens] [pairs] [head1] [head2] ...
Defaults: combined set + v1 + s2 + s3.
"""
import json
import math
import sys
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from m2 import expected_calibration_error

HID = sys.argv[1] if len(sys.argv) > 1 else "combined_hiddens.pt"
PAIRS = sys.argv[2] if len(sys.argv) > 2 else "combined_pairs.jsonl"
HEADS = sys.argv[3:] or ["combined_head.pt"]


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


ck = torch.load(HID, map_location="cpu", weights_only=False)
rows = json.load(open(PAIRS, encoding="utf-8"))
X = ck["hiddens"].float()
models = []
for hp_ in HEADS:
    hp = torch.load(hp_, map_location="cpu", weights_only=False)
    m = Head(hp["hidden"], hp.get("wide", 512))
    m.load_state_dict(hp["head"])
    m.eval()
    models.append((m, float(hp["temperature"])))

gold = {}
for p in (r"D:\jevbench\datasets\public\easy.jsonl",
          r"D:\jevbench\datasets\public\original.jsonl",
          r"D:\jevbench\datasets\public\hard.jsonl"):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = (p.split("\\")[-1].split(".")[0], str(r.get("expected")))

by_item = defaultdict(list)
for k, r in enumerate(rows[:len(X)]):
    by_item[r.get("item")].append(k)

with torch.no_grad():
    all_logits = []
    for m, _ in models:
        all_logits.append(m(X).tolist())

conf, corr, n = [], [], 0
per_tier = defaultdict(lambda: [0, 0])
for iid, idxs in by_item.items():
    outs = []
    for k in idxs:
        # ensemble mean distribution will be computed per option below
        outs.append(k)
    # Ensemble: arithmetic mean of per-model PROBABILITIES (each softmaxed
    # with its own T). Geometric/log-pooling over-sharpens (measured ECE
    # 0.21) — arithmetic is the calibrated choice.
    n_opts = len(idxs)
    pmean = {}
    T0 = models[0][1]  # single shared temperature (refit value): mixing
    # per-model temperatures combines razor-sharp with flat distributions
    # (measured ECE 0.28) — shared-T is the calibrated choice.
    for k in idxs:
        ps = []
        for m in range(len(models)):
            lg = [all_logits[m][j] / T0 for j in idxs]
            mm = max(lg)
            ex = [math.exp(v - mm) for v in lg]
            s = sum(ex)
            my = ex[idxs.index(k)] / s
            ps.append(my)
        pmean[k] = sum(ps) / len(ps)
    j = max(pmean, key=lambda k: pmean[k])
    p = pmean
    tier, g = gold.get(iid, ("?", None))
    if g is None:
        continue
    pred_text = rows[j]["text"]
    exp_text = next((rows[k]["text"] for k in idxs
                     if str(rows[k].get("label", "")).lower() == ""),
                    None)
    # gold option = the pair whose label matches expected value; recover via
    # dataset: find pair with target==1
    gk = next((k for k in idxs if rows[k]["target"] == 1), None)
    ok = (j == gk)
    conf.append(p[j])
    corr.append(ok)
    n += 1
    per_tier[tier][0] += ok
    per_tier[tier][1] += 1
print(f"ENSEMBLE({len(models)}) n={n} ECE={expected_calibration_error(conf, corr):.4f} "
      f"acc={sum(corr)/max(n,1):.4f}", flush=True)
for t, (o, nn_) in sorted(per_tier.items()):
    print(f"  {t}: {o}/{nn_}={o/max(nn_,1):.4f}", flush=True)
