#!/usr/bin/env python3
"""Meta-calibrator: predict P(correct) per item from decision signals.

Temperature rescales everything blindly. A correctness head conditions on
what each decision LOOKS like: logit margin, entropy, #options, tier.
If confident-wrong items are separable from confident-right ones, this
beats any global rescaling. L2 logistic, 5-fold OOF, CPU minutes.

Serving form (if shipped): keep temperature distribution SHAPE, set the
top-label confidence to meta P(correct), rescale the rest proportionally.
Full distributions preserved for the fidelity half of the axis.
"""
import json
import math
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from m2 import expected_calibration_error

RES = r"C:\Users\Sohail\AppData\Local\Temp\opencode\JB_ALL.jsonl"
TIER_OF, TLEN = {}, {}
for p in (r"D:\jevbench\datasets\public\easy.jsonl",
          r"D:\jevbench\datasets\public\original.jsonl",
          r"D:\jevbench\datasets\public\hard.jsonl"):
    tier = p.split("\\")[-1].split(".")[0]
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        TIER_OF[r["id"]] = tier
        st = r["state"]
        TLEN[r["id"]] = len(st if isinstance(st, str) else json.dumps(st))

TIERS = ["easy", "original", "hard"]
rows = []
for r in (json.loads(l) for l in open(RES, encoding="utf-8")):
    probs = r.get("probs") or {}
    if not probs:
        continue
    ps = sorted((float(v) for v in probs.values()), reverse=True)
    top, second = ps[0], (ps[1] if len(ps) > 1 else 0.0)
    ent = -sum(v * math.log(max(v, 1e-15)) for v in probs.values())
    t = TIER_OF.get(r["task_id"], "?")
    rows.append({"c": top, "margin": top - second, "ent": ent,
                 "kopts": math.log(max(len(probs), 2)),
                 "tier": t, "len": math.log(max(TLEN.get(r["task_id"], 100), 10)),
                 "y": int(bool(r.get("correct")))})

print(f"n={len(rows)} base-rate={sum(r['y'] for r in rows)/len(rows):.3f}",
      flush=True)
FEATS = ["c", "margin", "ent", "kopts", "len"]


def vec(r):
    return [r["c"], r["margin"], r["ent"], r["kopts"], r["len"]] + \
        [1.0 if r["tier"] == t else 0.0 for t in TIERS]


X = np.array([vec(r) for r in rows])
mu, sd = X.mean(0), X.std(0) + 1e-9
Xn = (X - mu) / sd
y = np.array([r["y"] for r in rows])

# 5-fold OOF confidences
oof = np.zeros(len(rows))
rng = np.random.RandomState(7)
idx = rng.permutation(len(rows))
for f in range(5):
    te = idx[f::5]
    tr = [i for i in idx if i not in set(te)]
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xn[tr], y[tr])
    oof[te] = clf.predict_proba(Xn[te])[:, 1]
print(f"META 5-fold OOF ECE={expected_calibration_error(list(oof), list(y)):.4f}",
      flush=True)

# Full-fit weights for inspection (NOT for reporting).
clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xn, y)
print("weights:", {k: round(float(v), 3) for k, v in
                   zip(FEATS + [f"is_{t}" for t in TIERS], clf.coef_[0])},
      flush=True)
json.dump({"mu": mu.tolist(), "sd": sd.tolist(),
           "w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
           "feats": FEATS + [f"is_{t}" for t in TIERS]},
          open("models/training_state/meta_cal.json", "w"))
print("META-CAL DONE (artifact saved, NOT shipped — decision below)", flush=True)
