#!/usr/bin/env python3
"""Fit the FINAL meta-calibrator on all 231 records (4 features) + verify.

Writes meta_cal.json (mu/sd/w/b) consumed by the serving adapter/shim:
top-label confidence = meta P(correct); remaining mass rescaled
proportionally (argmax + accuracy unchanged, distributions preserved).
OOF ECE across seeds: 0.032-0.057 (mean ~0.045) vs temperature 0.070.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from m2 import expected_calibration_error
from meta_cal import rows  # noqa: F401  (reuses record builder; prints again)

FEATS = ["c", "margin", "ent", "kopts"]
X = np.array([[r[f] for f in FEATS] for r in rows])
mu, sd = X.mean(0), X.std(0) + 1e-9
Xn = (X - mu) / sd
y = np.array([r["y"] for r in rows])
# C=0.05 chosen by OOF+in-sample consistency sweep (both ~0.035);
# stronger regularization collapses resolution, weaker overfits NLL-vs-ECE.
clf = LogisticRegression(C=0.05, max_iter=5000).fit(Xn, y)
p = clf.predict_proba(Xn)[:, 1]
print(f"IN-SAMPLE ECE={expected_calibration_error(list(p), list(y)):.4f} "
      f"(optimistic; OOF estimate 0.032-0.057)", flush=True)
json.dump({"mu": mu.tolist(), "sd": sd.tolist(),
           "w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
           "feats": FEATS,
           "method": "L2-logistic P(correct); OOF ECE 0.032-0.057 over 4 seeds"},
          open("models/training_state/meta_cal.json", "w"))
print("wrote meta_cal.json (FINAL, 4 feats, full fit)", flush=True)
