#!/usr/bin/env python3
"""Legacy TF-IDF trainer: sklearn 1-2gram + logreg, exported to Gavel native.

Mirrors the proven phishing_tfidf recipe (retrain_all.py): the underscore
format (up_free_fC_sO) survives as single unigrams AND forms binding bigrams,
which the server's hashed-BOW logistic engine cannot do.

Usage:
  python legacy_tfidf.py [train_json] [val_json] [question] [artifact]
Defaults: tiny_snake4_train.jsonl tiny_snake4_val.jsonl snake_tfidf
          snake_tfidf_prod.json (in training_state dir)
Needs server up. Requires sklearn+numpy. Writes versioned artifact + defines.
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

TS = Path(__file__).resolve().parent
TRAIN = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake4_train.jsonl"
VAL = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake4_val.jsonl"
QUESTION = sys.argv[3] if len(sys.argv) > 3 else "snake_tfidf"
ART = TS / (sys.argv[4] if len(sys.argv) > 4 else "snake_tfidf_prod.json")
BASE = "http://127.0.0.1:7575"

sys.path.insert(0, str(TS))
from m2 import calibrate_logits


def api(path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


train = json.load(open(TS / TRAIN, encoding="utf-8"))
val = json.load(open(TS / VAL, encoding="utf-8"))
labels = sorted({r["label"] for r in train})
print(f"train={len(train)} val={len(val)} labels={labels}", flush=True)

import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

t0 = time.perf_counter()
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
X = vec.fit_transform([r["text"] for r in train])
clf = LogisticRegression(C=8.0, max_iter=2000).fit(X, [r["label"] for r in train])
train_s = time.perf_counter() - t0
print(f"train_seconds={train_s:.3f} vocab={len(vec.vocabulary_)}", flush=True)

Xv = vec.transform([r["text"] for r in val])
yv = [r["label"] for r in val]
print(f"val_accuracy={clf.score(Xv, yv):.4f} (sklearn, pre-export)", flush=True)

import numpy as np
D = clf.decision_function(Xv)
cls = list(clf.classes_)
W = clf.coef_
if D.ndim > 1:
    # Multinomial: take each label's column in LABELS order.
    col = {c: np.asarray(D[:, cls.index(c)], dtype=float) for c in labels}
    sets = [[float(col[c][i]) for c in labels] for i in range(len(yv))]
    co = {c: W[cls.index(c)] for c in labels}
    bi = {c: float(clf.intercept_[cls.index(c)]) for c in labels}
else:
    # Binary sklearn: signed distance for classes_[1]; mirror it so the
    # served logits are [neg, pos] in LABELS order (same recipe as ladder).
    pos = cls[1]
    d = np.asarray(D.tolist(), dtype=float)
    co = {c: (W[0] if c == pos else -W[0]) for c in labels}
    _b = float(clf.intercept_[0])
    bi = {c: (_b if c == pos else -_b) for c in labels}
    sets = [[(float(v) if c == pos else -float(v)) for c in labels] for v in d]
targets = [labels.index(l) for l in yv]
T, ece_b, ece_a = calibrate_logits(sets, targets)
print(f"T={T:.4f} ece={ece_b:.4f}->{ece_a:.4f}", flush=True)

terms, idf = vec.get_feature_names_out(), vec.idf_
art = {"model_id": QUESTION + "-prod", "classes": labels, "temperature": T,
       "eval": {"ece_before": ece_b, "ece_after": ece_a},
       "intercept": [bi[c] for c in labels],
       "vocab": {t: [float(idf[j])] + [float(co[c][j]) for c in labels]
                 for j, t in enumerate(terms)}}
if ART.exists():
    i = 1
    while (TS / f"{ART.stem}.bak{i}.json").exists():
        i += 1
    shutil.copy(ART, TS / f"{ART.stem}.bak{i}.json")
json.dump(art, open(ART, "w"))
api("/define", {"name": QUESTION, "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"},
                "engine": {"tfidf": {"model": str(ART)}}})

ok = 0
for r in val:
    a = api("/ask", {"question": QUESTION, "input": r["text"]}, timeout=30)
    if (a.get("output") or {}).get("label") == r["label"]:
        ok += 1
print(f"served_accuracy={ok}/{len(val)}={ok/max(len(val),1):.4f}", flush=True)
print("LEGACY TFIDF DONE", flush=True)
