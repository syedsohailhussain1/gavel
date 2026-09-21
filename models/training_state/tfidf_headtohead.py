#!/usr/bin/env python3
"""TF-IDF head-to-head for task_router (new decide-tfidf engine).

Trains sklearn TfidfVectorizer(1-2gram, sublinear_tf) + LogisticRegression on
the SAME 484 train (seed 20260920), exports the artifact schema decide-tfidf
expects, loads via engine.tfidf, calibrates, evals same 124 val.
Compares: logistic 0.798 / MLP-ONNX 0.8065 / TF-IDF ?.
"""
import json
import random

LABELS = ["ARC-Easy", "CommonsenseQA", "MATH/algebra", "MetaMathQA",
          "NuminaMath-CoT", "OrcaMath", "QASC", "SciQ", "file_ops", "multi_research"]

rng = random.Random(20260920)


def split(items, f=0.8):
    by = {}
    for i, l in items:
        by.setdefault(l, []).append((i, l))
    tr, va = [], []
    for lab, exs in by.items():
        rng.shuffle(exs)
        n = max(1, int(len(exs) * f)) if len(exs) > 1 else 1
        tr += exs[:n]
        va += exs[n:] if len(exs) > 1 else []
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import QA_SEEDS
seeds = json.load(open(QA_SEEDS))
split([(x["question"], x["domain"]) for x in seeds])  # advance rng identically
from collections import Counter
keep = {t for t, c in Counter(x["task_type"] for x in seeds).items() if c >= 30}
pool = {}
for x in seeds:
    if x["task_type"] in keep:
        pool.setdefault(x["task_type"], []).append((x["question"], x["task_type"]))
items = []
for t, exs in pool.items():
    rng.shuffle(exs)
    items += exs[:80]
train, val = split(items)
print(f"train={len(train)} val={len(val)}", flush=True)
assert len(val) == 124

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform([t for t, _ in train])
ytr = [l for _, l in train]
clf = LogisticRegression(max_iter=2000)
clf.fit(Xtr, ytr)
Xva = vec.transform([t for t, _ in val])
print(f"sklearn train={clf.score(Xtr, ytr):.4f} val={clf.score(Xva, [l for _, l in val]):.4f}",
      flush=True)
print(f"vocab={len(vec.vocabulary_)}", flush=True)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import TS as _TS
from m2 import calibrate_logits
_dh = clf.decision_function(Xva)
_cls = list(clf.classes_)
_sets = [[float(_dh[i, _cls.index(c)]) for c in LABELS] for i in range(len(val))]
_targets = [LABELS.index(l) for _, l in val]
T, ece_b, ece_a = calibrate_logits(_sets, _targets)
print(f"T={T:.4f} ece={ece_b:.4f}->{ece_a:.4f}", flush=True)

# Export artifact: term -> [idf, coef_c0..c9] in LABELS order
terms = vec.get_feature_names_out()
idf = vec.idf_
coefs = {c: clf.coef_[list(clf.classes_).index(c)] for c in LABELS}
vocab = {}
for j, t in enumerate(terms):
    vocab[t] = [float(idf[j])] + [float(coefs[c][j]) for c in LABELS]
art = {"model_id": "task-router-tfidf-v1", "classes": LABELS, "temperature": T,
       "eval": {"ece_before": ece_b, "ece_after": ece_a},
       "intercept": [float(b) for b in
                     [clf.intercept_[list(clf.classes_).index(c)] for c in LABELS]],
       "vocab": vocab}
path = str(_TS) + "/task_tfidf.json"
json.dump(art, open(path, "w"))
import os
print(f"artifact {os.path.getsize(path)/1e6:.2f} MB", flush=True)

import urllib.request
from gavel_paths import GAVEL_URL as BASE


def api(p, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print("define:", api("/define", {
    "name": "task_router_tfidf", "inputSchema": {"type": "object"},
    "outputSchema": {"type": "object"},
    "engine": {"tfidf": {"model": path}}}), flush=True)
try:
    print("calibrate:", api("/calibrate", {
        "question": "task_router_tfidf",
        "validation": [{"input": i, "label": l} for i, l in val]}), flush=True)
except Exception as e:
    print("calibrate:", str(e)[:150], flush=True)
print("metrics:", json.dumps(api("/metrics?question=task_router_tfidf")), flush=True)

correct, skipped = 0, 0
for k, (i, l) in enumerate(val):
    try:
        r = api("/ask", {"question": "task_router_tfidf", "input": i}, timeout=15)
    except Exception:
        skipped += 1
        continue
    if (r.get("output") or {}).get("label", "") == l:
        correct += 1
    if (k + 1) % 25 == 0:
        print(f"{k+1}/{len(val)}...", flush=True)
done = len(val) - skipped
print(f"TF-IDF val_acc={correct/max(done,1):.4f}({correct}/{done}) skipped={skipped}",
      flush=True)
print("baselines: logistic=0.798 mlp_onnx=0.8065", flush=True)
print("DONE", flush=True)
