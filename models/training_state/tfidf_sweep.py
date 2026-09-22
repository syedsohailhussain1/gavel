#!/usr/bin/env python3
"""Sweep ngram/C for the legacy TF-IDF student (training-side only)."""
import json
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = json.load(open("models/training_state/tiny_snake4_100k_train.jsonl", encoding="utf-8"))
val = json.load(open("models/training_state/tiny_snake4_100k_val.jsonl", encoding="utf-8"))
yv = [r["label"] for r in val]
Xv_txt = [r["text"] for r in val]
Xtr_txt = [r["text"] for r in train]
ytr = [r["label"] for r in train]

for ng in ((1, 2), (1, 3)):
    for C in (4.0, 8.0):
        t0 = time.perf_counter()
        vec = TfidfVectorizer(lowercase=True, ngram_range=ng, sublinear_tf=True)
        X = vec.fit_transform(Xtr_txt)
        clf = LogisticRegression(C=C, max_iter=2000).fit(X, ytr)
        el = time.perf_counter() - t0
        print(f"ngram={ng} C={C} vocab={len(vec.vocabulary_)} "
              f"train_s={el:.1f} val={clf.score(vec.transform(Xv_txt), yv):.4f}",
              flush=True)
