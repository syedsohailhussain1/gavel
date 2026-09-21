#!/usr/bin/env python3
"""Export phishing TF-IDF artifact (same 800 train) and load as phishing_tfidf."""
import json

LABELS = ["legitimate", "phishing"]


def loadjl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


tr = loadjl("D:/gavel/models/training_state/phishing_train.jsonl")
print(f"train={len(tr)}", flush=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform([x["input"] for x in tr])
ytr = [x["label"] for x in tr]
clf = LogisticRegression(max_iter=2000)
clf.fit(Xtr, ytr)
print(f"sklearn train_acc={clf.score(Xtr, ytr):.4f} vocab={len(vec.vocabulary_)}", flush=True)

terms = vec.get_feature_names_out()
idf = vec.idf_
# Binary sklearn: coef_[0]/intercept_[0] belong to classes_[1]; mirror for classes_[0].
pos = list(clf.classes_).index("phishing")
w_pos, b_pos = clf.coef_[0 if len(clf.classes_) == 2 else pos], clf.intercept_[0 if len(clf.classes_) == 2 else pos]
coefs = {"phishing": w_pos, "legitimate": -w_pos}
bias = {"phishing": float(b_pos), "legitimate": float(-b_pos)}
vocab = {t: [float(idf[j])] + [float(coefs[c][j]) for c in LABELS]
         for j, t in enumerate(terms)}
art = {"model_id": "phishing-tfidf-v1", "classes": LABELS, "temperature": 1.0,
       "intercept": [bias[c] for c in LABELS],
       "vocab": vocab}
path = "D:/gavel/models/training_state/phishing_tfidf.json"
json.dump(art, open(path, "w"))
import os
print(f"artifact {os.path.getsize(path)/1e6:.2f} MB", flush=True)

import urllib.request
BASE = "http://127.0.0.1:7575"


def api(p, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print("define:", api("/define", {
    "name": "phishing_tfidf", "inputSchema": {"type": "object"},
    "outputSchema": {"type": "object"},
    "engine": {"tfidf": {"model": path}}}), flush=True)
try:
    print("calibrate:", api("/calibrate", {
        "question": "phishing_tfidf",
        "validation": [{"input": x["input"], "label": x["label"]} for x in
                       loadjl("D:/gavel/models/training_state/phishing_val.jsonl")]}),
        flush=True)
except Exception as e:
    print("calibrate refuses:", str(e)[:120], flush=True)
print("metrics:", json.dumps(api("/metrics?question=phishing_tfidf")), flush=True)
print("DONE", flush=True)
