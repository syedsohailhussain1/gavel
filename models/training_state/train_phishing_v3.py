#!/usr/bin/env python3
"""Versioned retrain: phishing_tfidf_v3 on 800 + 48 hand-aug + 360 contrastive."""
import json
import urllib.request

BASE = "http://127.0.0.1:7575"
LABELS = ["legitimate", "phishing"]
TS = "D:/gavel/models/training_state/"
VER = "v3"


def api(p, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def loadjl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


base = loadjl(TS + "phishing_train.jsonl")
hand = json.load(open(TS + "negation_aug_train.json"))
gen = json.load(open(TS + "contrastive_gen.json"))
train = base + hand + gen
from collections import Counter
print(f"train={len(train)}", Counter(x["label"] for x in train), flush=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
X = vec.fit_transform([x["input"] for x in train])
clf = LogisticRegression(max_iter=2000).fit(X, [x["label"] for x in train])
print(f"train_acc={clf.score(X, [x['label'] for x in train]):.4f} vocab={len(vec.vocabulary_)}",
      flush=True)
terms, idf, w = vec.get_feature_names_out(), vec.idf_, clf.coef_[0]
art = {"model_id": f"phishing-tfidf-{VER}-contrastive", "classes": LABELS, "temperature": 1.0,
       "intercept": [float(-clf.intercept_[0]), float(clf.intercept_[0])],
       "vocab": {t: [float(idf[j]), float(-w[j]), float(w[j])] for j, t in enumerate(terms)}}
mpath = TS + f"phishing_tfidf_{VER}.json"
json.dump(art, open(mpath, "w"))
import os
print(f"artifact {os.path.getsize(mpath)/1e6:.2f} MB", flush=True)
print(api("/define", {"name": f"phishing_tfidf_{VER}", "inputSchema": {"type": "object"},
                      "outputSchema": {"type": "object"},
                      "engine": {"tfidf": {"model": mpath}}}), flush=True)
print("metrics:", json.dumps(api(f"/metrics?question=phishing_tfidf_{VER}")), flush=True)
print("DONE", flush=True)
