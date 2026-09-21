#!/usr/bin/env python3
"""Retrain phishing engines on 800 + 32 negation-aug; tune energy gate on aug-val."""
import json
import urllib.request

BASE = "http://127.0.0.1:7575"
LABELS = ["legitimate", "phishing"]
TS = "D:/gavel/models/training_state/"


def api(p, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def loadjl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


tr800 = loadjl(TS + "phishing_train.jsonl")
va200 = loadjl(TS + "phishing_val.jsonl")
aug_tr = json.load(open(TS + "negation_aug_train.json"))
aug_va = json.load(open(TS + "negation_aug_val.json"))
print(f"base={len(tr800)}/{len(va200)} aug={len(aug_tr)}/{len(aug_va)}", flush=True)
full_tr = tr800 + aug_tr

# --- TF-IDF ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
X = vec.fit_transform([x["input"] for x in full_tr])
clf = LogisticRegression(max_iter=2000).fit(X, [x["label"] for x in full_tr])
print(f"tfidf train_acc={clf.score(X, [x['label'] for x in full_tr]):.4f}", flush=True)
terms, idf = vec.get_feature_names_out(), vec.idf_
w = clf.coef_[0]
vocab = {t: [float(idf[j]), float(-w[j]), float(w[j])] for j, t in enumerate(terms)}
art = {"model_id": "phishing-tfidf-v2-neg", "classes": LABELS, "temperature": 1.0,
       "intercept": [float(-clf.intercept_[0]), float(clf.intercept_[0])], "vocab": vocab}
mpath = TS + "phishing_tfidf_v2.json"
json.dump(art, open(mpath, "w"))
import os
print(f"artifact {os.path.getsize(mpath)/1e6:.2f} MB", flush=True)


def ask(q, text, timeout=30):
    body = json.dumps({"question": q, "input": text}).encode()
    req = urllib.request.Request(
        BASE + "/ask", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# define WITHOUT gate first, to observe raw energies
print(api("/define", {"name": "phishing_tfidf", "inputSchema": {"type": "object"},
                      "outputSchema": {"type": "object"},
                      "engine": {"tfidf": {"model": mpath}}}), flush=True)
for name, items in (("aug_val", aug_va), ("orig_val_sample", va200[:12])):
    print(f"-- energies {name} --", flush=True)
    for x in items:
        r = ask("phishing_tfidf", x["input"])
        out = (r.get("output") or {}).get("label", "?")
        mark = "OK " if out == x["label"] else "MISS"
        print(f"[{mark}] want={x['label']} got={out} "
              f"conf={r.get('confidence', 0):.3f} energy={r.get('energy')} "
              f"::{x['input'][:70]}", flush=True)

# --- logistic retrain on 832 ---
api("/define", {"name": "phishing", "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"}})
api("/train", {"question": "phishing", "examples": [
    {"input": x["input"], "label": x["label"]} for x in full_tr]})
api("/calibrate", {"question": "phishing", "validation": [
    {"input": x["input"], "label": x["label"]} for x in va200]})
print("logistic metrics:", json.dumps(api("/metrics?question=phishing")), flush=True)
print("DONE-RETRAIN", flush=True)
