#!/usr/bin/env python3
"""Train 5 Jev-style signal questions (TF-IDF, true/false) on the 800 phishing train.
Labels: bench heuristics for free_hosting/domain_mismatch (exact published code),
transparent keyword rules for lure/urgency/generic_sender (documented, heuristic).
Same splits as phishing_repro (verified by construction: same seeds).
Artifacts: models/training_state/sig_<name>.json ; questions: sig_<name>.
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Sohail/AppData/Local/Temp/opencode/jev-phishing-bench")
from bench.heuristics import features as heur  # exact published semantics

WORK = Path("C:/Users/Sohail/AppData/Local/Temp/opencode/gavel-phish")
TS = Path("D:/gavel/models/training_state")
SEED, SPLIT_SEED, TRAINVAL_SEED = 20260916, 20260917, 0x9E3779B9

LURE_RE = re.compile(
    r"sign.?in|log.?in|verif|confirm.*account|account.*(locked|suspend|hold)|"
    r"open.*document|download|claim.*(prize|refund|reward)|invoice.*(pay|due|overdue)|"
    r"password.*reset|reset.*password|update.*(payment|billing|card)|claim now|"
    r"gift.?card|pay.*fee|submit.*detail", re.I)
URGENCY_RE = re.compile(
    r"\burgent|immediately|asap|deadline|expir\w*\s*(today|soon|within)|within\s*24|"
    r"last chance|act now|threat|suspend|locked|lose access|final notice|"
    r"do not (wait|delay|hesitate|ignore)|today only|not later|at once|right away",
    re.I)
WEBMAIL = {"gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "aol.com",
           "proton.me", "protonmail.com", "icloud.com", "live.com", "msn.com",
           "ymail.com", "gmx.com", "mail.com", "zoho.com", "yandex.com"}


def link_host(u):
    from urllib.parse import urlparse
    return urlparse((u or "").strip()).netloc.lower().split(":")[0]


def sig_labels(email):
    h = heur(email)
    text = " ".join([email.get("subject", ""), email.get("body", ""),
                     email.get("link_display_text", "")])
    sender_dom = (email.get("from", "").split("@")[-1].lower()
                  if "@" in email.get("from", "") else "")
    return {
        "sig_free_hosting": "true" if h["hosting_or_shortener"] else "false",
        "sig_domain_mismatch": "true" if h["etld1_mismatch"] else "false",
        "sig_lure": "true" if LURE_RE.search(text) else "false",
        "sig_urgency": "true" if URGENCY_RE.search(text) else "false",
        "sig_generic_sender": "true" if sender_dom in WEBMAIL else "false",
    }


def serialize(email):
    e = email
    return (f"from: {e['from']} sender: {e['sender']} subject: {e['subject']} "
            f"link_text: {e['link_display_text']} link_url: {e['link_url']} body: {e['body']}")


recs = [json.loads(l) for l in open(WORK / "emails.jsonl", encoding="utf-8")]
by_id = {r["id"]: r for r in recs}
ids = [r["id"] for r in recs]
y = {r["id"]: r["y"] for r in recs}
rng = np.random.default_rng(SPLIT_SEED)
A, B = set(), set()
for lab in (0, 1):
    g = sorted(i for i in ids if y[i] == lab)
    perm = rng.permutation(len(g))
    A.update(g[k] for k in perm[:len(g) // 2])
    B.update(g[k] for k in perm[len(g) // 2:])
assert len(A) == 1000 and len(B) == 1000
rr = random.Random(TRAINVAL_SEED)
train_ids, val_ids = [], []
for lab in (0, 1):
    g = sorted(i for i in A if y[i] == lab)
    rr.shuffle(g)
    train_ids += g[:400]
    val_ids += g[400:]
print(f"split ok: train={len(train_ids)} val={len(val_ids)}", flush=True)

import urllib.request
BASE = "http://127.0.0.1:7575"


def api(p, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NAMES = ["sig_free_hosting", "sig_domain_mismatch", "sig_lure", "sig_urgency",
         "sig_generic_sender"]
for name in NAMES:
    ytr = [sig_labels(by_id[i]["email"])[name] for i in train_ids]
    Xtr_text = [serialize(by_id[i]["email"]) for i in train_ids]
    yva = [sig_labels(by_id[i]["email"])[name] for i in val_ids]
    Xva_text = [serialize(by_id[i]["email"]) for i in val_ids]
    from collections import Counter
    print(f"[{name}] train_pos_rate={sum(v == 'true' for v in ytr)}/{len(ytr)}", flush=True)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(Xtr_text)
    clf = LogisticRegression(max_iter=2000).fit(X, ytr)
    va_acc = clf.score(vec.transform(Xva_text), yva)
    print(f"[{name}] sklearn train={clf.score(X, ytr):.4f} val={va_acc:.4f} vocab={len(vec.vocabulary_)}",
          flush=True)
    terms, idf = vec.get_feature_names_out(), vec.idf_
    w = clf.coef_[0]
    art = {"model_id": name + "-v1", "classes": ["false", "true"], "temperature": 1.0,
           "intercept": [float(-clf.intercept_[0]), float(clf.intercept_[0])],
           "vocab": {t: [float(idf[j]), float(-w[j]), float(w[j])]
                     for j, t in enumerate(terms)}}
    mp = TS / f"{name}.json"
    json.dump(art, open(mp, "w"))
    api("/define", {"name": name, "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"},
                    "engine": {"tfidf": {"model": str(mp)}}})
    print(f"[{name}] defined", flush=True)
print("DONE", flush=True)
