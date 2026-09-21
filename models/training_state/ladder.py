#!/usr/bin/env python3
"""Zero-shot ladder (generic): cold-start a NEW question with zero labels.
  1. COLD-START: Qwen labels N unlabeled items (no examples given — zero-shot).
  2. SPOT-CHECK: prints K samples for human truth-anchor (teacher is fallible).
  3. TRAIN: TF-IDF candidate on train split of Qwen labels.
  4. GATE: held-out Qwen labels must hit >= target (fidelity, not ground truth).
  5. PROMOTE: defines production question on pass (human confirms spot-check).
Usage:
  python ladder.py <question> '<labels-csv>' <corpus.json> <text-field> [N] [target]
Example:
  python ladder.py topic_router 'math,code,science,other' frontier_qa.json question 120 0.80
Corpus: JSON list of objects. Writes ladder_<question>_labels.jsonl + artifact.
"""
import json
import random
import sys
import urllib.request

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import TS as _TS, GAVEL_URL as _BASE, QWEN as _QWEN

QUESTION, LABELS_CSV, CORPUS, FIELD = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 120
TARGET = float(sys.argv[6]) if len(sys.argv) > 6 else 0.80
LABELS = [l.strip() for l in LABELS_CSV.split(",")]
TS = str(_TS) + "/"
BASE = _BASE
QWEN = str(_QWEN)


def api(p, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print(f"[ladder] cold-starting '{QUESTION}' labels={LABELS} N={N} target={TARGET}", flush=True)
items = json.load(open(CORPUS, encoding="utf-8"))
rng = random.Random(20260921)
rng.shuffle(items)
sample = items[:N]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    QWEN, dtype=torch.float16, device_map="auto",
    low_cpu_mem_usage=True, trust_remote_code=False)
model.eval()
print("[ladder] Qwen loaded", flush=True)

PREF = ("Classify the text into exactly one of these types: " + ", ".join(LABELS)
        + ". Reply with only the type.\n\nText: ")
labeled = []
for k, x in enumerate(sample):
    t = str(x[FIELD])[:600]
    enc = tok(PREF + t + "\nType:", return_tensors="pt")
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    enc = {kk: vv.to(dev) for kk, vv in enc.items()}
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=10, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    tail = tok.decode(gen[0], skip_special_tokens=True)
    tail = tail[len(tok.decode(enc["input_ids"][0], skip_special_tokens=True)):].strip()
    pick = next((l for l in LABELS if l.lower() in tail.lower()), None)
    if pick:
        labeled.append({"input": str(x[FIELD]), "label": pick})
    if (k + 1) % 10 == 0:
        print(f"[ladder] {k+1}/{N} usable={len(labeled)}", flush=True)

lp = TS + f"ladder_{QUESTION}_labels.jsonl"
with open(lp, "w", encoding="utf-8") as f:
    for r in labeled:
        f.write(json.dumps(r) + "\n")
print(f"[ladder] collected {len(labeled)}/{N} -> {lp}", flush=True)

print("--- HUMAN SPOT-CHECK (truth anchor: reply ok/fix) ---", flush=True)
for r in labeled[:12]:
    print(f"  [{r['label']}] {r['input'][:150]}", flush=True)

rng.shuffle(labeled)
ntr = int(len(labeled) * 0.75)
train, held = labeled[:ntr], labeled[ntr:]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from m2 import calibrate_logits
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
X = vec.fit_transform([x["input"] for x in train])
clf = LogisticRegression(max_iter=2000).fit(X, [x["label"] for x in train])
Xh = vec.transform([x["input"] for x in held])
yh = [x["label"] for x in held]
acc = clf.score(Xh, yh)
print(f"[ladder] student train={len(train)} held-out(Qwen)={len(held)} fidelity={acc:.3f}",
      flush=True)
# Per-class raw scores in LABELS order (order-safe for binary and multinomial).
# Binary sklearn: decision_function is the signed distance for classes_[1].
_dh = clf.decision_function(Xh)
_cls = list(clf.classes_)
if _dh.ndim > 1:
    _col = {c: _dh[:, _cls.index(c)] for c in LABELS}
else:
    _pos = _cls[1]
    _col = {c: (_dh if c == _pos else -_dh) for c in LABELS}
_sets = [[float(_col[c][i]) for c in LABELS] for i in range(len(yh))]
targets = [LABELS.index(l) for l in yh]
T, ece_b, ece_a = calibrate_logits(_sets, targets)
print(f"[ladder] T={T:.4f} ece={ece_b:.4f}->{ece_a:.4f}", flush=True)
terms, idf, w = vec.get_feature_names_out(), vec.idf_, clf.coef_
ncls = len(clf.classes_)
cls = list(clf.classes_)
if ncls == 2:
    _pos = cls[1]
    co = {c: (w[0] if c == _pos else -w[0]) for c in LABELS}
    _b = float(clf.intercept_[0])
    bi = {c: (_b if c == _pos else -_b) for c in LABELS}
    order = LABELS
else:
    co = {c: w[cls.index(c)] for c in LABELS}
    bi = {c: float(clf.intercept_[cls.index(c)]) for c in LABELS}
    order = LABELS
art = {"model_id": f"{QUESTION}-ladder-v1", "classes": order, "temperature": T,
       "eval": {"ece_before": ece_b, "ece_after": ece_a},
       "intercept": [bi[c] for c in order],
       "vocab": {t: [float(idf[j])] + [float(co[c][j]) for c in order]
                 for j, t in enumerate(terms)}}
mp = TS + f"ladder_{QUESTION}.json"
json.dump(art, open(mp, "w"))
api("/define", {"name": QUESTION + "_zero", "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"},
                "engine": {"tfidf": {"model": mp}}})
print(f"[ladder] candidate '{QUESTION}_zero' defined", flush=True)
if acc >= TARGET:
    print(f"[ladder] GATE PASS ({acc:.3f} >= {TARGET}) — confirm spot-check, then promote",
          flush=True)
else:
    print(f"[ladder] GATE HOLD ({acc:.3f} < {TARGET}) — more labels or better prompt needed",
          flush=True)
