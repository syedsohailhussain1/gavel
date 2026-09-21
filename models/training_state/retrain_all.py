#!/usr/bin/env python3
"""One-command rebuild of all Gavel questions (Step 1: persistence).

Gavel server keeps models in memory only — restart wipes them.
This script rebuilds every question deterministically from local sources.

Usage:
  python retrain_all.py [--skip-frontier]   # --skip-frontier skips the 98MB load

Questions rebuilt:
  domain_router  (qa_seeds, 4 domains)            seed 20260920
  task_router    (qa_seeds, 10 task types,<=80)   seed 20260920
  dolly_category (dolly catalog, 3 cats)          seed 20260920
  teacher_style  (frontier_qa sample 120/source)  seed 99
  code_task      (code seeds, 2 labels)           seed 99
  phishing       (local phishing_train/val.jsonl in this dir)
  route_ticket   (30-example demo, inline)

Needs: gavel-server up on :7575. Stdlib only.
"""
import json
import random
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7575"
HERE = Path(__file__).resolve().parent
rng = random.Random(20260920)


def api(path, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def split_examples(items, train_frac=0.8):
    by_label = {}
    for inp, lab in items:
        by_label.setdefault(lab, []).append((inp, lab))
    train, val = [], []
    for lab, exs in by_label.items():
        rng.shuffle(exs)
        n_train = max(1, int(len(exs) * train_frac)) if len(exs) > 1 else 1
        train += exs[:n_train]
        val += exs[n_train:] if len(exs) > 1 else []
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def train_q(name, items, calibrate=True):
    train, val = split_examples(items)
    api("/define", {"name": name, "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"}})
    api("/train", {"question": name, "examples": [
        {"input": i, "label": l} for i, l in train]})
    cal = "skip"
    if calibrate and val and all(any(l2 == l for _, l2 in train) for _, l in val):
        try:
            api("/calibrate", {"question": name, "validation": [
                {"input": i, "label": l} for i, l in val]})
            cal = f"ok({len(val)})"
        except Exception as e:
            cal = f"fail:{e}"
    m = api(f"/metrics?question={name}")
    print(f"[{name}] train={len(train)} val={len(val)} cal={cal} "
          f"T={m.get('temperature')} ece={m.get('ece_before')}->{m.get('ece_after')}",
          flush=True)
    return {"train": len(train), "val": len(val), "T": m.get("temperature")}


def main():
    report = {}
    with open("D:/virtual-brain/distillation_seeds/qa_seeds.json") as f:
        seeds = json.load(f)

    report["domain_router"] = train_q(
        "domain_router", [(x["question"], x["domain"]) for x in seeds])

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
    report["task_router"] = train_q("task_router", items)

    with open("D:/research on LLM breakthroughs/data/dolly_knowledge_catalog.json") as f:
        dolly = json.load(f)
    report["dolly_category"] = train_q(
        "dolly_category", [(x["prompt"], x["category"]) for x in dolly])

    if "--skip-frontier" not in sys.argv:
        print("loading frontier_qa (98MB)...", flush=True)
        with open("D:/virtual-brain/frontier_seeds/frontier_qa.json") as f:
            frontier = json.load(f)
        rng2 = random.Random(99)
        by_src = {}
        for x in frontier:
            by_src.setdefault(x["source"], []).append((x["question"], x["source"]))
        items = []
        for s, exs in by_src.items():
            rng2.shuffle(exs)
            items += exs[:120]
        # NOTE: split with main rng would diverge from original; use local split
        by_label, train, val = {}, [], []
        for i, l in items:
            by_label.setdefault(l, []).append((i, l))
        for lab, exs in by_label.items():
            rng2.shuffle(exs)
            n = max(1, int(len(exs) * 0.8)) if len(exs) > 1 else 1
            train += exs[:n]
            val += exs[n:] if len(exs) > 1 else []
        api("/define", {"name": "teacher_style", "inputSchema": {"type": "object"},
                        "outputSchema": {"type": "object"}})
        api("/train", {"question": "teacher_style", "examples": [
            {"input": i, "label": l} for i, l in train]})
        try:
            api("/calibrate", {"question": "teacher_style", "validation": [
                {"input": i, "label": l} for i, l in val]})
            cal = f"ok({len(val)})"
        except Exception as e:
            cal = f"fail:{e}"
        m = api("/metrics?question=teacher_style")
        print(f"[teacher_style] train={len(train)} val={len(val)} cal={cal} "
              f"T={m.get('temperature')} ece={m.get('ece_before')}->{m.get('ece_after')}",
              flush=True)
        report["teacher_style"] = {"train": len(train), "val": len(val)}

    with open("D:/virtual-brain/distillation_seeds/code_seeds.json") as f:
        code = json.load(f)
    rng3 = random.Random(99)
    exs = [(x["code"], x["task_type"]) for x in code]
    rng3.shuffle(exs)
    ntr = int(len(exs) * 0.8)
    api("/define", {"name": "code_task", "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"}})
    api("/train", {"question": "code_task", "examples": [
        {"input": i, "label": l} for i, l in exs[:ntr]]})
    print(f"[code_task] train={ntr} (uncalibrated, tiny)", flush=True)
    report["code_task"] = {"train": ntr}

    def loadjl(p):
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    tr = loadjl(HERE / "phishing_train.jsonl")
    va = loadjl(HERE / "phishing_val.jsonl")
    api("/define", {"name": "phishing", "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"}})
    api("/train", {"question": "phishing", "examples": [
        {"input": x["input"], "label": x["label"]} for x in tr]})
    api("/calibrate", {"question": "phishing", "validation": [
        {"input": x["input"], "label": x["label"]} for x in va]})
    m = api("/metrics?question=phishing")
    print(f"[phishing] train={len(tr)} val={len(va)} "
          f"T={m.get('temperature')} ece={m.get('ece_before')}->{m.get('ece_after')}",
          flush=True)
    report["phishing"] = {"train": len(tr), "val": len(va)}

    demo = ([("charged twice on my card", "billing")] * 8
            + [("refund my invoice", "billing")] * 7
            + [("app crashes on upload", "technical")] * 8
            + [("500 error from the api", "technical")] * 7)
    rng.shuffle(demo)
    ntr = 24
    api("/define", {"name": "route_ticket", "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"}})
    api("/train", {"question": "route_ticket", "examples": [
        {"input": i, "label": l} for i, l in demo[:ntr]]})
    api("/calibrate", {"question": "route_ticket", "validation": [
        {"input": i, "label": l} for i, l in demo[ntr:]]})
    print(f"[route_ticket] train={ntr} val={len(demo)-ntr} (demo)", flush=True)
    report["route_ticket"] = {"train": ntr}

    # Production TF-IDF (stable question name; versioned artifacts on disk).
    # Regenerates contrastive coverage EVERY rebuild so the failure mode
    # can never silently drop out of training.
    sys.path.insert(0, str(HERE))
    from contrast_augment import augment as contrast_gen
    hand_aug = json.load(open(HERE / "negation_aug_train.json"))
    gen = contrast_gen(tr)
    json.dump(gen, open(HERE / "contrastive_gen.json", "w"))
    full = tr + hand_aug + gen
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform([x["input"] for x in full])
    clf = LogisticRegression(max_iter=2000).fit(X, [x["label"] for x in full])
    terms, idf, w = vec.get_feature_names_out(), vec.idf_, clf.coef_[0]
    art = {"model_id": "phishing-tfidf-prod", "classes": ["legitimate", "phishing"],
           "temperature": 1.0,
           "intercept": [float(-clf.intercept_[0]), float(clf.intercept_[0])],
           "vocab": {t: [float(idf[j]), float(-w[j]), float(w[j])]
                     for j, t in enumerate(terms)}}
    # rotate previous prod artifact aside (rollback), then write new one
    import shutil
    prod = HERE / "phishing_tfidf_prod.json"
    if prod.exists():
        i = 1
        while (HERE / f"phishing_tfidf_prod.bak{i}.json").exists():
            i += 1
        shutil.copy(prod, HERE / f"phishing_tfidf_prod.bak{i}.json")
    json.dump(art, open(prod, "w"))
    api("/define", {"name": "phishing_tfidf", "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"},
                    "engine": {"tfidf": {"model": str(prod)}}})
    from collections import Counter as _C
    print(f"[phishing_tfidf] train={len(full)} "
          f"{dict(_C(x['label'] for x in full))} "
          f"(base={len(tr)} hand={len(hand_aug)} contrast={len(gen)})", flush=True)
    report["phishing_tfidf"] = {"train": len(full), "artifact": str(prod)}
    report["production_phishing_tfidf"] = "phishing_tfidf"

    (HERE / "manifest.json").write_text(json.dumps(report, indent=1))
    print("questions:", api("/questions"), flush=True)
    print("manifest ->", HERE / "manifest.json", flush=True)
    print("REBUILD DONE", flush=True)


if __name__ == "__main__":
    main()
