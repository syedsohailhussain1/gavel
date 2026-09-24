#!/usr/bin/env python3
"""Pair dataset for the frozen-trunk scorer head (Stage 1, fast path).

Each JevBench item becomes (state, option) pairs labeled 1/0 — the
openJev-Verdict NLI framing, learnable by a tiny head on frozen hiddens.
Negatives downsampled per item. Optional MNLI supplement for robustness.

Usage:
  python pair_builder.py [out] [--nli N] [--neg K]
Defaults: decision_pairs.jsonl, no NLI, 3 negatives/item kept.
Writes one JSON array. No GPU. No API.
"""
import json
import random
import sys

JB = r"D:\jevbench\datasets\public"
OUT = sys.argv[1] if len(sys.argv) > 1 else "decision_pairs.jsonl"
args = sys.argv[2:]
NLI = 0
NEG = 3
if "--nli" in args:
    NLI = int(args[args.index("--nli") + 1])
if "--neg" in args:
    NEG = int(args[args.index("--neg") + 1])

rng = random.Random(20260924)
pairs = []
stats = {"items": 0, "pos": 0, "neg": 0, "skipped_score": 0}


def add(text, target, iid, tier):
    pairs.append({"text": text, "target": target, "item": iid, "tier": tier})
    stats["pos" if target else "neg"] += 1


for tier in ("easy", "original", "hard"):
    for line in open(f"{JB}/{tier}.jsonl", encoding="utf-8"):
        it = json.loads(line)
        q = it["question"]
        st = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"])
        exp = str(it["expected"]).strip()
        stats["items"] += 1
        if q["type"] == "choice":
            labels = list(q["criteria"].keys())
            negs = [l for l in labels if l.lower() != exp.lower()]
            rng.shuffle(negs)
            pos = [l for l in labels if l.lower() == exp.lower()]
            for l in pos:
                add(f"State: {st}\nQuestion: {q['instructions']}\n"
                    f"Option: {l}: {q['criteria'][l]}", 1, it["id"], tier)
            for l in negs[:NEG]:
                add(f"State: {st}\nQuestion: {q['instructions']}\n"
                    f"Option: {l}: {q['criteria'][l]}", 0, it["id"], tier)
        elif q["type"] == "noul":
            labels = list(q.get("labels", ["yes", "no"]))
            for l in labels:
                crit = (q.get("criteria", {}) or {}).get(l, "")
                add(f"State: {st}\nQuestion: {q['instructions']}\n"
                    f"Option: {l}" + (f": {crit}" if crit else ""),
                    1 if l.lower() == exp.lower() else 0, it["id"], tier)
        else:
            # Score items: one pair per level (same BCE head; expected level
            # = probability-weighted sum at serve time).
            levels = list(q.get("criteria", {}).keys()) or \
                list(q.get("labels", []))
            for l in levels:
                crit = (q.get("criteria", {}) or {}).get(l, "")
                add(f"State: {st}\nQuestion: {q['instructions']}\n"
                    f"Option: {l}" + (f": {crit}" if crit else ""),
                    1 if str(l).lower() == exp.lower() else 0,
                    it["id"], tier)
            stats["score_items"] = stats.get("score_items", 0) + 1

if NLI > 0:
    from datasets import load_dataset
    ds = load_dataset("SetFit/mnli", split="train").shuffle(seed=7)
    n = 0
    for r in ds:
        if n >= NLI:
            break
        if r["label"] not in (0, 1, 2):
            continue
        # entailment -> (hypothesis true); contradiction -> false; skip neutral
        if r["label"] == 1:
            continue
        truth = "true" if r["label"] == 0 else "false"
        add(f"State: {r['text']}.\nQuestion: Does the hypothesis follow?\n"
            f"Option: {truth}", 1, f"mnli-{n}", "nli")
        other = "false" if truth == "true" else "true"
        add(f"State: {r['text']}.\nQuestion: Does the hypothesis follow?\n"
            f"Option: {other}", 0, f"mnli-{n}", "nli")
        n += 1
    stats["nli"] = n

json.dump(pairs, open(OUT, "w"))
print(json.dumps({"pairs": len(pairs), **stats, "out": OUT}, indent=1),
      flush=True)
