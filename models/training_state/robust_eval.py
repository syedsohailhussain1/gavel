#!/usr/bin/env python3
"""Unseen robustness tests for the v1 head (gates the LoRA burn).

Test 1 — shuffled-order invariance: permute options within each item before
softmax. Our architecture scores options independently, so accuracy MUST be
identical; any delta is an implementation bug (like the A/B flip before).

Test 2 — adversarial distractors: inject K wrong options sampled from OTHER
items' options (same tier) into each item, then argmax over the expanded set.
Chance drops; surviving accuracy measures real discrimination. Distractor
hiddens must be cached first (see --cache-distractors flow below).

Usage:
  python robust_eval.py [hiddens] [head] [pairs] [distractors] [seed]
Defaults: combined_hiddens.pt combined_head.pt combined_pairs.jsonl none 7
Distractor file format: JSON array {text, target:0, item, tier, distractor:1}.
"""
import json
import math
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn

HID = sys.argv[1] if len(sys.argv) > 1 else "combined_hiddens.pt"
HEAD = sys.argv[2] if len(sys.argv) > 2 else "combined_head.pt"
PAIRS = sys.argv[3] if len(sys.argv) > 3 else "combined_pairs.jsonl"
DIST = sys.argv[4] if len(sys.argv) > 4 else "none"
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


ck = torch.load(HID, map_location="cpu", weights_only=False)
hp = torch.load(HEAD, map_location="cpu", weights_only=False)
rows = json.load(open(PAIRS, encoding="utf-8"))
X = ck["hiddens"].float()
T = hp["temperature"]
head = Head(hp["hidden"], hp.get("wide", 512))
head.load_state_dict(hp["head"])
head.eval()
with torch.no_grad():
    base_logits = head(X).tolist()

# Optional distractor block appended after base rows.
dlogits, drows = [], []
if DIST != "none":
    drows = json.load(open(DIST, encoding="utf-8"))
    dh = torch.load(DIST.rsplit(".jsonl", 1)[0] + "_hiddens.pt",
                    map_location="cpu", weights_only=False)["hiddens"].float()
    with torch.no_grad():
        dlogits = head(dh).tolist()

by_item = defaultdict(list)
for k, r in enumerate(rows[:len(base_logits)]):
    by_item[(r.get("tier"), r.get("item"))].append(("b", k))
if drows:
    assert len(drows) == len(dlogits)
    for k, r in enumerate(drows):
        by_item[(r.get("tier"), r.get("item"))].append(("d", k))

rng = random.Random(SEED)


def accuracy(order_fn):
    ok = tot = 0
    for key, idxs in by_item.items():
        outs = []
        for src, k in idxs:
            if src == "b":
                outs.append((rows[k], base_logits[k]))
            else:
                outs.append((drows[k], dlogits[k]))
        exp = next((r for r, _ in outs if r["target"] == 1), None)
        if exp is None:
            continue
        seq = order_fn(outs)
        m = max(v / T for _, v in seq)
        ex = [math.exp(v / T - m) for _, v in seq]
        s = sum(ex)
        p = [e / s for e in ex]
        j = max(range(len(p)), key=lambda k: p[k])
        ok += (seq[j][0]["text"] == exp["text"])
        tot += 1
    return ok, tot


o, t = accuracy(lambda outs: outs)
print(f"STANDARD: {o}/{t}={o/max(t,1):.4f}", flush=True)
o2, t2 = accuracy(lambda outs: sorted(outs, key=lambda z: rng.random()))
print(f"SHUFFLED: {o2}/{t2}={o2/max(t2,1):.4f} "
      f"(must match standard)", flush=True)
if drows:
    print(f"(distractors active: {len(drows)} extra options)", flush=True)
print("ROBUST-EVAL DONE", flush=True)
