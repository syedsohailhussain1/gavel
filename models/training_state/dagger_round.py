#!/usr/bin/env python3
"""DAgger round: run the current head over public items, harvest errors.

For each public item: score all options with head+T, and if argmax is wrong,
emit the item's pairs (labeled) into a DAgger file. Retrain mixes these in —
the head learns exactly where it fails. CPU only (uses cached hiddens where
possible, trunk forwards only for fresh pairs).

Usage:
  python dagger_round.py [hiddens_pt] [head_pt] [pairs_json] [out_json]
Defaults: combined_hiddens.pt combined_head.pt combined_pairs.jsonl
          dagger_pairs.jsonl
"""
import json
import math
import sys
from collections import defaultdict

import torch
import torch.nn as nn

HID = sys.argv[1] if len(sys.argv) > 1 else "combined_hiddens.pt"
HEAD = sys.argv[2] if len(sys.argv) > 2 else "combined_head.pt"
PAIRS = sys.argv[3] if len(sys.argv) > 3 else "combined_pairs.jsonl"
OUT = sys.argv[4] if len(sys.argv) > 4 else "dagger_pairs.jsonl"


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

by_item = defaultdict(list)
for k, r in enumerate(rows[:len(X)]):
    by_item[(r.get("tier"), r.get("item"))].append(k)
with torch.no_grad():
    logits = head(X).tolist()

harvested, items = [], 0
for key, idxs in by_item.items():
    outs = [(rows[k], logits[k]) for k in idxs]
    exp = next((r for r, _ in outs if r["target"] == 1), None)
    if exp is None:
        continue
    items += 1
    m = max(v / T for _, v in outs)
    ex = [math.exp(v / T - m) for _, v in outs]
    s = sum(ex)
    p = [e / s for e in ex]
    j = max(range(len(p)), key=lambda k: p[k])
    if outs[j][0]["text"] != exp["text"]:
        # Wrong: harvest all this item's pairs with true labels.
        harvested.extend(rows[k] for k in idxs)
json.dump(harvested, open(OUT, "w"))
print(json.dumps({"items": items, "wrong": len(harvested),
                  "pairs_harvested": len(harvested), "out": OUT}, indent=1),
      flush=True)
