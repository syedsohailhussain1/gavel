#!/usr/bin/env python3
"""Item-level eval of the cached-head scorer (Stage 1).

Groups cached option-hiddens by item, softmaxes head logits with fitted T,
argmaxes. Reports accuracy per tier + overall + ECE. CPU only, seconds.

Usage:
  python eval_head.py [hiddens_pt] [head_pt] [pairs_json]
Defaults: pair_hiddens_short.pt pair_head.pt decision_pairs.jsonl
"""
import json
import math
import sys
from collections import defaultdict

import torch
import torch.nn as nn

HID = sys.argv[1] if len(sys.argv) > 1 else "pair_hiddens_short.pt"
HEAD = sys.argv[2] if len(sys.argv) > 2 else "pair_head.pt"
PAIRS = sys.argv[3] if len(sys.argv) > 3 else "decision_pairs.jsonl"


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
order = ck["order"]
T = hp["temperature"]
head = Head(hp["hidden"], hp.get("wide", 512))
head.load_state_dict(hp["head"])
head.eval()

# Map cached rows back to pair records (cache preserves dataset order).
by_item = defaultdict(list)
for k, r in enumerate(rows[:len(X)]):
    by_item[(r.get("tier"), r.get("item"))].append(k)

with torch.no_grad():
    logits = head(X).tolist()

tier_ok = defaultdict(int)
tier_n = defaultdict(int)
conf, corr = [], []
skipped = 0
seen = set()
for (tier, iid), idxs in by_item.items():
    # Reconstruct option order from the pair records.
    outs = []
    for k in idxs:
        r = rows[k]
        outs.append((r, logits[k]))
    # Expected label = the positive record's option text.
    exp = None
    for r, _ in outs:
        if r["target"] == 1:
            exp = r
    if exp is None:
        skipped += 1
        continue
    # Softmax over options with T; argmax.
    m = max(v / T for _, v in outs)
    ex = [math.exp(v / T - m) for _, v in outs]
    s = sum(ex)
    p = [e / s for e in ex]
    j = max(range(len(p)), key=lambda k: p[k])
    # Recover predicted option identity: compare against expected text.
    pred_text = outs[j][0]["text"]
    ok = pred_text == exp["text"]
    t = exp.get("tier", tier)
    tier_ok[t] += ok
    tier_n[t] += 1
    conf.append(p[j])
    corr.append(ok)

tot_ok = sum(tier_ok.values())
tot_n = sum(tier_n.values())
print(json.dumps({
    "head": HEAD, "T": round(T, 4),
    "overall": f"{tot_ok}/{tot_n}={tot_ok/max(tot_n,1):.4f}",
    "by_tier": {t: f"{tier_ok[t]}/{tier_n[t]}={tier_ok[t]/max(tier_n[t],1):.4f}"
                for t in sorted(tier_n)},
    "skipped_no_positive": skipped,
}, indent=1), flush=True)
