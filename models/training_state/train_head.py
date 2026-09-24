#!/usr/bin/env python3
"""Train the pairwise scorer head on CACHED trunk hiddens (Stage 1, fast).

Tiny MLP (hidden->512->1, BCE) trains on CPU in minutes — the trunk is never
reloaded. Group split by item id (no same-item leakage). Temperature fit via
m2.py. Saves head .pt + report. No GPU. No API.

Usage:
  python train_head.py [hiddens_pt] [out_pt] [epochs] [lr]
Defaults: pair_hiddens.pt pair_head.pt 40 3e-4
"""
import json
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from m2 import calibrate_logits

HID = sys.argv[1] if len(sys.argv) > 1 else "pair_hiddens.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pair_head.pt"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-4
WIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 512

t0 = time.perf_counter()
ck = torch.load(HID, map_location="cpu", weights_only=False)
X = ck["hiddens"].float()
y = torch.tensor(ck["targets"], dtype=torch.float32)
order = ck["order"]
H = ck["hidden"]
print(f"n={len(X)} hidden={H} trunk={ck['model']}", flush=True)

by_item = defaultdict(list)
for k, iid in enumerate(order):
    by_item[str(iid)].append(k)
items = sorted(by_item)
rng = torch.Generator().manual_seed(20260924)
perm = torch.randperm(len(items), generator=rng).tolist()
cut = int(len(items) * 0.8)
tr_idx = [j for t in perm[:cut] for j in by_item[items[t]]]
va_idx = [j for t in perm[cut:] for j in by_item[items[t]]]
Xtr, ytr = X[tr_idx], y[tr_idx]
Xva, yva = X[va_idx], y[va_idx]
print(f"train={len(Xtr)} val={len(Xva)} items={len(items)}", flush=True)


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


head = Head(H, WIDE)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
bs = 256
best, best_ep, best_state = 1e9, 0, None
for ep in range(1, EPOCHS + 1):
    head.train()
    perm_b = torch.randperm(len(Xtr)).tolist()
    tot = 0.0
    for i in range(0, len(Xtr), bs):
        idx = perm_b[i:i + bs]
        opt.zero_grad()
        loss = lossf(head(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        tot += float(loss) * len(idx)
    head.eval()
    with torch.no_grad():
        vl = float(lossf(head(Xva), yva))
    if vl < best:
        best, best_ep = vl, ep
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
    if ep % 5 == 0 or ep == EPOCHS:
        print(f"ep{ep} train={tot/len(Xtr):.4f} val={vl:.4f} best={best:.4f}@{best_ep}",
              flush=True)
head.load_state_dict(best_state)

with torch.no_grad():
    va_logits = head(Xva).tolist()
sets = [[-v, v] for v in va_logits]
tg = [int(t) for t in yva.tolist()]
T, ece_b, ece_a = calibrate_logits(sets, tg)
tr_acc = ((head(Xtr).sign() + 1) / 2 == ytr).float().mean().item()
va_acc = ((head(Xva).sign() + 1) / 2 == yva).float().mean().item()
el = round((time.perf_counter() - t0) / 60, 1)
torch.save({"head": best_state, "hidden": H, "wide": WIDE, "trunk": ck["model"],
            "temperature": T}, OUT)
rep = {"out": OUT, "minutes": el, "train_acc": round(tr_acc, 4),
       "val_acc": round(va_acc, 4), "T": round(T, 4),
       "ece": [round(ece_b, 4), round(ece_a, 4)], "best_ep": best_ep}
json.dump(rep, open(OUT.rsplit(".pt", 1)[0] + "_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1), flush=True)
