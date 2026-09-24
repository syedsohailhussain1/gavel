#!/usr/bin/env python3
"""Fine-tune a tiny encoder (default ModernBERT-base, ~150M) on Snake states.

Manual torch loop, GPU if available. Class-weighted CE for the up/down skew.
Saves best-val checkpoint + tokenizer to out dir.

Usage:
  python tiny_train.py [model_id] [data_base] [out_dir] [epochs] [batch]
Defaults: answerdotai/ModernBERT-base tiny_snake.jsonl tiny_snake_modernbert 3 64
(data_base names <base>_train.jsonl / <base>_val.jsonl from tiny_data.py)
"""
import json
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

MODEL = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/ModernBERT-base"
DATA = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake.jsonl"
OUT = sys.argv[3] if len(sys.argv) > 3 else "tiny_snake_modernbert"
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 64
LABELS = sys.argv[6].split(",") if len(sys.argv) > 6 else ["up", "down", "left", "right"]
LI = {l: i for i, l in enumerate(LABELS)}

from transformers import AutoModelForSequenceClassification, AutoTokenizer

base = DATA.rsplit(".jsonl", 1)[0]
train = json.load(open(base + "_train.jsonl", encoding="utf-8"))
val = json.load(open(base + "_val.jsonl", encoding="utf-8"))
print(f"train={len(train)} val={len(val)}", flush=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)


def prep(rows):
    enc = tok([r["text"] for r in rows], truncation=True, max_length=96,
              padding=True, return_tensors="pt")
    y = torch.tensor([LI[r["label"]] for r in rows])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


trl, val = prep(train), prep(val)
counts = torch.bincount(trl.tensors[2], minlength=4).float()
w = counts.sum() / (4 * counts)
print("class weights:", [round(float(x), 3) for x in w], flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4)
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
lossf = torch.nn.CrossEntropyLoss(weight=w.to(dev))
tl = DataLoader(trl, batch_size=BATCH, shuffle=True)
vl = DataLoader(val, batch_size=BATCH * 2)


def acc_of(loader):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for ids, msk, y in loader:
            p = model(ids.to(dev), msk.to(dev)).logits.argmax(-1).cpu()
            ok += (p == y).sum().item()
            tot += len(y)
    return ok / tot


t0 = time.perf_counter()
best, best_ep = 0, 0
for ep in range(1, EPOCHS + 1):
    model.train()
    tot_loss, n = 0.0, 0
    for ids, msk, y in tl:
        ids, msk, y = ids.to(dev), msk.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(ids, msk, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += out.loss.item() * len(y)
        n += len(y)
    va = acc_of(vl)
    el = (time.perf_counter() - t0) / 60
    print(f"ep{ep} loss={tot_loss/n:.4f} val_acc={va:.4f} [{el:.1f}m]", flush=True)
    if va > best:
        best, best_ep = va, ep
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
print(f"BEST ep{best_ep} val_acc={best:.4f} -> {OUT} "
      f"[{round((time.perf_counter()-t0)/60,1)}m total]", flush=True)
