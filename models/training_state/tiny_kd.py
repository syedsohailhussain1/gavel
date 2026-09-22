#!/usr/bin/env python3
"""Soft-target distillation: tiny student learns teacher LOGITS, not labels.

KL(student/T || teacher/T) + CE(hard) with cosine schedule. The standard
recipe for squeezing big-model behavior into small models without growing them.

Usage:
  python tiny_kd.py [student_id] [teacher_dir] [data_base] [out_dir] [epochs] [batch] [temp] [alpha]
Defaults: google/bert_uncased_L-2_H-128_A-2 tiny_snake_distil tiny_snake.jsonl
          tiny_snake_gtinykd 25 128 2.5 0.7
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SID = sys.argv[1] if len(sys.argv) > 1 else "google/bert_uncased_L-2_H-128_A-2"
TDIR = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake_distil"
DATA = sys.argv[3] if len(sys.argv) > 3 else "tiny_snake.jsonl"
OUT = sys.argv[4] if len(sys.argv) > 4 else "tiny_snake_gtinykd"
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 25
BATCH = int(sys.argv[6]) if len(sys.argv) > 6 else 128
TEMP = float(sys.argv[7]) if len(sys.argv) > 7 else 2.5
ALPHA = float(sys.argv[8]) if len(sys.argv) > 8 else 0.7
LABELS = ["up", "down", "left", "right"]
LI = {l: i for i, l in enumerate(LABELS)}

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

base = DATA.rsplit(".jsonl", 1)[0]
train = json.load(open(base + "_train.jsonl", encoding="utf-8"))
val = json.load(open(base + "_val.jsonl", encoding="utf-8"))
print(f"train={len(train)} val={len(val)} T={TEMP} alpha={ALPHA}", flush=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stok = AutoTokenizer.from_pretrained(SID)
tmodel = AutoModelForSequenceClassification.from_pretrained(TDIR).to(dev).eval()
smodel = AutoModelForSequenceClassification.from_pretrained(SID, num_labels=4).to(dev)


def prep(rows, stoker):
    enc = stoker([r["text"] for r in rows], truncation=True, max_length=96,
                 padding=True, return_tensors="pt")
    y = torch.tensor([LI[r["label"]] for r in rows])
    return enc["input_ids"], enc["attention_mask"], y


tr_ids, tr_msk, tr_y = prep(train, stok)
va_ids, va_msk, va_y = prep(val, stok)

# Cache teacher logits once (the expensive part, done one time).
tlogits = []
ttok = AutoTokenizer.from_pretrained(TDIR)
te_ids, te_msk, _ = prep(train, ttok)
with torch.no_grad():
    for i in range(0, len(train), 512):
        sl = slice(i, i + 512)
        tlogits.append(tmodel(te_ids[sl].to(dev), te_msk[sl].to(dev)).logits.cpu())
tlogits = torch.cat(tlogits)
print("teacher logits cached:", tuple(tlogits.shape), flush=True)
del tmodel
torch.cuda.empty_cache()

trl = TensorDataset(tr_ids, tr_msk, tr_y, tlogits)
vll = TensorDataset(va_ids, va_msk, va_y)
tl = DataLoader(trl, batch_size=BATCH, shuffle=True)
vl = DataLoader(vll, batch_size=BATCH * 2)
ce = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(smodel.parameters(), lr=3e-5)
sched = get_cosine_schedule_with_warmup(opt, int(0.1 * EPOCHS * len(tl)),
                                        EPOCHS * len(tl))


def acc_of():
    smodel.eval()
    ok = tot = 0
    with torch.no_grad():
        for ids, msk, y in vl:
            p = smodel(ids.to(dev), msk.to(dev)).logits.argmax(-1).cpu()
            ok += (p == y).sum().item()
            tot += len(y)
    return ok / tot


t0 = time.perf_counter()
best = 0
for ep in range(1, EPOCHS + 1):
    smodel.train()
    for ids, msk, y, tl_gt in tl:
        ids, msk, y, tl_gt = (t.to(dev) for t in (ids, msk, y, tl_gt))
        opt.zero_grad()
        sl = smodel(ids, msk).logits
        loss = (ALPHA * (TEMP ** 2) * F.kl_div(
                    F.log_softmax(sl / TEMP, -1),
                    F.softmax(tl_gt / TEMP, -1), reduction="batchmean")
                + (1 - ALPHA) * ce(sl, y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(smodel.parameters(), 1.0)
        opt.step()
        sched.step()
    va = acc_of()
    print(f"ep{ep} val_acc={va:.4f} [{(time.perf_counter()-t0)/60:.1f}m]", flush=True)
    if va > best:
        best = va
        smodel.save_pretrained(OUT)
        stok.save_pretrained(OUT)
print(f"BEST val_acc={best:.4f} -> {OUT}", flush=True)
