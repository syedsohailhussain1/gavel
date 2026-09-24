#!/usr/bin/env python3
"""Cache frozen-trunk hiddens for the scorer head (Stage 1, fast path).

ONE forward per pair text through Qwen3-4B 4-bit (frozen, no grad).
Last-token last-layer hidden states -> fp16 .pt on disk, dataset order.
Heads train later on CPU in minutes without ever reloading the trunk.

Usage:
  python cache_hiddens.py [pairs_json] [out_pt] [model] [batch] [max_tok]
Defaults: decision_pairs.jsonl pair_hiddens.pt <local qwen3-4b> 4 768
Long states left-truncated (question+option survive); logged honestly.
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "decision_pairs.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pair_hiddens.pt"
MODEL = sys.argv[3] if len(sys.argv) > 3 else r"D:\gavel\models\qwen3-4b"
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 2
MAXT = int(sys.argv[5]) if len(sys.argv) > 5 else 512
LOG = open(sys.argv[6], "a", encoding="utf-8") if len(sys.argv) > 6 else None
SLEEP = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
START = int(sys.argv[8]) if len(sys.argv) > 8 else 0
END = int(sys.argv[9]) if len(sys.argv) > 9 else -1


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()

rows = json.load(open(PAIRS, encoding="utf-8"))
if END < 0:
    END = len(rows)
rows = rows[START:END]
log(f"pairs={len(rows)} (shard {START}:{END}) model={MODEL} batch={BATCH} maxtok={MAXT}")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
lm = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    trust_remote_code=False).eval()
for p in lm.parameters():
    p.requires_grad = False
dev = next(lm.parameters()).device
H = lm.config.hidden_size
log(f"hidden={H} device={dev}")

feats = []
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        enc = tok([r["text"] for r in chunk], return_tensors="pt",
                  truncation=False, padding=True)
        ids, msk = enc["input_ids"], enc["attention_mask"]
        if ids.shape[1] > MAXT:
            # Left-truncate: drop old context, never the question+option.
            ids, msk = ids[:, -MAXT:], msk[:, -MAXT:]
        input_ids = ids.to(dev)
        attn = msk.to(dev)
        o = lm(input_ids=input_ids, attention_mask=attn,
               output_hidden_states=True, use_cache=False, return_dict=True)
        hs = o.hidden_states[-1].float().cpu()
        last = (attn.sum(1) - 1).clamp_min(0).tolist()
        for j in range(len(chunk)):
            feats.append(hs[j][last[j]])
        done = min(i + BATCH, len(rows))
        if done % max(BATCH * 5, 20) == 0 or done == len(rows):
            el = time.perf_counter() - t0
            log(f"[{done}/{len(rows)}] {done/max(el,1e-9):.1f} fwd/s")
        if SLEEP > 0:
            time.sleep(SLEEP)  # thermal pacing: sustained 100% heat-soaks
            # the GTX 1650 into throttle territory (~84C); breathing room
            # keeps the rate constant instead of collapsing.
torch.save({"hiddens": torch.stack(feats).half(), "hidden": H, "model": MODEL,
            "order": [r.get("item", k) for k, r in enumerate(rows)],
            "targets": [r["target"] for r in rows]}, OUT)
log(f"saved {OUT} shape={torch.stack(feats).shape}")
