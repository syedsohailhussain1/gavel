#!/usr/bin/env python3
"""Retrieve-then-read cache: full coverage at constant memory.

Long states are split into 400-token windows (stride 200). Each window is
scored by keyword overlap with question+option (CPU, instant); the top-3
windows are forwarded (with question+option attached) and mean-pooled into
ONE vector per pair. Context length becomes linear time, flat memory —
million-token docs work the same way, just more windows.

Usage:
  python ret_pool_cache.py [pairs_json] [out_pt] [model] [start] [end] [log]
Defaults: pairs_long.jsonl pool_hiddens.pt <local qwen3-4b> 0 -1 none
Shard with start/end across turns; merge like the trunk cache.
"""
import json
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "pairs_long.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pool_hiddens.pt"
MODEL = sys.argv[3] if len(sys.argv) > 3 else r"D:\gavel\models\qwen3-4b"
START = int(sys.argv[4]) if len(sys.argv) > 4 else 0
END = int(sys.argv[5]) if len(sys.argv) > 5 else -1
LOG = open(sys.argv[6], "a", encoding="utf-8") if len(sys.argv) > 6 else None
WIN, STRIDE, TOPK, MAXT = 256, 128, 4, 384


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def keywords(t):
    return {w.lower() for w in re.findall(r"[A-Za-z0-9$.,/\-]{5,}", t)}


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

rows = json.load(open(PAIRS, encoding="utf-8"))
if END < 0:
    END = len(rows)
rows = rows[START:END]
log(f"pairs={len(rows)} (shard {START}:{END})")


def split_state(text):
    """Split pair text into (state, question_option) at the Question marker;
    fall back to whole-text windows if the marker is absent."""
    if "\nQuestion:" in text:
        st, qo = text.split("\nQuestion:", 1)
        return st, "Question:" + qo
    return text, ""


feats, t0 = [], time.perf_counter()
with torch.no_grad():
    for k, r in enumerate(rows):
        st, qo = split_state(r["text"])
        keys = keywords(qo + " " + r["text"][:500])
        ids = tok(st, truncation=False)["input_ids"]
        wins = [ids[i:i + WIN] for i in range(0, max(len(ids) - WIN + 1, 1), STRIDE)]
        if not wins:
            wins = [ids[:WIN]]
        scored = []
        for w in wins:
            txt = tok.decode(w)
            tl = txt.lower()
            scored.append((sum(1 for key in keys if key in tl), txt))
        scored.sort(key=lambda t: -t[0])
        # ONE batched forward for all TOPK windows (GPU stays fed).
        texts = ["Excerpt: " + wtxt + "\n" + qo.strip()
                 for _, wtxt in scored[:TOPK]]
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAXT,
                  padding=True).to(dev)
        o = lm(input_ids=enc["input_ids"],
               attention_mask=enc.get("attention_mask"),
               output_hidden_states=True, use_cache=False, return_dict=True)
        h = o.hidden_states[-1].float().cpu()
        last = (enc.get("attention_mask").sum(1) - 1).clamp_min(0).tolist()
        vecs = [h[j][last[j]] for j in range(len(texts))]
        feats.append(torch.stack(vecs).mean(0))
        done = k + 1
        if done % 10 == 0 or done == len(rows):
            el = time.perf_counter() - t0
            log(f"[{done}/{len(rows)}] {done/max(el,1e-9):.1f} pairs/s")
torch.save({"hiddens": torch.stack(feats).half(), "hidden": H, "model": MODEL,
            "order": [r.get("item", k) for k, r in enumerate(rows)],
            "targets": [r["target"] for r in rows]}, OUT)
log(f"saved {OUT} shape={torch.stack(feats).shape}")
