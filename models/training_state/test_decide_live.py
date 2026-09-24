#!/usr/bin/env python3
"""Live test: v1 head + frozen 4B trunk answer real decisions end to end.

Loads Qwen3-4B 4-bit (frozen) + v1 combined head, runs 3 sample public
items live (no cache): option logits -> temperature softmax -> decision.
Measures per-decision wall time. CPU fallback if no CUDA.

Usage:
  python test_decide_live.py [head_pt] [trunk]
Defaults: combined_head.pt <local qwen3-4b>
"""
import json
import math
import sys
import time

import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

HEAD = sys.argv[1] if len(sys.argv) > 1 else "combined_head.pt"
TRUNK = sys.argv[2] if len(sys.argv) > 2 else r"D:\gavel\models\qwen3-4b"
N_EASY = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_ORIG = int(sys.argv[4]) if len(sys.argv) > 4 else 20
N_HARD = int(sys.argv[5]) if len(sys.argv) > 5 else 10


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


tok = AutoTokenizer.from_pretrained(TRUNK, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True)
    lm = AutoModelForCausalLM.from_pretrained(
        TRUNK, quantization_config=bnb, device_map="auto",
        trust_remote_code=False).eval()
else:
    lm = AutoModelForCausalLM.from_pretrained(
        TRUNK, dtype=torch.float32, trust_remote_code=False).eval()
for p in lm.parameters():
    p.requires_grad = False
print(f"[trunk] on {dev}", flush=True)

hp = torch.load(HEAD, map_location="cpu", weights_only=False)
head = Head(hp["hidden"], hp.get("wide", 512))
head.load_state_dict(hp["head"])
head.eval()
T = hp["temperature"]
print(f"[head] {HEAD} T={T}", flush=True)


def decide(state, instructions, options):
    """options: [(label, criteria_or_None)] -> (best_label, probs)."""
    texts = []
    for lab, crit in options:
        t = f"State: {state}\nQuestion: {instructions}\nOption: {lab}"
        if crit:
            t += f": {crit}"
        texts.append(t)
    enc = tok(texts, return_tensors="pt", truncation=True,
              max_length=512, padding=True)
    gin = {k: v.to(dev) for k, v in enc.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        o = lm(**gin, output_hidden_states=True, use_cache=False,
               return_dict=True)
        hs = o.hidden_states[-1]
        last = (gin["attention_mask"].sum(1) - 1).clamp_min(0)
        vec = hs[torch.arange(len(texts)), last].float().cpu()
        lg = head(vec).tolist()
    ms = (time.perf_counter() - t0) * 1000
    m = max(v / T for v in lg)
    ex = [math.exp(v / T - m) for v in lg]
    s = sum(ex)
    probs = [e / s for e in ex]
    j = max(range(len(probs)), key=lambda k: probs[k])
    return options[j][0], probs, ms


import random

TIERS = [("easy", r"D:\jevbench\datasets\public\easy.jsonl", N_EASY),
         ("original", r"D:\jevbench\datasets\public\original.jsonl", N_ORIG),
         ("hard", r"D:\jevbench\datasets\public\hard.jsonl", N_HARD)]

ok = tot = 0
lat = []
for tier, path, n in TIERS:
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng = random.Random(20260925)
    rng.shuffle(items)
    used = 0
    for it in items:
        if used >= n:
            break
        q = it["question"]
        if q["type"] not in ("choice", "noul"):
            continue
        st = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"])
        if q["type"] == "choice":
            opts = [(k, q["criteria"][k]) for k in q["criteria"]]
        else:
            labs = list(q.get("labels", ["yes", "no"]))
            opts = [(l, (q.get("criteria", {}) or {}).get(l)) for l in labs]
        best, probs, ms = decide(st, q["instructions"], opts)
        hit = str(best).lower() == str(it["expected"]).lower()
        ok += hit
        tot += 1
        used += 1
        lat.append(ms)
        print(f"[{tier} {used}/{n}] decision={best} expected={it['expected']} "
              f"p={max(probs):.3f} {ms:.0f}ms "
              f"{'HIT' if hit else 'miss'}", flush=True)
lat.sort()
print(f"LIVE-50+: {ok}/{tot}={ok/max(tot,1):.3f} "
      f"p50={lat[len(lat)//2]:.0f}ms p99={lat[int(len(lat)*0.99)]:.0f}ms",
      flush=True)
