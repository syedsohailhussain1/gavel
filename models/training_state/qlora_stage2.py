#!/usr/bin/env python3
"""LoRA Stage 2, chunked for a 4GB box (the slow path, quarantined until now).

QLoRA rank-8 on q/v projections only, 4-bit frozen trunk, gradient
checkpointing, batch-1/accum-8, 384 ctx, paged 8-bit Adam. Pairwise BCE on
(state, option) texts with the v1 MLP head attached (warm-started).
Saves adapter + head + optimizer + position every --save-every steps;
resume continues exactly. Designed for ~25-min wall-clock chunks.

Usage:
  python qlora_stage2.py [pairs] [run_dir] [steps] [resume]
Defaults: combined_pairs.jsonl stage2a 200 fresh
Memory target: <4GB VRAM (bnb 4-bit base ~2.3GB + adapters/grads/states).
"""
import json
import os
import sys
import time

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "combined_pairs.jsonl"
RUN = sys.argv[2] if len(sys.argv) > 2 else "stage2a"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
RESUME = (len(sys.argv) > 4 and sys.argv[4] == "resume")
TRUNK = r"D:\gavel\models\qwen3-4b"
CTX, ACCUM, LR = 384, 8, 2e-4
os.makedirs(RUN, exist_ok=True)
LOG = open(os.path.join(RUN, "train.log"), "a", encoding="utf-8")


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


rows = json.load(open(PAIRS, encoding="utf-8"))
log(f"pairs={len(rows)} steps={STEPS} resume={RESUME} -> {RUN}")

tok = AutoTokenizer.from_pretrained(TRUNK, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    TRUNK, quantization_config=bnb, device_map="auto",
    trust_remote_code=False)
base.gradient_checkpointing_enable()
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                 lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, cfg)
dev = next(model.parameters()).device
H = model.config.hidden_size
head = Head(H).float()
try:
    hp = torch.load("models/training_state/combined_head.pt",
                    map_location="cpu", weights_only=False)
    head.load_state_dict(hp["head"])
    log("head warm-started from v1")
except Exception as e:
    log(f"head fresh init ({str(e)[:80]})")
head.to(dev).train()
lossf = nn.BCEWithLogitsLoss()
trn = [p for p in list(model.parameters()) + list(head.parameters())
       if p.requires_grad]
print(f"[trainable] {sum(p.numel() for p in trn)/1e6:.2f}M params", flush=True)
opt = torch.optim.AdamW(trn, lr=LR)
try:
    from bitsandbytes.optim import PagedAdamW8bit
    opt = PagedAdamW8bit(trn, lr=LR)
    log("optimizer: paged 8-bit AdamW")
except Exception as e:
    log(f"optimizer: AdamW ({str(e)[:60]})")

pos = 0
if RESUME:
    try:
        set_peft_model_state_dict(model, torch.load(
            os.path.join(RUN, "adapter.bin"), map_location="cpu"))
        head.load_state_dict(torch.load(os.path.join(RUN, "head.bin"),
                                        map_location="cpu"))
        opt.load_state_dict(torch.load(os.path.join(RUN, "opt.bin"),
                                       map_location="cpu"))
        pos = json.load(open(os.path.join(RUN, "state.json")))["pos"]
        log(f"resumed at pair {pos}")
    except Exception as e:
        log(f"resume failed, fresh start ({str(e)[:100]})")
        pos = 0

import random
rng = random.Random(20260926)
order = list(range(len(rows)))
rng.shuffle(order)

model.train()
accum = 0
t0 = time.perf_counter()
step = 0
k = pos
while step < STEPS:
    r = rows[order[k % len(rows)]]
    k += 1
    enc = tok(r["text"], return_tensors="pt", truncation=True,
              max_length=CTX).to(dev)
    y = torch.tensor([float(r["target"])]).to(dev)
    o = model(input_ids=enc["input_ids"],
              attention_mask=enc.get("attention_mask"),
              output_hidden_states=True, use_cache=False, return_dict=True)
    hs = o.hidden_states[-1][0].float()
    last = int(enc.get("attention_mask").sum().item()) - 1
    logit = head(hs[max(last, 0)].unsqueeze(0).to(dev))
    (lossf(logit, y) / ACCUM).backward()
    accum += 1
    if accum >= ACCUM:
        torch.nn.utils.clip_grad_norm_(trn, 1.0)
        opt.step()
        opt.zero_grad()
        accum = 0
        step += 1
        if step % 10 == 0 or step == STEPS:
            el = time.perf_counter() - t0
            log(f"step {step}/{STEPS} loss~{float(lossf(logit.detach(), y)):.4f} "
                f"[{el/60:.1f}m, {el/max(step,1):.1f}s/step]")
        if step % 50 == 0 or step == STEPS:
            model.save_pretrained(os.path.join(RUN, "adapter_tmp"))
            import shutil
            for f in os.listdir(os.path.join(RUN, "adapter_tmp")):
                shutil.move(os.path.join(RUN, "adapter_tmp", f),
                            os.path.join(RUN, f))
            torch.save(head.state_dict(), os.path.join(RUN, "head.bin"))
            torch.save(opt.state_dict(), os.path.join(RUN, "opt.bin"))
            json.dump({"pos": k % len(rows)},
                      open(os.path.join(RUN, "state.json"), "w"))
            log(f"ckpt @ step {step}")
log("CHUNK DONE — resume with: qlora_stage2.py ... resume")
