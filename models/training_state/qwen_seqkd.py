#!/usr/bin/env python3
"""Qwen SeqKD labeling: 120 frontier_qa questions -> 10 task_router labels.
Writes D:/gavel/models/training_state/qwen_seqkd.jsonl incrementally.
~8s/label on this box => ~16 min. Run AFTER stopping hybrid (RAM)."""
import json
import random
import time

LABELS = ["ARC-Easy", "CommonsenseQA", "MATH/algebra", "MetaMathQA",
          "NuminaMath-CoT", "OrcaMath", "QASC", "SciQ", "file_ops", "multi_research"]
OUT = "D:/gavel/models/training_state/qwen_seqkd.jsonl"

rng = random.Random(7)
with open("D:/virtual-brain/frontier_seeds/frontier_qa.json") as f:
    frontier = json.load(f)
rng.shuffle(frontier)
sample = frontier[:120]
print(f"sampled {len(sample)}", flush=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("D:/gavel/models/qwen3-4b", trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "D:/gavel/models/qwen3-4b", dtype=torch.float16, device_map="auto",
    low_cpu_mem_usage=True, trust_remote_code=False)
model.eval()
print("Qwen loaded", flush=True)

PREF = ("Classify the question into exactly one of these types: "
        + ", ".join(LABELS) + ". Reply with only the type.\n\nQuestion: ")
n_ok = 0
with open(OUT, "w", encoding="utf-8") as out:
    for k, x in enumerate(sample):
        q = x["question"][:600]
        enc = tok(PREF + q + "\nType:", return_tensors="pt")
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        enc = {kk: vv.to(dev) for kk, vv in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=10, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        cont = tok.decode(gen[0], skip_special_tokens=True)
        tail = cont[len(tok.decode(enc["input_ids"][0], skip_special_tokens=True)):].strip()
        low = tail.lower()
        pick = next((l for l in LABELS if l.lower() in low), None)
        if pick:
            out.write(json.dumps({"input": x["question"], "label": pick}) + "\n")
            out.flush()
            n_ok += 1
        if (k + 1) % 10 == 0:
            print(f"{k+1}/120 usable={n_ok}", flush=True)
print(f"DONE usable={n_ok}/120 -> {OUT}", flush=True)
