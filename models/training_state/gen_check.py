#!/usr/bin/env python3
"""Generation-vs-logits check on JevBench easy tier (standalone)."""
import json
import re
import sys

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
MAXT = 2048


def flat(s):
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def render(item, tok):
    q = item["question"]
    st = flat(item["state"])
    if q["type"] == "choice":
        labels = list(q["criteria"].keys())
        opts = "\n".join(f"{chr(65 + i)}. {k}: {q['criteria'][k]}"
                         for i, k in enumerate(labels))
        tail = (f"{q['instructions']}\nOptions:\n{opts}\n"
                f"Answer with only the letter:")
    elif q["type"] == "noul":
        labels = ["yes", "no"]
        tail = (f"{q['instructions']}\nAnswer with only YES or NO:")
    else:
        return None, None
    budget = MAXT - len(tok(tail)["input_ids"]) - 32
    ids = tok(st, truncation=True, max_length=max(budget, 32))["input_ids"]
    return "Case: " + tok.decode(ids[-max(budget, 32):]) + "\n\n" + tail, labels


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    trust_remote_code=False).eval()
dev = next(model.parameters()).device
print("[model] on", dev, flush=True)

items = [json.loads(l) for l in
         open(sys.argv[2] if len(sys.argv) > 2
              else r"D:\jevbench\datasets\public\easy.jsonl", encoding="utf-8")]
if len(sys.argv) > 3 and sys.argv[3] != "4bit":
    try:
        items = items[:int(sys.argv[3])]
    except ValueError:
        pass
ok = tot = 0
for it in items:
    head, labels = render(it, tok)
    if head is None:
        continue
    enc = tok(head, return_tensors="pt", truncation=False).to(dev)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=15, do_sample=False)
    tail = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip().lower()
    exp = str(it["expected"]).strip().lower()
    # The model answers with LETTERS ("the answer is e"), not label names:
    # map the last-mentioned option letter back to its label.
    lets = re.findall(r"(?<![a-z])[a-e](?![a-z])", tail)
    got = labels[ord(lets[-1]) - ord("a")] if lets and ord(lets[-1]) - ord("a") < len(labels) else None
    tot += 1
    ok += (got is not None and got.lower() == exp)
    print(f"got={got} exp={exp} :: {tail[:90]}", flush=True)
print(f"GEN EASY: {ok}/{tot}={ok/max(tot,1):.3f}", flush=True)
