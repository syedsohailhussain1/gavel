#!/usr/bin/env python3
"""Generalist baseline: 1B instruct model, option-letter readout + fitted T.

Jqv pattern: state+question+options in, P(letter) out of one forward pass,
temperature fitted on the easy tier, evaluated on all public tiers.
4-bit quantized so it flies on 4GB VRAM. Batched forwards.

Usage:
  python generalist_baseline.py [model] [max_tok] [batch]
Defaults: Qwen/Qwen3-1.7B 2048 8
Public sets: easy + original + hard (datasets/public/*.jsonl in D:/jevbench).
"""
import json
import math
import sys

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 8
JB = r"D:\jevbench\datasets\public"

SETS = {}
for name in ("easy", "original", "hard"):
    SETS[name] = [json.loads(l) for l in
                  open(f"{JB}/{name}.jsonl", encoding="utf-8")]
print({k: len(v) for k, v in SETS.items()}, flush=True)


def flat(s):
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def render(item, tok):
    """Prompt text + ordered labels. Returns (text, labels) or (None, None)
    for score items (levels need a different readout; counted separately).
    State is left-truncated (keep the TAIL: question + options must survive);
    chat template applied (instruct models need it)."""
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
    st_ids = tok(st, truncation=True, max_length=max(budget, 32))["input_ids"]
    # left-truncate the state: drop old context, never the question.
    # NOTE: no chat template — Qwen3 starts thinking after the assistant
    # header, which scrambles first-token letter logits. Raw prompt reads
    # cleaner for logit readout (measured: template collapses easy 48→27%).
    st_ids = st_ids[-max(budget, 32):]
    head = "Case: " + tok.decode(st_ids) + "\n\n" + tail
    return head, labels


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    trust_remote_code=False).eval()
dev = next(model.parameters()).device
print("[model] on", dev, flush=True)

# Letter tokens: " A".."H", " Yes"/" No" fallbacks via first-substring match.
LET = {}
for i in range(8):
    for cand in (f" {chr(65 + i)}", chr(65 + i)):
        ids = tok(cand, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            LET[chr(65 + i)] = ids[0]
            break
print(f"[letters] {len(LET)} single-token letters", flush=True)


def letter_ids(labels):
    out = []
    for i, l in enumerate(labels):
        if l.lower() in ("yes", "no"):
            for cand in (f" {l.capitalize()}", f" {l}", l.capitalize(), l):
                ids = tok(cand, add_special_tokens=False)["input_ids"]
                if ids:
                    out.append(ids[0])
                    break
        else:
            out.append(LET.get(chr(65 + i)))
    return out


def batch_logits(rows):
    """rows: [(text, labels)] -> list of logit-vectors over labels."""
    enc = tok([r[0] for r in rows], return_tensors="pt",
              truncation=False, padding=True).to(dev)
    with torch.no_grad():
        lg = model(**enc, use_cache=False).logits[:, -1, :].float().cpu()
    res = []
    for j, (_, labels) in enumerate(rows):
        li = letter_ids(labels)
        res.append([float(lg[j][i]) if i is not None else float("-inf")
                    for i in li])
    return res


def fit_temperature(pairs):
    """pairs: [(logits, target_idx)] -> T minimizing NLL (grid like m2)."""
    def nll(t):
        tot = 0.0
        for lg, y in pairs:
            m = max(v / t for v in lg)
            ex = [math.exp(v / t - m) for v in lg]
            tot += -math.log(ex[y] / sum(ex) + 1e-15)
        return tot / max(len(pairs), 1)
    best_t, best_n = 1.0, float("inf")
    for i in range(80):
        t = 0.05 * (200.0 ** (i / 79.0))
        n = nll(t)
        if n < best_n:
            best_n, best_t = n, t
    return best_t


def predict(pairs, T):
    ok = tot = 0
    ece_c, ece_n = [], []
    for lg, y in pairs:
        m = max(v / T for v in lg)
        ex = [math.exp(v / T - m) for v in lg]
        s = sum(ex)
        p = [e / s for e in ex]
        j = max(range(len(p)), key=lambda k: p[k])
        ok += (j == y)
        tot += 1
        ece_c.append(p[j])
        ece_n.append(j == y)
    return ok, tot


# Collect logits per tier.
data = {}
for name, items in SETS.items():
    rows = []
    for it in items:
        head, labels = render(it, tok)
        if head is None:
            continue
        exp = str(it["expected"]).strip()
        yi = next((k for k, l in enumerate(labels) if l.lower() == exp.lower()),
                  None)
        if yi is None and len(labels) == 2:
            # yes/no expected values sometimes capitalized differently
            yi = next((k for k, l in enumerate(labels)
                       if exp.lower().startswith(l.lower())), None)
        if yi is None:
            continue
        rows.append((head, labels, yi))
    print(f"[{name}] usable={len(rows)}/{len(items)}", flush=True)
    pairs = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        for (h, l, y), lg in zip(chunk, batch_logits([(h, l) for h, l, _ in chunk])):
            pairs.append((lg, y))
    data[name] = pairs
    if (len(pairs)):
        o, t = predict(pairs, 1.0)
        print(f"[{name}] T=1.0 acc={o}/{t}={o/t:.3f}", flush=True)

T = fit_temperature(data.get("easy", []) or data["original"])
print(f"[fit] T={T:.4f} on easy tier", flush=True)
for name, pairs in data.items():
    o, t = predict(pairs, T)
    print(f"[{name}] T-fit acc={o}/{t}={o/t:.4f}", flush=True)
