#!/usr/bin/env python3
"""Gavel local adapter for the JevBench harness (our code, their runner).

Drives the v1 frozen-trunk + MLP head: one batched forward per task over
(state, option) pairs, softmax with fitted temperature, native probability
distributions over the task's EXACT label strings. Long states left-
truncated to 512 total tokens (matches training distribution; laya does
the same at its 512 budget — documented, not hidden).

Follows the laya_local adapter contract: __init__ kwargs, load(), run(task)
-> DecisionResult, reserve_estimate(). No edits to the JevBench repo.
"""
from __future__ import annotations

import json
import math
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r"D:\jevbench")

from jevbench.adapters.base import DecisionResult  # noqa: E402


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def flat(s):
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


class GavelLocalAdapter:
    name = "gavel_local"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, threads=4,
                 revision=None, head=None, ctx=512):
        self.trunk = endpoint or r"D:\gavel\models\qwen3-4b"
        self.model = model or "gavel-decide-4b"
        self.head_path = head or (r"D:\gavel\models\training_state"
                                  r"\combined_head.pt")
        self.ctx = ctx
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.revision = revision
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.tok = AutoTokenizer.from_pretrained(
            self.trunk, trust_remote_code=False)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.trunk, quantization_config=bnb, device_map="auto",
                trust_remote_code=False).eval()
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.trunk, dtype=torch.float32,
                trust_remote_code=False).eval().to(dev)
        for p in self.lm.parameters():
            p.requires_grad = False
        self.dev = next(self.lm.parameters()).device
        hp = torch.load(self.head_path, map_location="cpu", weights_only=False)
        self.head = Head(hp["hidden"], hp.get("wide", 512))
        self.head.load_state_dict(hp["head"])
        self.head.to(self.dev).eval()
        self.temp = float(hp["temperature"])
        self._loaded = True

    def _pair_texts(self, task):
        st = flat(task.state)
        instr = task.question.get("instructions", "")
        qtype = task.question.get("type")
        crit = task.question.get("criteria") or {}
        if qtype == "choice":
            opts = [(k, crit.get(k) if isinstance(crit, dict) else None)
                    for k in task.labels]
        elif qtype == "noul":
            opts = [(l, crit.get(l) if isinstance(crit, dict) else None)
                    for l in task.labels]
        elif qtype == "score":
            if isinstance(crit, dict):
                opts = [(k, crit[k]) for k in task.labels]
            else:
                lv = list(crit) if crit else list(task.labels)
                opts = [(l, lv[i] if i < len(lv) else None)
                        for i, l in enumerate(task.labels)]
        else:
            return None
        texts = []
        for lab, desc in opts:
            t = f"State: {st}\nQuestion: {instr}\nOption: {lab}"
            if desc:
                t += f": {desc}"
            texts.append(t)
        return opts, texts

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False,
                             probs_source="native", model=self.model)
        res.request_body = {"task": task.id, "type": task.question.get("type")}
        try:
            if not self._loaded:
                self.load()
            built = self._pair_texts(task)
            if built is None:
                res.error = f"unsupported type {task.question.get('type')}"
                return res
            opts, texts = built
            # Left-truncate: drop old state context, never question+option.
            ids_list, tail_budget = [], self.ctx
            tail_probe = self.tok(opts[0][0] + (opts[0][1] or ""),
                                  truncation=False)["input_ids"]
            keep_state = max(self.ctx - len(tail_probe) - 64, 64)
            cut = []
            for t in texts:
                parts = t.split("\nQuestion:", 1)
                if len(parts) == 2:
                    sids = self.tok(parts[0], truncation=False)["input_ids"]
                    cut.append(self.tok.decode(sids[-keep_state:]) +
                               "\nQuestion:" + parts[1])
                else:
                    cut.append(t)
            enc = self.tok(cut, return_tensors="pt", truncation=True,
                           max_length=self.ctx, padding=True).to(self.dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                o = self.lm(input_ids=enc["input_ids"],
                            attention_mask=enc.get("attention_mask"),
                            output_hidden_states=True, use_cache=False,
                            return_dict=True)
                hs = o.hidden_states[-1].float()
                last = (enc.get("attention_mask").sum(1) - 1).clamp_min(0)
                lg = self.head(hs[torch.arange(len(texts)),
                                  last].to(self.dev)).tolist()
            res.latency_s = time.perf_counter() - t0
            m = max(v / self.temp for v in lg)
            ex = [math.exp(v / self.temp - m) for v in lg]
            s = sum(ex)
            probs = {lab: e / s for (lab, _), e in zip(opts, ex)}
            res.probs = {k: float(v) for k, v in probs.items()}
            res.usage = {"input_tokens": int(enc.get("attention_mask").sum())}
            res.raw = {"runtime": {"trunk": self.trunk, "ctx": self.ctx,
                                   "temperature": self.temp,
                                   "revision": self.revision,
                                   "probability_origin": "native-softmax"}}
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
