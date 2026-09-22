#!/usr/bin/env python3
"""Retrain the Legacy (server/logistic) question on rollout data.

Same treatment the 4.4M model got: train on REAL game states (not uniform
random) with guarded-policy labels and bucketed features. Small BOW vocab
(fC/fM/fF, sT/sR/sO) keeps hash collisions low for the linear engine.

Usage:
  python legacy_retrain.py [train_jsonl] [val_jsonl] [question] [max_train]
Defaults: tiny_snake3_train.jsonl tiny_snake3_val.jsonl snake_move 0 (all)
Reports train time, val accuracy, calibration. Stdlib only, needs server.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7575"
TRAIN = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake3_train.jsonl"
VAL = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake3_val.jsonl"
QUESTION = sys.argv[3] if len(sys.argv) > 3 else "snake_move"
MAXN = int(sys.argv[4]) if len(sys.argv) > 4 else 0


def api(path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


train = json.load(open(TRAIN, encoding="utf-8"))
val = json.load(open(VAL, encoding="utf-8"))
if MAXN:
    train = train[:MAXN]
print(f"train={len(train)} val={len(val)} -> {QUESTION}", flush=True)

t0 = time.perf_counter()
api("/define", {"name": QUESTION, "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"}})
api("/train", {"question": QUESTION, "examples": [
    {"input": r["text"], "label": r["label"]} for r in train]})
train_s = time.perf_counter() - t0
print(f"train_seconds={train_s:.3f} ({len(train)/max(train_s,1e-9):.0f} ex/s, CPU)",
      flush=True)

t0 = time.perf_counter()
api("/calibrate", {"question": QUESTION, "validation": [
    {"input": r["text"], "label": r["label"]} for r in val]})
cal_s = time.perf_counter() - t0
m = api(f"/metrics?question={QUESTION}")
print(f"calibrate_seconds={cal_s:.3f} T={m.get('temperature')} "
      f"ece={m.get('ece_before')}->{m.get('ece_after')}", flush=True)

ok = 0
for r in val:
    a = api("/ask", {"question": QUESTION, "input": r["text"]}, timeout=30)
    if (a.get("output") or {}).get("label") == r["label"]:
        ok += 1
print(f"val_accuracy={ok}/{len(val)}={ok/max(len(val),1):.4f}", flush=True)
print("LEGACY RETRAIN DONE", flush=True)
