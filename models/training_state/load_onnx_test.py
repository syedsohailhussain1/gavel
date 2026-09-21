#!/usr/bin/env python3
"""Load task_mlp_win.onnx into Gavel, calibrate, compare vs logistic."""
import json
import random
import urllib.request
from collections import Counter

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import GAVEL_URL as BASE, TS as _TS, QA_SEEDS
LABELS = ["ARC-Easy", "CommonsenseQA", "MATH/algebra", "MetaMathQA",
          "NuminaMath-CoT", "OrcaMath", "QASC", "SciQ", "file_ops", "multi_research"]


def api(path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print("define:", api("/define", {
    "name": "task_router_onnx",
    "inputSchema": {"type": "object"},
    "outputSchema": {"type": "object"},
    "engine": {"onnx": {
        "model": str(_TS) + "/task_mlp_win.onnx",
        "labels": LABELS}}}), flush=True)

rng = random.Random(20260920)


def split(items, f=0.8):
    by = {}
    for i, l in items:
        by.setdefault(l, []).append((i, l))
    tr, va = [], []
    for lab, exs in by.items():
        rng.shuffle(exs)
        n = max(1, int(len(exs) * f)) if len(exs) > 1 else 1
        tr += exs[:n]
        va += exs[n:] if len(exs) > 1 else []
    return tr, va


seeds = json.load(open(QA_SEEDS))
split([(x["question"], x["domain"]) for x in seeds])
keep = {t for t, c in Counter(x["task_type"] for x in seeds).items() if c >= 30}
pool = {}
for x in seeds:
    if x["task_type"] in keep:
        pool.setdefault(x["task_type"], []).append((x["question"], x["task_type"]))
items = []
for t, exs in pool.items():
    rng.shuffle(exs)
    items += exs[:80]
_, val = split(items)
print("val:", len(val), flush=True)

try:
    print("train attempt:", api("/train", {
        "question": "task_router_onnx",
        "examples": [{"input": "x", "label": "ARC-Easy"}]}), flush=True)
except Exception as e:
    print("train refuses (expected, inference-only):", str(e)[:150], flush=True)

print("calibrate:", api("/calibrate", {
    "question": "task_router_onnx",
    "validation": [{"input": i, "label": l} for i, l in val]}), flush=True)
print("metrics:", json.dumps(api("/metrics?question=task_router_onnx")), flush=True)

correct = 0
for i, l in val[:30]:
    rr = api("/ask", {"question": "task_router_onnx", "input": i}, timeout=30)
    got = (rr.get("output") or {}).get("label", "")
    mark = "OK " if got == l else "MISS"
    if got == l:
        correct += 1
    print(f"[{mark}] want={l} got={got} conf={rr.get('confidence', 0):.3f} "
          f"{rr.get('action')}", flush=True)
print(f"spot {correct}/30", flush=True)
print("DONE", flush=True)
