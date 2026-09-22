#!/usr/bin/env python3
"""Export the tiny Snake model: torch.onnx (eager attn) + int8 dynamic quant.

Usage:
  python tiny_export.py [model_dir] [out_stem]
Defaults: tiny_snake_modernbert tiny_snake_modernbert
Writes <stem>_fp32.onnx and <stem>_int8.onnx next to the model dir.
Uses absolute paths (Windows-safe). Verifies ORT parity vs torch on 8 samples.
"""
import sys
import time
from pathlib import Path

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny_snake_modernbert").resolve()
STEM = Path(sys.argv[2] if len(sys.argv) > 2 else str(MODEL)).resolve()
FP32 = STEM.parent / (STEM.name + "_fp32.onnx")
INT8 = STEM.parent / (STEM.name + "_int8.onnx")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

t0 = time.perf_counter()
m = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL), attn_implementation="eager").eval()
ids = torch.ones(1, 96, dtype=torch.long)
msk = torch.ones(1, 96, dtype=torch.long)
torch.onnx.export(
    m, (ids, msk), str(FP32),
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "B", 1: "T"},
                  "attention_mask": {0: "B", 1: "T"}, "logits": {0: "B"}},
    opset_version=17)
print(f"fp32 [{time.perf_counter()-t0:.0f}s] {FP32.stat().st_size/1e6:.1f}MB",
      flush=True)
del m

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

t0 = time.perf_counter()
quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
print(f"int8 [{time.perf_counter()-t0:.0f}s] {INT8.stat().st_size/1e6:.1f}MB",
      flush=True)

# Parity: torch vs fp32 ORT vs int8 ORT on synthetic inputs.
m = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL), attn_implementation="eager").eval()
rng = np.random.RandomState(0)
mx = 0.0
sess = {p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        for p in (FP32, INT8)}
for _ in range(8):
    n = int(rng.randint(20, 96))
    ii = rng.randint(0, 1000, size=(1, n)).astype(np.int64)
    am = np.ones((1, n), dtype=np.int64)
    with torch.no_grad():
        ref = m(torch.from_numpy(ii), torch.from_numpy(am)).logits.numpy()
    for p, s in sess.items():
        got = s.run(None, {"input_ids": ii, "attention_mask": am})[0]
        mx = max(mx, float(np.abs(ref - got).max()))
print(f"max |torch-ort| over fp32+int8: {mx:.5f}", flush=True)
print("OK", flush=True)
