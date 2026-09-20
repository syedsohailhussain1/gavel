#!/usr/bin/env python3
"""Download + build the tiny sentiment model used by the gavel M3 ONNX demo.

One command:
    pip install torch optimum[onnxruntime] onnxruntime tokenizers huggingface_hub
    python download.py

What it does:
  1. Downloads the PyTorch checkpoint `philschmid/tiny-bert-sst2-distilled`
     (TinyBERT, 4 layers, ~14.5M params, fine-tuned for SST-2 sentiment)
     from HuggingFace.
  2. Exports it to ONNX with HuggingFace Optimum.
  3. Applies dynamic int8 quantization via onnxruntime, so the final
     artifact is small and fast on CPU with no GPU needed.

Output: models/tinybert-sst2/{model.onnx,tokenizer.json,config.json}
Re-running is safe: it skips the build if model.onnx already exists
(pass --force to rebuild).
"""

import os
import shutil
import sys

# Workaround for a known Python urllib issue: bracketed IPv6 entries
# (e.g. "[::1]") in no_proxy make proxy handling raise "Invalid port".
# Dropping just the bracketed entries is safe and only affects this process.
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val and "[" in _val:
        os.environ[_var] = ",".join(p for p in _val.split(",") if "[" not in p)

MODEL_ID = "philschmid/tiny-bert-sst2-distilled"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinybert-sst2")
FP32_DIR = os.path.join(OUT_DIR, "_fp32_export")


def main() -> int:
    force = "--force" in sys.argv
    final_model = os.path.join(OUT_DIR, "model.onnx")
    if os.path.exists(final_model) and not force:
        print(f"Already built: {final_model} (pass --force to rebuild)")
        return 0

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FP32_DIR):
        shutil.rmtree(FP32_DIR)

    print(f"[1/4] Downloading checkpoint {MODEL_ID} ...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID, export=True
    )
    print("[2/4] Exporting to ONNX ...")
    ort_model.save_pretrained(FP32_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(OUT_DIR)
    # keep config.json alongside the final artifacts
    ort_model.config.save_pretrained(OUT_DIR)

    fp32_model = os.path.join(FP32_DIR, "model.onnx")
    size_fp32 = os.path.getsize(fp32_model) / 1e6
    print(f"[3/4] Quantizing (dynamic int8) ... (fp32 was {size_fp32:.1f} MB)")
    quantize_dynamic(fp32_model, final_model, weight_type=QuantType.QInt8)

    shutil.rmtree(FP32_DIR)
    size_q = os.path.getsize(final_model) / 1e6
    print(f"[4/4] Done: {final_model} ({size_q:.1f} MB)")
    print("Run: python test_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
