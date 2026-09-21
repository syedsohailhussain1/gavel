#!/usr/bin/env python3
"""Distill task_router into a small MLP -> HashedBow ONNX (M3 step 3a: seeds-only student).

Replicates retrain_all.py steps 1-2 splits EXACTLY (seed 20260920) so the
val set (124) matches the logistic baseline (acc 0.798, ECE 0.049).
Featurizer mirrors decide-core (FNV-1a, lowercase alnum unigrams, TF, 2^14).
Output: models/training_state/task_mlp.onnx  (Gemm->Relu->Gemm, logits)
"""
import json
import random
import re
import numpy as np

N_BUCKETS = 16384
LABELS = sorted(["MetaMathQA", "SciQ", "CommonsenseQA", "NuminaMath-CoT",
                 "OrcaMath", "QASC", "ARC-Easy", "multi_research",
                 "file_ops", "MATH/algebra"])
LI = {l: i for i, l in enumerate(LABELS)}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)  # alnum runs (no underscore)


def fnv1a(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def bow(text: str) -> np.ndarray:
    v = np.zeros(N_BUCKETS, dtype=np.float32)
    for tok in TOKEN_RE.findall(text.lower()):
        v[fnv1a(tok) % N_BUCKETS] += 1.0
    return v


rng = random.Random(20260920)


def split_examples(items, train_frac=0.8):
    by_label = {}
    for inp, lab in items:
        by_label.setdefault(lab, []).append((inp, lab))
    train, val = [], []
    for lab, exs in by_label.items():
        rng.shuffle(exs)
        n_train = max(1, int(len(exs) * train_frac)) if len(exs) > 1 else 1
        train += exs[:n_train]
        val += exs[n_train:] if len(exs) > 1 else []
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


with open("D:/virtual-brain/distillation_seeds/qa_seeds.json") as f:
    seeds = json.load(f)
_ = split_examples([(x["question"], x["domain"]) for x in seeds])  # step 1: advance rng identically

from collections import Counter
keep = {t for t, c in Counter(x["task_type"] for x in seeds).items() if c >= 30}
pool = {}
for x in seeds:
    if x["task_type"] in keep:
        pool.setdefault(x["task_type"], []).append((x["question"], x["task_type"]))
items = []
for t, exs in pool.items():
    rng.shuffle(exs)
    items += exs[:80]
train, val = split_examples(items)
print(f"train={len(train)} val={len(val)}", flush=True)
assert len(val) == 124, f"split diverged: {len(val)}"

Xtr = np.stack([bow(t) for t, _ in train])
ytr = np.array([LI[l] for _, l in train])
Xva = np.stack([bow(t) for t, _ in val])
yva = np.array([LI[l] for _, l in val])

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(256,), activation="relu",
                    max_iter=400, random_state=0, verbose=False)
print("training MLP...", flush=True)
clf.fit(Xtr, ytr)
acc_tr = clf.score(Xtr, ytr)
acc_va = clf.score(Xva, yva)
print(f"sklearn train_acc={acc_tr:.4f} val_acc={acc_va:.4f} iters={clf.n_iter_}", flush=True)

# Pack to ONNX: bow[batch,16384] -> Gemm(W1,b1) -> Relu -> Gemm(W2,b2) -> logits[batch,10]
import onnx
from onnx import helper, TensorProto

W1 = clf.coefs_[0].astype(np.float32)      # [16384,256]
b1 = clf.intercepts_[0].astype(np.float32)  # [256]
W2 = clf.coefs_[1].astype(np.float32)      # [256,10]
b2 = clf.intercepts_[1].astype(np.float32)  # [10]
gemm1 = helper.make_node("Gemm", ["bow", "W1", "B1"], ["h"], alpha=1.0, beta=1.0, transB=0)
relu = helper.make_node("Relu", ["h"], ["hr"])
gemm2 = helper.make_node("Gemm", ["hr", "W2", "B2"], ["logits"], alpha=1.0, beta=1.0, transB=0)
graph = helper.make_graph(
    [gemm1, relu, gemm2], "task_mlp_bow",
    [helper.make_tensor_value_info("bow", TensorProto.FLOAT, ["batch", N_BUCKETS])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(LABELS)])],
    initializer=[
        helper.make_tensor("W1", TensorProto.FLOAT, [N_BUCKETS, 256], W1.flatten().tolist()),
        helper.make_tensor("B1", TensorProto.FLOAT, [256], b1.tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, [256, len(LABELS)], W2.flatten().tolist()),
        helper.make_tensor("B2", TensorProto.FLOAT, [len(LABELS)], b2.tolist()),
    ])
model = helper.make_model(graph, producer_name="gavel-distill-v1",
                          opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
onnx.checker.check_model(model)
out = "D:/gavel/models/training_state/task_mlp.onnx"
onnx.save(model, out)
print(f"saved {out}", flush=True)

# Local ORT eval (same val): proves ONNX == sklearn
import onnxruntime as ort
sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
logits = sess.run(None, {"bow": Xva})[0]
pred = logits.argmax(axis=1)
acc_onnx = (pred == yva).mean()
print(f"onnxruntime val_acc={acc_onnx:.4f} (must equal sklearn {acc_va:.4f})", flush=True)
print("LABELS:", json.dumps(LABELS), flush=True)
