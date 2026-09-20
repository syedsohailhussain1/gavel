# decide-onnx — ONNX Runtime inference engine for gavel

M3's inference core: a distilled small-LM (or any classifier exported to
ONNX) behind the same `Engine` trait as the logistic engine. Swap the brain,
keep everything else: typed questions, policy actions (`act` / `review` /
`escalate`), temperature-scaling calibration, the HTTP server, the SDKs.

## The one invariant

For classification questions the label set **is** the output schema. `ask`
runs the session, applies softmax (+ temperature) over the model's logits,
and takes the argmax **over the configured labels**. The engine is
structurally incapable of emitting anything else — there is no free-form
generation step, so there is nothing to validate or retry.

## Input modes

| Mode | Cargo feature | Model input | Produces |
|---|---|---|---|
| `HashedBow` (default) | — | one float32 tensor `[1, 16384]` | FNV-1a hashed bag-of-words, identical to decide-core's featurizer |
| `TokenIds` | `tokenizers` | `input_ids` + `attention_mask`, int64 `[1, max_length]` | HuggingFace `tokenizer.json` encoding |

`HashedBow` is dependency-free (no tokenizer) and matches the logistic
engine's input representation, so a hashed-BoW classifier can be swapped in
with zero plumbing changes. `TokenIds` is for real distilled transformers.

## Using it

```rust
use decide_onnx::{OnnxConfig, OnnxEngine};

let mut cfg = OnnxConfig::new("model.onnx", vec!["billing".into(), "technical".into()])?;
cfg.tokenizer_path = Some("tokenizer.json".into()); // TokenIds mode (feature)
cfg.max_length = 512;
cfg.temperature = 1.0;

let engine = OnnxEngine::load(&cfg)?; // descriptive error if the model is missing/invalid
// register: registry.define_question_with_engine(name, in_schema, out_schema, policy, Box::new(engine));
```

Via the HTTP server (`POST /define`):

```json
{
  "name": "route_ticket",
  "inputSchema": {"type": "object"},
  "engine": {
    "onnx": {
      "model": "model.int8.onnx",
      "tokenizer": "tokenizer.json",
      "labels": ["billing", "technical", "account"],
      "maxLength": 512,
      "temperature": 1.0
    }
  }
}
```

Omit `"engine"` (or use `"logistic"`) for the default engine. If the model
fails to load, the server logs a warning to stderr and defines the question
with the logistic engine — it never refuses to serve.

The engine is **inference-only**: `train()` returns a clear error telling
you to train offline. `calibrate()` runs your validation set through the
model and fits temperature scaling (the same M2 recipe as the logistic
engine).

## Producing a production model (distillation recipe)

The inference core is ready; a good *model* is a separate project. When you
have real traffic worth distilling against:

1. **Teacher labeling.** Have a large model (or human reviewers) label a
   corpus of real inputs for the question. Human spot-checks on a sampled
   subset — distillation amplifies teacher errors.
2. **Fine-tune a small classifier** (e.g. ModernBERT, 100–300M params) on
   the labeled corpus.
3. **Export to ONNX** — opset 17+, static shapes where possible
   (`optimum-cli export onnx --model ... --task text-classification`).
4. **Int8 quantize** for CPU serving (ONNX Runtime dynamic quantization).
5. **Configure gavel** with the artifacts (see above).
6. **Evaluate against the logistic baseline** on the same harness
   (accuracy, p50/p99 latency, ECE before/after temperature scaling).
   Ship only if it wins on accuracy without losing on calibration.

## Test fixture

`tests/fixtures/bow_3label.onnx` is a tiny synthetic model for the
integration tests: input `bow` `[1,16384]` float32, one `Gemm` node with
`W[i][c] = 3.0` iff `i % 3 == c` and zero bias, output `logits` `[1,3]`
(raw logits — the engine applies softmax + temperature).

Regenerate it (needs Python `onnx`):

```python
import numpy as np, onnx
from onnx import helper, TensorProto

N_BUCKETS, N_LABELS = 16384, 3
W = np.zeros((N_BUCKETS, N_LABELS), dtype=np.float32)
for i in range(N_BUCKETS):
    W[i, i % N_LABELS] = 3.0
B = np.zeros((N_LABELS,), dtype=np.float32)

gemm = helper.make_node("Gemm", ["bow", "W", "B"], ["logits"],
                        alpha=1.0, beta=1.0, transB=0)
graph = helper.make_graph(
    [gemm], "bow_3label",
    [helper.make_tensor_value_info("bow", TensorProto.FLOAT, [1, N_BUCKETS])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, N_LABELS])],
    initializer=[
        helper.make_tensor("W", TensorProto.FLOAT, [N_BUCKETS, N_LABELS], W.flatten().tolist()),
        helper.make_tensor("B", TensorProto.FLOAT, [N_LABELS], B.tolist()),
    ],
)
model = helper.make_model(graph, producer_name="gavel-test-fixture",
                          opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
onnx.checker.check_model(model)
onnx.save(model, "tests/fixtures/bow_3label.onnx")
```

## Feature flags

- default: `HashedBow` only. No tokenizer dependency.
- `tokenizers`: enables `InputMode::TokenIds` via a HuggingFace
  `tokenizer.json`. Heavier dependency tree — only enable it if you are
  serving real transformer models.
