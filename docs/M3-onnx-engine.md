# M3 — Distilled-LM / ONNX Engine (inference core shipped)

**Status:** shipped as an inference core (`crates/decide-onnx`). The engine
trait, registry, and HTTP server all speak ONNX today; what is *not* yet built
is the distillation pipeline that produces a production-grade model (teacher
labeling → fine-tune → export → quantize). See "Still ahead" below.

## What shipped

- **`crates/decide-onnx`** — `OnnxEngine` behind the same `Engine` trait as
  the logistic engine. Real `ort` (2.0.0-rc.x) inference, `Mutex<Session>`
  for shared use.
- **Required input mode: `HashedBow`.** Text is hashed with decide-core's
  FNV-1a bag-of-words featurizer (`hashed_bow_dense`) into one
  `[1, 16384]` float32 tensor. The model must accept a single float32 input
  of that shape.
- **Optional input mode: `TokenIds`** (cargo feature `tokenizers`):
  a HuggingFace `tokenizer.json` produces `input_ids` + `attention_mask`
  int64 tensors of shape `[1, max_length]`.
- **Schema-constrained output.** The label set is the output schema: `ask`
  runs the session, applies softmax (+ temperature), and takes the argmax
  **over the configured labels**. The engine is structurally incapable of
  emitting anything else — no free-form generation, no retry loop.
- **Calibration.** `calibrate()` runs validation samples through the model
  and fits temperature with decide-core's shared `fit_temperature`
  (the same M2 grid recipe).
- **Inference-only contract.** `train()` refuses with a clear error —
  train the model offline, load the artifact.
- **Server selection with graceful fallback.** `POST /define` accepts
  `"engine": "logistic"` (default) or
  `{"onnx": {"model": "...onnx", "tokenizer": "...json"?, "labels": [...],
  "maxLength": 512?, "temperature": 1.0?}}`. If the model fails to load, the
  server prints a stderr warning and defines the question with the logistic
  engine — it never refuses to serve.

## What changed from the original design

The original design said "no trait change is needed". In practice the
registry needed to be engine-agnostic, so the trait grew object-safe
lifecycle methods with sensible defaults:

```rust
pub trait Engine: Send {
    fn ask(&self, question: &Question, input: &serde_json::Value)
        -> Result<Decision, DecideError>;
    // defaulted: train / calibrate refuse (inference-only),
    // is_trained -> false, engine_metrics -> empty snapshot
    fn train(&mut self, examples: &[(String, String)]) -> Result<(), DecideError>;
    fn calibrate(&mut self, validation: &[(String, String)]) -> Result<(), DecideError>;
    fn is_trained(&self) -> bool;
    fn engine_metrics(&self) -> EngineMetrics;
}
```

`Registry` now holds `HashMap<String, Box<dyn Engine>>` plus
`define_question_with_engine(...)`; the original `define_question(...)`
still creates a logistic engine, so existing callers are untouched. The
"never trained → mock stub" fallback is preserved: `is_trained()` is false
for an untrained logistic engine and true for a loaded ONNX model.

Open question #5 from the original doc (fallback story) is settled:
**fall back to logistic, warn loudly, keep serving.**

## Latency (synthetic fixture)

`crates/decide-onnx/tests/fixtures/bow_3label.onnx` is a tiny
Gemm-only model (input `[1,16384]` → 3 logits). Measured on this machine:

- `ask` p50 ≈ **56 µs**, p99 ≈ 1.5 ms (n=200, warmed; p99 dominated by a
  cold scheduling blip).

Real distilled models will be slower — tens of ms on CPU for a 100–300M
model — but the harness and the measurement method are in place.

## Still ahead: the distillation pipeline

The inference core is ready; a *production* model is a separate project.
The recipe, when there is real traffic worth distilling against:

1. **Teacher labeling.** A large model (or human review) labels a corpus of
   real inputs per question. Human spot-checks on a sampled subset —
   distillation amplifies teacher errors.
2. **Fine-tune a small classifier** (e.g. ModernBERT) on the labeled corpus.
3. **Export to ONNX** (opset 17+, static shapes where possible).
4. **Int8 quantize** for CPU serving.
5. **Configure gavel** with the artifacts:
   `{"engine": {"onnx": {"model": "model.int8.onnx",
   "tokenizer": "tokenizer.json", "labels": [...]}}}`.
6. **Evaluate against the logistic baseline** on the same harness
   (decide-bench): accuracy, latency p50/p99, ECE before/after temperature
   scaling. Ship only if it wins on accuracy without losing on calibration.

See `crates/decide-onnx/README.md` for the operator-facing version of this
recipe and the fixture regeneration script.

## Original design notes (kept for context)

- **Candidate model classes** (unchanged): distilled per-question
  classifiers are the cheap, narrow option; small instruction models with
  constrained decoding are the general option. Both export to ONNX;
  the tokenizer ships with the model artifacts.
- **Constrained decoding** (unchanged principle): for classification the
  label set *is* the constraint — argmax over labels, no token masking
  needed. For richer schemas later, compile the schema to a token
  allow-list / grammar FSM and mask logits during decoding.
- **Open questions carried forward**: how much real traffic is "enough"
  before distillation stops embarrassing itself OOD (#1); one model per
  question vs shared backbone (#2); calibration under distribution shift —
  periodic `/calibrate` refit vs online (#3); `ort` vs a minimal custom
  runtime — `ort` won for now (#4).
