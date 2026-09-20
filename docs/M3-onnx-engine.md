# M3 — Distilled-LM / ONNX Engine (design doc, no implementation)

**Status:** deferred by design. This document records *why* M3 waits, what the
integration seam already looks like, and what has to be true before we build it.

## Why deferred

1. **Keep M1/M2 dependency-free.** The current engine is multinomial logistic
   regression on hashed bag-of-words features in pure Rust — no weights to
   download, no ONNX runtime, no tokenizer, builds in seconds. Shipping a
   distilled-LM engine now would trade that away before we know it's worth it.
2. **We need real usage data first.** Distillation needs a teacher and a
   dataset of representative inputs. Until gavel-server sees real traffic,
   we'd be distilling against synthetic data and guessing at the input
   distribution — a good way to build a worse model with extra steps.
3. **The M4 eval harness comes first.** M3 must be judged against the M1
   baseline on identical datasets (accuracy, latency, ECE). Building the
   engine before the harness exists means building without a scoreboard.

M3 starts when: (a) decide-bench is green and publishing numbers, (b) we have
a corpus of real inputs worth distilling against, and (c) a candidate model
beats the logistic baseline on that harness by enough to justify the
dependency weight.

## The seam already exists: the `Engine` trait

No trait change is needed. Today in `decide-core`:

```rust
pub trait Engine {
    fn ask(&self, question: &Question, input: &serde_json::Value) -> Result<Decision, DecideError>;
}
```

A future ONNX engine is just another implementor:

```rust
pub struct OnnxEngine { /* session, tokenizer, label map, temperature */ }

impl Engine for OnnxEngine {
    fn ask(&self, question: &Question, input: &Value) -> Result<Decision, DecideError> {
        let text = extract_text(input);
        let logits = self.session.run(&self.tokenize(&text))?;
        let label = self.constrained_decode(question, &logits)?;
        Decision::new(label, calibrated_confidence, &question.policy)
    }
}
```

`Registry` would hold `Box<dyn Engine>` per question (or an enum over engine
kinds) instead of today's concrete `LogisticEngine`; question definition,
policy mapping, training/calibration bookkeeping, and the HTTP + Node APIs
stay untouched. Calibration (M2: temperature scaling + ECE) applies
identically — it's computed over (confidence, correctness) pairs and doesn't
care where the logits came from.

## Candidate model classes

Target: 100–300M parameters, small enough to run on CPU in tens of
milliseconds, large enough to beat bag-of-words on genuinely ambiguous inputs
(negation, multi-intent tickets, domain jargon).

- **Distilled classifiers.** A task-specific head distilled from a larger
  teacher on real gavel traffic. Cheapest at inference; narrowest scope.
- **Small instruction models with constrained decoding.** More general —
  one model can serve many questions — but every token must be schema-valid,
  which is where the decoding sketch below matters.

Both export to ONNX for a single runtime dependency (`ort` or equivalent);
tokenizer stays with the model artifacts.

## Schema-constrained decoding sketch

The output schema is a contract, not a suggestion. For classification-style
questions (the M1/M2 shape), the cleanest approach:

1. At question-definition time, compile the output JSON schema into a
   **token allow-list per decode position** — for an enum output this is a
   fixed set of token sequences (e.g. the exact bytes of
   `{"label":"billing"}`).
2. During decoding, **mask logits** to the allowed set before sampling/argmax
   (standard logit bias = −∞ for disallowed tokens). With greedy decoding this
   reduces to scoring the candidate completions and picking the max — no
   free-form generation, no invalid JSON, ever.
3. Confidence comes from the softmax over the allowed completions, then
   through the same temperature scaling used in M2.

For richer output schemas (nested objects, numbers), the allow-list becomes a
finite-state machine over the schema grammar (the standard constrained-decoding
approach); same principle, more states. The key invariant: **the decoder can
only emit schema-valid output**, so validation is structural, not a retry loop.

## Eval plan (via decide-bench, M4)

Same datasets, same harness, no moving goalposts:

- **Accuracy** per question on held-out sets vs the M1 logistic baseline.
- **Latency**: p50/p99 per `ask`, CPU-only, cold and warm.
- **ECE** before/after temperature scaling — a bigger model with worse
  calibration is not an upgrade.
- **Cost**: bytes shipped (model artifacts), RAM at rest, build complexity.

Ship M3 only if it wins on accuracy *without* losing on calibration, and the
latency/cost budget is documented honestly — including the cases where the
logistic baseline stays the right choice.

## Open questions

1. **Distillation data**: synthetic-from-teacher vs real traffic — how much
   real traffic is "enough" before the distilled model stops embarrassing
   itself on out-of-distribution inputs?
2. **One model per question vs one shared model**: per-question heads are
   simpler to version; a shared backbone is cheaper to ship. Probably start
   per-question, revisit if question counts grow.
3. **Calibration under distribution shift**: temperature fitted on validation
   data degrades as traffic drifts — do we need online recalibration, or is
   periodic refit via `/calibrate` enough?
4. **ONNX runtime choice**: `ort` bindings vs a minimal custom runtime —
   dependency weight vs control. Undecided; benchmark both when M3 starts.
5. **Fallback story**: if the ONNX engine fails to load (missing weights,
   arch mismatch), the registry should fall back to the logistic engine or
   the mock stub rather than refusing to serve — decide the policy now, not
   during an incident.
