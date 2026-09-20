# Gavel — the open typed-decision layer for AI agents

Structured decisions in, calibrated answers out. No prose generation, no vendor lock-in.

Most of what agents use LLMs for — routing, classification, scoring, extraction — is really a *decision* hiding inside a prompt. Verdict is the open-source layer that makes those decisions typed, calibrated, fast, and cheap:

```ts
const { ask } = define({
  name: "route_ticket",
  inputSchema: { type: "object", properties: { subject: { type: "string" } } },
  outputSchema: { type: "object", properties: { route: { enum: ["billing", "support", "escalate"] } } },
  policy: { actAbove: 0.9, reviewLow: 0.6, reviewHigh: 0.9 },
});

const d = ask({ subject: "Charged twice!" });
// { output: { route: "billing" }, confidence: 0.94, action: "act" }
```

**Why open wins here:** reproducible benchmarks, self-hostable (your data never leaves), MIT-licensed, and calibration as a published science — not a marketing claim.

## Quickstart

Prerequisites: Rust (stable), Node.js 18+.

```bash
# 1. Build the Rust core + native binding
cargo test                                   # run core unit tests
cd crates/decide-napi && npm install && npm run build && cd ../..

# 2. Build the TypeScript SDK and run the example
cd ts/decide && npm install && npm run build && npm run example
```

You should see the ticket-routing decision printed as JSON with `action: "act"`.

## Status: M0 ✅

- [x] `decide-core` (Rust): `Question`, `Decision`, `Action`, `Policy`, `Engine` trait, `MockEngine` stub — tested
- [x] `decide-napi`: Node bindings (`DecideEngine`: `defineQuestion` / `ask` / `questions`)
- [x] `@gavel/decide` (TypeScript SDK): `define()` / `ask()` API over the native binding
- [x] Example: ticket routing end-to-end

## Roadmap

- **M1 — Real engines:** embed+classify engine (sub-10ms), `define()`-stable API, three reference decisions (routing, moderation, tool-call selection)
- **M2 — Calibration:** temperature/Platt scaling per decision, ECE reporting, published calibration methodology
- **M3 — Distilled LM engine:** distillation pipeline, constrained decoding (schema enforced at decode time), ONNX export, weights on HuggingFace
- **M4 — Honest benchmarks:** reproducible latency / cost / accuracy / calibration harness vs. LLM baselines — including where we lose
- **M5 — Launch:** docs, playground, launch post

## Layout

```
crates/decide-core/   Rust core: types, policy engine, Engine trait
crates/decide-napi/   Node.js bindings (napi-rs)
ts/decide/            TypeScript SDK (@gavel/decide) + examples
```

## License

MIT — see [LICENSE](LICENSE).
