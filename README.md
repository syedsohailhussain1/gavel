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

## Status: M1 ✅ M2 ✅ M4 ✅ M5-lite ✅ — M3 deferred by design

- [x] **M1 — Real engine:** `LogisticEngine` (multinomial logistic regression on hashed bag-of-words features, pure Rust, zero new dependencies) behind the `Engine` trait
- [x] **M2 — Calibration:** per-question temperature scaling + ECE reporting via `train` / `calibrate` / `metrics`
- [x] **M4 — Honest benchmarks:** `decide-bench` harness — reproducible latency / accuracy / calibration numbers, wins and losses both published
- [x] **M5-lite — Real-world testable server:** `gavel-server` (dependency-light HTTP server on `:7575`) + stdlib-only Python client + runnable demo
- [ ] **M3 — Distilled-LM / ONNX engine:** deferred on purpose — needs real usage data and the M4 harness first. Design doc only: [docs/M3-onnx-engine.md](docs/M3-onnx-engine.md)
- [ ] **M5 (full) — Launch:** docs, playground, launch post

## Try it on another device

`gavel-server` binds `0.0.0.0:7575`, so anything on your LAN can reach it. On the machine that will serve:

```bash
cargo build -p decide-server --release
./target/release/gavel-server
# gavel-server listening on 0.0.0.0:7575
```

From any other device — use `localhost` if it's the same machine, or swap in the server's LAN IP (e.g. `192.168.1.20`) if it's across the room:

```bash
# define a question
curl -s -X POST http://localhost:7575/define -H 'Content-Type: application/json' -d \
  '{"name":"route_ticket","inputSchema":{"type":"object"},"outputSchema":{"type":"object"},"policy":{"act_above":0.9,"review_low":0.6,"review_high":0.9}}'

# train it (a few examples per class is enough to see it work)
curl -s -X POST http://localhost:7575/train -H 'Content-Type: application/json' -d \
  '{"question":"route_ticket","examples":[{"input":"charged twice on my card","label":"billing"},{"input":"refund my invoice","label":"billing"},{"input":"app crashes on upload","label":"technical"},{"input":"500 error from the api","label":"technical"}]}'

# ask it
curl -s -X POST http://localhost:7575/ask -H 'Content-Type: application/json' -d \
  '{"question":"route_ticket","input":{"subject":"refund my invoice"}}'
# {"action":"act","confidence":0.983,"output":{"label":"billing"}}
```

Or run the full Python demo (server must be up; stdlib only, no pip install):

```bash
python3 py/example.py
```

It defines `route_ticket`, trains on 30 synthetic tickets, calibrates on 9, then asks 4 (clear billing, clear technical, ambiguous, gibberish) and prints each decision with its confidence and action.

## Roadmap

- **M1 — Real engines:** ✅ shipped (`LogisticEngine`, sub-10ms, `define()`-stable API)
- **M2 — Calibration:** ✅ shipped (temperature scaling per decision, ECE reporting)
- **M3 — Distilled LM engine:** design doc only ([docs/M3-onnx-engine.md](docs/M3-onnx-engine.md)) — deferred until real usage data + the M4 harness exist
- **M4 — Honest benchmarks:** ✅ `decide-bench` — including where we lose
- **M5 — Launch:** docs, playground, launch post

## Layout

```
crates/decide-core/   Rust core: types, policy engine, Engine trait, LogisticEngine
crates/decide-napi/   Node.js bindings (napi-rs)
crates/decide-server/ gavel-server: minimal HTTP API over decide-core (M5-lite)
crates/decide-bench/  Benchmark harness (M4)
ts/decide/            TypeScript SDK (@gavel/decide) + examples
py/                   Python client (stdlib only) + runnable demo
docs/                 Design docs (M3 deferred engine, …)
```

## License

MIT — see [LICENSE](LICENSE).
