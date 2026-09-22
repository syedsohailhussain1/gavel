# Cookbook: Snake, played by tiny models making typed decisions

Every move of a Snake game is a typed decision — `up | down | left | right`
with a calibrated confidence — made by a small locally-trained model. No
text generation, no cloud API, no GPU required. This cookbook runs the demo
two ways (a TF-IDF baseline and a fine-tuned tiny transformer) and reports
exactly what each cost and scored.

Scripts live in `models/training_state/` (`gavel_snake.py` plays and renders,
`showcase_snake.py` trains the legacy question, `tiny_*.py` train/bench the
transformers, `legacy_tfidf.py` trains the TF-IDF question). Model weights
stay on your machine (gitignored) — train them locally with the commands
below; nothing is downloaded except public base checkpoints.

## Setup

```bash
cargo build -p decide-server --release
./target/release/gavel-server &   # :7575, 27 MB, CPU only
```

Legacy question (TF-IDF + logistic regression, sklearn-trained, served by
the native Rust engine):

```bash
python models/training_state/legacy_tfidf.py \
  tiny_snake4_100k_train.jsonl tiny_snake4_100k_val.jsonl \
  snake_tfidf snake_tfidf_prod.json
# train 90,000 examples in ~4.6 s on CPU; val accuracy 89.4%, ECE 0.006→0.004
```

Transformer brain (fine-tune a small encoder on 27k rollout states, export
to ONNX; the 11M run below took ~17 min on a GTX 1650):

```bash
python models/training_state/tiny_train.py \
  models/training_state/bert11m_src models/training_state/tiny_snake.jsonl \
  models/training_state/tiny_snake_bert11m 20 128
python models/training_state/tiny_export.py \
  models/training_state/tiny_snake_bert11m models/training_state/tiny_snake_bert11m
# val accuracy 99.5%; ONNX fp32; parity vs torch exact
```

## How we play

```bash
# Legacy mode: every move is a live /ask call (keep-alive HTTP)
python models/training_state/gavel_snake.py play --brain server --fps 12
# Transformer mode: every move is a live in-process ONNX call
python models/training_state/gavel_snake.py play --brain onnx --fps 12
# Max speed (no pacing) for throughput numbers
python models/training_state/gavel_snake.py play --brain server --max-speed --steps 6000
```

A disclosed safety shield guards every move: it vetoes instant suicide
(walls, body) and, in tight spots, trap moves found by flood-fill lookahead
and a Hamiltonian-cycle substrate. Every veto is counted on screen — the
shield's intervention rate is reported next to the score, not hidden.

## What came back

Measured on a GTX 1650 / 8-core CPU box, 12×12 board, fixed seeds.
`val acc` is agreement with the teacher policy on held-out states; `play`
is closed-loop games; `veto` is shield interventions per move.

| Brain | Params | Train | Val acc | Raw decision | Live loop | Play (best) |
|---|---|---|---|---|---|---|
| Legacy TF-IDF | ~5k weights | 4.6 s CPU | 89.4% | 0.2 ms | ~820/s | 176 pts / 6,000 moves |
| google-tiny | 4.4M | ~2 min GPU | 95.1% | 0.7 ms CPU | ~650/s | **board clear, 141 pts, 0 deaths** |
| bert-11M | 11M | ~17 min GPU | 99.5% | 3.6 ms GPU | ~190/s | ~47 pts avg, rare endgame deaths |
| distil-66M | 66M | ~38 min GPU | 98.8% | 4.5 ms GPU | ~43/s | ~47 pts avg, rare endgame deaths |
| ModernBERT-150M | 150M | ~48 min/epoch GPU | 97.8% | 9–71 ms | — | rejected (see below) |

Reference point we measured ourselves on this box: a hosted zero-shot
decision API at 436 ms p50 and $0.042/M input tokens (≈$0.016 per 1,000
short decisions). Against it, the 4.4M model is ~600× faster on CPU alone
at $0 marginal cost.

## The pattern

Speed and intelligence split cleanly here, and the split is the point.
The 4.4M model never learned grand strategy — it learned pursuit (simple,
regular labels), while flood-fill, BFS pathfinding, and cycle-following do
survival duty algorithmically. Each side does what it's good at; the veto
counter on screen is the receipt showing exactly how much each side
contributed. Small models are only "dumb" when you ask them to be
everything. Ask them to be one good thing and guard the rest.

## Caveats — read before citing any number above

- **Val accuracy ≠ play quality, repeatedly.** A 92%-combined phase-split
  model played worse (lower score, 10× vetoes) than the 89% mixed model; a
  model that survives perfectly can still never eat. Every row above earned
  its place with closed-loop games, not val numbers.
- **The shield does real work.** Veto rates run 1–50% depending on brain
  and game phase. Zero-veto play is not claimed anywhere; the counter is
  part of the demo output.
- **Endgame deaths happen** (greedy pursuit boxes itself around 25–30%
  board fill, roughly every few hundred moves depending on brain). Rounds
  auto-reset with cumulative scoring; a `--immortal` cycle-discipline mode
  reaches zero deaths at the price of slow scoring.
- **Machine-specific numbers.** Latencies are batch-1 on the box above;
  relative ordering is robust, exact milliseconds are not. Costs exclude
  owned hardware and electricity.
- **A toy game, honestly labeled.** Snake states are short structured
  texts, not open language. This cookbook proves cheap typed decisions can
  *drive* a real-time loop — not that a 4.4M model understands anything.
