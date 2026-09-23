# Cookbook: Tetris, played by one typed decision per piece

Tetris is the second capability demo (the game is the showroom, Gavel is
the product). For every falling piece, a Dellacherie heuristic shortlists
the two strongest legal landings and **one** `/ask` call to `tetris_pick`
picks the cleaner one. No text generation, no cloud API, CPU only.

Scripts live in `models/training_state/` (`gavel_tetris.py` plays and
renders, `tetris_game.py` is the engine + heuristic + data gen,
`tetris_dagger.py` harvests on-policy corrections). The question artifact
rebuilds locally in under a second — nothing to download.

## Setup

```bash
cargo build -p decide-server --release
./target/release/gavel-server &   # :7575, 27 MB, CPU only
python models/training_state/tetris_game.py tetris_pairs.jsonl 20000 40
python models/training_state/legacy_tfidf.py \
  tetris_pairs_train.jsonl tetris_pairs_val.jsonl tetris_pick tetris_pick.json
# train 18,000 pairs in ~0.3 s on CPU; val accuracy 98.4%, ECE 0.004→0.002
```

A second DAgger round (harvest states the model itself visits, label with
the heuristic, mix back in) holds fidelity on live boards; see
`tetris_dagger.py`. Final question: 24k pairs, 98.5% val.

## How we play

```bash
python models/training_state/gavel_tetris.py play --fps 8
python models/training_state/gavel_tetris.py play --max-speed --pieces 2000
```

Near-ties (heuristic gap < 3.0, outside training distribution),
low-confidence calls (< 0.70 — calibrated, so it means something), and high
stacks (above `--maxh`, default 8) fall back to the heuristic directly.
Every fallback is counted on screen next to the fidelity readout
(model-vs-heuristic agreement). The panel also shows live machine load
(CPU/RAM via psutil, GTX util/VRAM via nvidia-smi): the GPU row sits at
~0% throughout, because nothing here needs it.

## What came back

Measured on a GTX 1650 / 8-core CPU box, 10×20 well, fixed seeds:

| | Gavel (model) | Heuristic only | Reference (laya/Core ML, M5 Pro) |
|---|---|---|---|
| Per decision | **0.2 ms** (TF-IDF, CPU) | — | 3.7 ms (Neural Engine) |
| Pieces/sec | **~675** end-to-end | ~720 | 60/sec capped demo |
| 2,000 pieces | 91,400 pts, 797 lines, alive | — | — |
| Pure heuristic | — | 447,800 pts / 9,821 pieces | their heuristic control |
| Model fidelity | 0.90–0.98 agreement | — | port fidelity 63/63 |

## The pattern

Same split as the Snake demo, one level up: the heuristic proposes,
the model disposes — but only where it was trained to. The three fallback
rules (margin, confidence, stack height) are the actual product insight:
a cheap model plus explicit "I don't know" gates beats an expensive model
with no gates, at 3,000× the speed. Every gate is a counter on screen, so
the division of labor is auditable per run, not a story told afterward.

## Caveats — read before citing any number above

- Guideline-ish rules only: 7-bag randomizer, no hold, no T-spins, next
  piece preview only. Real Tetris strategy (T-spin setups, bag counting)
  is out of scope — this demos fast comparative judgment, not mastery.
- The heuristic does heavy lifting: ties, high stacks, and ~15–50% of
  pieces depending on settings. The `--maxh` knob trades model-play vs
  survival openly; the reported runs state their setting.
- DAgger rounds were needed: the first model (98.4% val) died in a few
  hundred pieces on live boards (fidelity ~0.70) — closed-loop drift, fixed
  by training on visited states. Val accuracy lied; survival runs told.
- Box-specific numbers, batch-1, localhost. Relative ordering is robust,
  exact milliseconds are not.
