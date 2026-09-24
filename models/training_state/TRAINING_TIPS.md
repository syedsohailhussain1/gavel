# Training tips: consistent 97%+ from tiny Gavel questions

How we take small supervised questions to high accuracy *and* high play
quality, measured on Snake (12x12-36x36, empty + pillars) and 2048.
Nothing here is theory: every rule cost us a failed run first.

## 0. The split that matters

Accuracy (label agreement on held-out teacher states) and quality
(closed-loop behavior: deaths, vetoes, hunger, tiles) are different
numbers. We have measured 97.7% accuracy play perfectly (Snake board
clear, 0 overrides) and 84% accuracy starve completely (0 apples). Tune
for both, report both, never infer one from the other.

## 1. One cheap question per regime

Gavel's unit is a small specialist, not a generalist. New board size, new
map, new task mix -> new question (or mixed-regime training), dispatched
by a router. Measured: one question trained on 12x12 scored 97.7% at home
but 81.0% served on 24x24 states. Adding 13.5k new-regime rows + 2.3s
retrain -> 94.3% new / 93.7% home. Adaptation is data + seconds, not a
new serving stack.

## 2. Capacity matched to the surface

- Plateau on **train** accuracy = model too small. We measured TF-IDF
  stuck at 84% train on pillar routing (underfit: DAgger and rebalancing
  cannot fix underfit). Climb the ladder (tiny transformer does 95-99.5%).
- Plateau only on **val** with train near 100% = need data/diversity.
- Price list (measured): TF-IDF trains in ~1-4s, decides in ~0.2-0.5ms
  over HTTP (2.5us in-process); 4.4M transformer trains in ~1min on GPU,
  decides in ~0.7-1.5ms CPU.

## 3. Teachers generate labels; models make decisions

Heuristics and proofs are *teachers*, never players. The demo menu must
not offer a hardcoded brain next to trained ones without labeling it:
we did, and it nearly became the story. Distill strong teachers into
questions (Hamiltonian jumps-bfs -> 0-death Snake labels; expectimax ->
2048 labels; BFS router -> pillar labels), then every live move is a
real `/ask` with calibrated confidence.

## 4. Train/serve byte parity, asserted at runtime

Two real bugs, both silent accuracy killers:

- **Format skew**: data built in the transformer's colon format, served
  in the logistic underscore format. Per-move facts went OOV at serve;
  the model played blind (val looked fine: 93%!). Fix: one text builder
  shared by training and serve (`legacy_text`), or per-mode builders
  with a parity assert. Our data generator now asserts
  `phase_of(game, danger) == phase` on every TF-IDF row.
- **Lying features**: pillar-adjacent cells reported as `free`. The
  model can't learn what the text won't say. Per-move facts now report
  `pillar`; empty-board strings are byte-identical to before.
- **Rule parity**: the phase threshold lived in two places (26 vs 20).
  Serve recomputes the prefix, so any constant drift is train/serve
  skew. Single source of truth, asserted.

Rule: if training and serving don't share code for input building, they
will diverge. Assert, don't eyeball.

## 5. Ask the model to be one good thing

Small models asked to be everything learn nothing; asked for one thing
they learn it. Measured: a 4.4M model never learned Snake strategy, it
learned pursuit while flood-fill/BFS did survival (4.6s TF-IDF baseline
89.4% -> veto receipt shows the split). On pillars, hunt-only training
(13k rows) beat mixed training (33k rows): 10 apples vs 0. Filter the
data to the job, let the architecture (shield, tail invariant, safe
sets) do the rest.

## 6. Make accuracy non-load-bearing (choice architecture)

Don't verify safety move-by-move; constrain the choice. List the valid
options in the input (`options_*` tokens), train the model to pick among
them, replace out-of-set outputs deterministically, count every
override. Measured (Snake): 97.7% val -> board clear, 0 deaths, 0
overrides; hunger matches the teacher (19.5 moves/apple). Accuracy then
buys efficiency, never survival. This is the portable pattern for user
use cases: define the safe set, let the tiny model choose inside it.

## 7. Calibrate and abstain per regime

Temperature and ECE on held-out data *from the regime being served*;
abstention energy thresholds likewise (a p95 gate from one distribution
means nothing on another). Our Snake choice question: ECE 0.0040->0.0018
(T=0.8942). Refit on every retrain; publish before/after.

## 8. Validate closed-loop, expect gaps, DAgger with care

Val accuracy is agreement on *teacher-visited* states. Grade the student
on *student-visited* states (record, replay, compare): we measured
93.3% -> 50.9% (Snake raw) and 79.7% -> 41.7% (pillars). DAgger (label
the student's states, retrain) helps only if the model has capacity to
absorb them: it backfired on linear TF-IDF (10 -> 2 apples, reverted).
Capacity first (rule 2), DAgger second.

## 9. Know your ceilings, state them

- No student beats its teacher's archetype (19.4 moves/apple cycle laps
  grow with board size: ~89 on 36x36 -- structural, not learnable).
- Some directions are tried-and-killed: dynamic cycle repair (dead),
  learned rankers over fixed cycles (no gain), free RL without a net
  (20/20 deaths). The remaining gap is usually the price of the proof.
- Per-move certificates don't compose into closed-loop safety (hungry
  certified planner: 7 deaths). Prefer structural invariants.

## 10. Demos prove infrastructure, not games

Every demo number must be live, measured, and reproducible: decisions/s,
ms/decision with the network stack included, agreement %, override/veto
counters on screen, seeds disclosed (showcase seed AND the 1-10 mean).
Bit-exact reruns where deterministic. The game is the showroom; the
product is typed decisions at $0 marginal cost. Never let a heuristic
wear the product's clothes.
