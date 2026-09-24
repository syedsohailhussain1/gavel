#!/usr/bin/env python3
"""Regime-labeled training data: hunt early, survive late.

Each state gets a phase prefix + a label from the matching policy:
- hunt (roomy board, food reachable with margin): BFS food pursuit / teacher.
  Aggressive eating — "goes for it".
- survive (long body, cramped, or food cut off): immortal cycle-discipline
  policy. Zero-death behavior.

The model learns WHEN to switch, not just how to move. Labels are
deterministic policy outputs (clean rules → very high learnability).

Usage:
  python tiny_data2.py [out_base] [total] [seeds] [teacher] [obstacles] [textfmt] [width] [height]
Defaults: tiny_snake2.jsonl 30000 200 mixed 0 legacy 12 12
Writes <base>_train.jsonl / <base>_val.jsonl. Stdlib + gavel_snake only.

teacher: mixed (default) = BFS pursuit in hunt, cycle-discipline in survive;
  immortal = jumps-bfs Hamiltonian policy for every state (0-death labels).
  choice = immortal labels PLUS an options_* suffix listing the proof's
    safe set for that state (see gavel_snake.choice_suffix -- byte-identical
    at train and serve). Train a question on choice data and serve it with
    gavel_snake --brain server --safe-choices: the model picks AMONG safe
    moves, so survival/convergence hold by construction.
  pillars = RouterBrain labels on pillar boards (obstacles > 0 required)
    with an options_* suffix listing the DANGER-FREE moves for that state.
    The router hunts where the Hamiltonian teacher starves, and the tail
    invariant keeps its labels death-free. Serve with --brain server
    --safe-choices --obstacles N (same N): options match by construction.

TEXT FORMAT CONTRACT: modes immortal/choice/pillars feed TF-IDF questions,
which are served with gavel_snake.legacy_text -- so those modes emit
legacy_text verbatim (asserted phase-identical below). Serving anything
else makes per-move facts OOV at serve time and the model plays blind.
textfmt=brain switches pillars output to gavel_snake.brain_text (colon
format) for TRANSFORMER fine-tuning, served via --brain onnx.
Mode mixed keeps the legacy colon format (transformer lineage: matches
gavel_snake.brain_text); do NOT serve TF-IDF questions on mixed data.
Phase rule (SURVIVE_LEN) matches gavel_snake.phase_of exactly.

Teachers generate labels offline. Live play must go through a trained
Gavel question (server/onnx brain), never this script's output.
"""
import json
import random
import sys
from collections import Counter

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from gavel_snake import (DELTA, MOVES, Game, brain_text, choice_suffix,
                         ensure_obstacles, flood_info, food_step, legacy_text,
                         phase_of, pick_move, teacher)

OUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake2.jsonl"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
TEACHER = sys.argv[4] if len(sys.argv) > 4 else "mixed"
N_OBS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TEXTFMT = sys.argv[6] if len(sys.argv) > 6 else "legacy"
WIDTH = int(sys.argv[7]) if len(sys.argv) > 7 else 12
HEIGHT = int(sys.argv[8]) if len(sys.argv) > 8 else 12
if TEACHER not in ("mixed", "immortal", "choice", "pillars"):
    raise SystemExit("teacher must be 'mixed', 'immortal', 'choice' or 'pillars'")
if TEACHER == "pillars" and N_OBS <= 0:
    raise SystemExit("pillars teacher needs obstacles > 0")
if TEXTFMT not in ("legacy", "brain"):
    raise SystemExit("textfmt must be 'legacy' or 'brain'")
if TEXTFMT == "brain" and TEACHER != "pillars":
    raise SystemExit("textfmt=brain only pairs with the pillars teacher")
SURVIVE_LEN = 20  # must match gavel_snake.phase_of (serve recomputes it)

if TEACHER in ("immortal", "choice"):
    from immortal.policies import POLICIES as _IMMORTAL
    _IMMORTAL_FACTORY = _IMMORTAL["jumps-bfs"]
_router = None
if TEACHER == "pillars":
    from gavel_snake import RouterBrain
    _router = RouterBrain()


def state_text(game, heading, phase):
    """Legacy colon format -- TRANSFORMER lineage (matches brain_text).
    Do not use for TF-IDF questions (see contract above)."""
    hx, hy = game.snake[0]
    body = list(game.snake)
    occ = set(body)
    tail = body[-1]
    parts = ["phase %s head(%d,%d) apple(%d,%d) len%d heading %s" %
             (phase, hx, hy, game.apple[0], game.apple[1], len(body), heading)]
    mv = []
    for m in MOVES:
        dx, dy = DELTA[m]
        t = (hx + dx, hy + dy)
        if not (0 <= t[0] < game.w and 0 <= t[1] < game.h):
            st = "wall"
        elif t in occ and t != tail:
            st = "body"
        elif t == tail:
            st = "tail"
        else:
            st = "free"
        fd = abs(game.apple[0] - hx - dx) + abs(game.apple[1] - hy - dy)
        legal, space, _ = flood_info(body, game.apple, game.w, game.h, m,
                                     getattr(game, "obstacles", None))
        mv.append("%s:%s f%d s%d" % (m, st, fd, space if legal else 0))
    parts.append(" | ".join(mv))
    return " || ".join(parts), body


def main():
    rows = []
    phase_count = Counter()
    rng = random.Random(20260922)
    seed = 7
    while len(rows) < TOTAL and seed < 7 + SEEDS:
        game = Game(WIDTH, HEIGHT, seed)
        if N_OBS > 0:
            # Pillar layout per game seed: diverse boards across the dataset,
            # deterministic within a seed. Never on the starting snake.
            ensure_obstacles(game, N_OBS, seed)
        heading = "right"
        from gavel_snake import head_dir
        heading = head_dir(game)
        imm = None
        if TEACHER in ("immortal", "choice"):
            imm = _IMMORTAL_FACTORY()
            imm.reset(game)
        guard = 0
        while len(rows) < TOTAL and guard < 1200:
            hx, hy = game.snake[0]
            danger = game.danger()
            ax, ay = game.apple[0] - hx, game.apple[1] - hy
            # Regime: survive when long, cramped, or food cut off.
            # Obstacles threaded (serve computes phase the same way).
            _go = getattr(game, "obstacles", None)
            _sn = list(game.snake)
            roomy = any(flood_info(_sn, game.apple, game.w, game.h, m,
                                   _go)[1]
                        >= 2 * len(game.snake)
                        for m in MOVES if not danger.get(m))
            food_ok = any(flood_info(_sn, game.apple, game.w, game.h, m,
                                     _go)[2]
                          for m in MOVES if not danger.get(m))
            survive = (len(game.snake) >= SURVIVE_LEN or not roomy or not food_ok)
            if TEACHER == "pillars":
                # Router labels on pillar boards; options list the danger-free
                # moves (same rule as serve time). The tail invariant inside
                # pick_move keeps labels death-free; assert proves no doomed
                # label ever enters the data.
                phase = "survive" if survive else "hunt"
                label, _, _ = _router.decide_game(game)
                safe = [m for m in MOVES if not danger.get(m)]
                if not safe:
                    break  # fully surrounded: no statable choice, end game data
                assert label in safe, (label, safe, seed)
            elif TEACHER in ("immortal", "choice"):
                # 0-death labels from the Hamiltonian teacher; the phase prefix
                # stays honest (same rule as mixed) so text format is unchanged.
                phase = "survive" if survive else "hunt"
                safe = None
                if TEACHER == "choice":
                    # Read the safe set BEFORE decide() advances the guard --
                    # serve time reads it pre-execution too, so the same state
                    # yields the same suffix in both places.
                    hx0, hy0 = game.snake[0]
                    safe = []
                    for c in imm.candidates(game):
                        m = imm._move_of.get((c[0] - hx0, c[1] - hy0))
                        if m is not None and m not in safe:
                            safe.append(m)
                label = imm.decide(game)
            elif not survive:
                phase = "hunt"
                step = food_step(list(game.snake), game.apple, game.w, game.h)
                if step is None or danger.get(step):
                    step = teacher(ax, ay, danger)
                    if danger.get(step):
                        step = pick_move(game, ax, ay, danger,
                                         teacher(ax, ay, danger), False)[0]
                label = step
            else:
                phase = "survive"
                label, _ = pick_move(game, ax, ay, danger,
                                     teacher(ax, ay, danger), False, False, True)
            if TEACHER in ("immortal", "choice", "pillars"):
                # Serve-parity: TF-IDF questions are served with legacy_text,
                # transformer questions with brain_text (textfmt=brain).
                # The embedded phase must equal the loop phase (same rule).
                assert phase_of(game, danger) == phase, (phase, seed)
                if TEXTFMT == "brain":
                    text = brain_text(game, heading, True)
                else:
                    text = legacy_text(game, heading)
            else:
                text, _ = state_text(game, heading, phase)
            if TEACHER == "pillars":
                if TEXTFMT == "legacy":
                    text += choice_suffix(safe)
            elif TEACHER == "choice":
                # The label must be a member of the listed safe set (assert
                # proves candidates()/decide() consistency on empty boards --
                # detours never trigger there).
                assert label in safe, (label, safe)
                text += choice_suffix(safe)
            rows.append({"text": text, "label": label, "phase": phase})
            phase_count[phase] += 1
            game.step(label)
            heading = label
            guard += 1
            if not game.alive or game.won or game.apple is None:
                break
        seed += 1

    rng.shuffle(rows)
    rows = rows[:TOTAL]
    n_val = max(500, TOTAL // 10)
    val, train = rows[:n_val], rows[n_val:]
    base = OUT.rsplit(".jsonl", 1)[0]
    json.dump(train, open(base + "_train.jsonl", "w"))
    json.dump(val, open(base + "_val.jsonl", "w"))
    print(json.dumps({
        "teacher": TEACHER,
        "train": len(train), "val": len(val),
        "phases": dict(phase_count),
        "train_balance": dict(Counter(r["label"] for r in train)),
        "seeds_used": seed - 7,
    }, indent=1))


if __name__ == "__main__":
    main()
