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
  python tiny_data2.py [out_base] [total] [seeds]
Defaults: tiny_snake2.jsonl 30000 200
Writes <base>_train.jsonl / <base>_val.jsonl. Stdlib + gavel_snake only.
"""
import json
import random
import sys
from collections import Counter

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from gavel_snake import (DELTA, MOVES, Game, flood_info, food_step,
                         pick_move, teacher)

OUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake2.jsonl"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
SURVIVE_LEN = 26


def state_text(game, heading, phase):
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
        legal, space, _ = flood_info(body, game.apple, game.w, game.h, m)
        mv.append("%s:%s f%d s%d" % (m, st, fd, space if legal else 0))
    parts.append(" | ".join(mv))
    return " || ".join(parts), body


rows = []
phase_count = Counter()
rng = random.Random(20260922)
seed = 7
while len(rows) < TOTAL and seed < 7 + SEEDS:
    game = Game(12, 12, seed)
    heading = "right"
    from gavel_snake import head_dir
    heading = head_dir(game)
    guard = 0
    while len(rows) < TOTAL and guard < 1200:
        hx, hy = game.snake[0]
        danger = game.danger()
        ax, ay = game.apple[0] - hx, game.apple[1] - hy
        # Regime: survive when long, cramped, or food cut off.
        roomy = any(flood_info(list(game.snake), game.apple, 12, 12, m)[1]
                    >= 2 * len(game.snake)
                    for m in MOVES if not danger.get(m))
        food_ok = any(flood_info(list(game.snake), game.apple, 12, 12, m)[2]
                      for m in MOVES if not danger.get(m))
        survive = (len(game.snake) >= SURVIVE_LEN or not roomy or not food_ok)
        if not survive:
            phase = "hunt"
            step = food_step(list(game.snake), game.apple, 12, 12)
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
        text, _ = state_text(game, heading, phase)
        rows.append({"text": text, "label": label, "phase": phase})
        phase_count[phase] += 1
        game.step(label)
        heading = label
        guard += 1
        if not game.alive:
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
    "train": len(train), "val": len(val),
    "phases": dict(phase_count),
    "train_balance": dict(Counter(r["label"] for r in train)),
    "seeds_used": seed - 7,
}, indent=1))
