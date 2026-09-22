#!/usr/bin/env python3
"""Generate tiny-model training data from real Snake rollouts.

Each example: a rich text state (head/apple/body facts + per-move safety,
food distance, and flood-fill free space) labeled by the guarded policy
(teacher + lookahead + tail-chase) — i.e. near-optimal play including
trap avoidance, which is exactly what the linear model couldn't learn.

Usage:
  python tiny_data.py [out.jsonl] [total_examples] [seeds]
Defaults: tiny_snake.jsonl 30000 40
Stdlib only (+ imports game logic from gavel_snake.py). No server needed.
"""
import json
import random
import sys
from collections import Counter

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from gavel_snake import DELTA, MOVES, Game, flood_info, pick_move, teacher

OUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake.jsonl"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 40


def state_text(game, heading):
    hx, hy = game.snake[0]
    ax, ay = game.apple[0] - hx, game.apple[1] - hy
    body = list(game.snake)
    occ = set(body)
    tail = body[-1]
    parts = ["head(%d,%d) apple(%d,%d) len%d heading %s" %
             (hx, hy, game.apple[0], game.apple[1], len(body), heading)]
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
        fd = abs(ax - dx) + abs(ay - dy)
        legal, space, _ = flood_info(body, game.apple, game.w, game.h, m)
        mv.append("%s:%s f%d s%d" % (m, st, fd, space if legal else 0))
    parts.append(" | ".join(mv))
    return " || ".join(parts), ax, ay


rows, agree = [], 0
rng = random.Random(20260921)
seed = 7
while len(rows) < TOTAL and seed < 7 + SEEDS * 4:
    game = Game(12, 12, seed)
    heading = "right"
    guard = 0
    while len(rows) < TOTAL and guard < 1500:
        hx, hy = game.snake[0]
        text, ax, ay = state_text(game, heading)
        danger = game.danger()
        raw = teacher(ax, ay, danger)
        final, _ = pick_move(game, ax, ay, danger, raw, False)
        agree += (final == raw)
        rows.append({"text": text, "label": final, "teacher": raw})
        game.step(final)
        heading = final
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
    "train_balance": dict(Counter(r["label"] for r in train)),
    "teacher_agreement": round(agree / max(len(rows), 1), 4),
    "seeds_used": seed - 7,
}, indent=1))
