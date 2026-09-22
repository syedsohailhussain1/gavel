#!/usr/bin/env python3
"""Survival test: tiny ONNX model plays Snake raw, shield as backstop.

Reuses Game + guarded pick_move from gavel_snake; the model's raw proposal
replaces the Gavel server call. Reports score/deaths/veto-rate/decisions-sec
per seed — the veto rate measures what the model hasn't learned yet.

Usage:
  python tiny_play.py [onnx] [tok_dir] [steps_per_seed] [seeds] [label_order]
Defaults: tiny_snake_gtiny_fp32.onnx tiny_snake_gtiny 1000 5 up,down,left,right
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from gavel_snake import DELTA, MOVES, Game, flood_info, pick_move

ONNX = sys.argv[1] if len(sys.argv) > 1 else "tiny_snake_gtiny_fp32.onnx"
TOK = sys.argv[2] if len(sys.argv) > 2 else "tiny_snake_gtiny"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TOK)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 8
sess = ort.InferenceSession(ONNX, sess_options=opts, providers=["CPUExecutionProvider"])
sess.run(None, {"input_ids": np.ones((1, 32), dtype=np.int64),
                "attention_mask": np.ones((1, 32), dtype=np.int64)})


def state_text(game, heading):
    hx, hy = game.snake[0]
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
        fd = abs(game.apple[0] - hx - dx) + abs(game.apple[1] - hy - dy)
        legal, space, _ = flood_info(body, game.apple, game.w, game.h, m)
        mv.append("%s:%s f%d s%d" % (m, st, fd, space if legal else 0))
    parts.append(" | ".join(mv))
    return " || ".join(parts)


tot = {"steps": 0, "score": 0, "deaths": 0, "veto": 0, "sec": 0.0}
for s in range(7, 7 + SEEDS):
    game = Game(12, 12, s)
    heading = "right"
    steps = 0
    t0 = time.perf_counter()
    while steps < STEPS:
        if game.won:
            print(f"seed={s} BOARD CLEAR score={game.score} [{time.perf_counter()-t0:.1f}s]",
                  flush=True)
            break
        hx, hy = game.snake[0]
        text = state_text(game, heading)
        e = tok(text, return_tensors="np", truncation=True, max_length=96)
        logits = sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                                 "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
        raw = MOVES[int(logits.argmax())]
        ax, ay = game.apple[0] - hx, game.apple[1] - hy
        final, intr = pick_move(game, ax, ay, game.danger(), raw, False)
        tot["veto"] += intr
        game.step(final)
        heading = final
        steps += 1
        if not game.alive:
            tot["deaths"] += 1
            break
    el = time.perf_counter() - t0
    tot["steps"] += steps
    tot["score"] += game.score
    tot["sec"] += el
    print(f"seed={s} steps={steps} score={game.score} "
          f"alive={game.alive} [{el:.1f}s]", flush=True)

print(json.dumps({
    "onnx": ONNX, "seeds": SEEDS,
    "avg_score": round(tot["score"] / SEEDS, 1),
    "deaths": tot["deaths"], "total_steps": tot["steps"],
    "veto_rate": round(tot["veto"] / max(tot["steps"], 1), 4),
    "decisions_per_sec": round(tot["steps"] / max(tot["sec"], 1e-9), 1),
}, indent=1))
