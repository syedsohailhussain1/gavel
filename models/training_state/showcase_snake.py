#!/usr/bin/env python3
"""Tiny-model speed showcase: teach Gavel to play Snake, then time it.

This is the Gavel analogue of the "small local model plays Snake" demo:
a tiny supervised question learns state -> move decisions, then makes
hundreds of sequential game decisions through the live Gavel server.

What it measures:
  - train_seconds: /define + /train wall time for a tiny logistic model
  - calibrate_seconds: /calibrate wall time
  - valid_accuracy: agreement with the synthetic teacher on held-out states
  - decisions_per_second: timed /ask calls during a live Snake simulation
  - survival_steps/score: how long the tiny model stays alive

Usage:
  python showcase_snake.py [train_n] [val_n] [test_n] [sim_steps] [grid]
Defaults: 5000 800 500 500 12. Needs gavel-server on :7575. Stdlib only.
"""
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7575"
QUESTION = "snake_move"
MOVES = ("up", "down", "left", "right")
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

TRAIN_N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
VAL_N = int(sys.argv[2]) if len(sys.argv) > 2 else 800
TEST_N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
SIM_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
GRID = int(sys.argv[5]) if len(sys.argv) > 5 else 12


def api(path, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sign(v):
    return "neg" if v < 0 else ("pos" if v > 0 else "zero")


def teacher(ax, ay, danger):
    """Synthetic optimal-ish teacher: safe move minimizing distance to apple."""
    best, best_d = None, None
    for m in MOVES:
        if danger.get(m):
            continue
        dx, dy = DELTA[m]
        d = abs(ax - dx) + abs(ay - dy)
        if best_d is None or d < best_d:
            best, best_d = m, d
    if best is None:
        return "up"
    return best


def tok(v):
    return ("n%d" % -v) if v < 0 else ("p%d" % v if v > 0 else "zero")


def encode(ax, ay, danger, heading):
    parts = ["ax%s" % tok(ax), "ayn%s" % tok(ay)]
    for m in MOVES:
        if danger[m]:
            parts.append(m + "_dead")
            continue
        dx, dy = DELTA[m]
        d = abs(ax - dx) + abs(ay - dy)
        bucket = "02" if d <= 2 else ("35" if d <= 5 else "6p")
        parts.append("%s_d%s" % (m, bucket))
    parts.append("heading_%s" % heading)
    return " ".join(parts)


def random_state(rng):
    ax = rng.randint(-11, 11)
    ay = rng.randint(-11, 11)
    if ax == 0 and ay == 0:
        ax = 1
    danger = {m: rng.random() < 0.25 for m in MOVES}
    if all(danger.values()):
        danger[rng.choice(MOVES)] = False
    return ax, ay, danger, rng.choice(MOVES)


def make_examples(rng, n):
    out = []
    for _ in range(n):
        s = random_state(rng)
        ax, ay, danger, heading = s
        out.append((encode(ax, ay, danger, heading),
                    teacher(ax, ay, danger)))
    return out


def ask_move(text):
    r = api("/ask", {"question": QUESTION, "input": text}, timeout=30)
    lab = (r.get("output") or {}).get("label")
    return lab if lab in MOVES else None


rng = random.Random(20260921)
train = make_examples(rng, TRAIN_N)
val = make_examples(rng, VAL_N)
test = make_examples(random.Random(4242), TEST_N)

t0 = time.perf_counter()
api("/define", {"name": QUESTION, "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"}})
api("/train", {"question": QUESTION, "examples": [
    {"input": t, "label": l} for t, l in train]})
train_s = time.perf_counter() - t0

t0 = time.perf_counter()
api("/calibrate", {"question": QUESTION, "validation": [
    {"input": t, "label": l} for t, l in val]})
cal_s = time.perf_counter() - t0
metrics = api(f"/metrics?question={QUESTION}")

ok = 0
for t, l in test:
    if ask_move(t) == l:
        ok += 1

# Live Snake simulation driven by Gavel decisions.
cx, cy = GRID // 2, GRID // 2
snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
heading = "right"
apple = (cx + 3, cy)
srng = random.Random(7)
score = steps = fallbacks = 0
ask_s = 0.0
alive = True
while alive and steps < SIM_STEPS:
    hx, hy = snake[0]
    ax, ay = apple[0] - hx, apple[1] - hy
    body = set(snake)
    danger = {}
    for m, (dx, dy) in DELTA.items():
        nx, ny = hx + dx, hy + dy
        danger[m] = not (0 <= nx < GRID and 0 <= ny < GRID) or (nx, ny) in body
    text = encode(ax, ay, danger, heading)
    t0 = time.perf_counter()
    move = ask_move(text)
    ask_s += time.perf_counter() - t0
    if move is None:
        move = teacher(ax, ay, danger)
        fallbacks += 1
    heading = move
    dx, dy = DELTA[move]
    head = (hx + dx, hy + dy)
    steps += 1
    if not (0 <= head[0] < GRID and 0 <= head[1] < GRID) or head in body:
        alive = False
        break
    snake.insert(0, head)
    if head == apple:
        score += 1
        free = [(x, y) for x in range(GRID) for y in range(GRID)
                if (x, y) not in set(snake)]
        if not free:
            break
        apple = srng.choice(free)
    else:
        snake.pop()

print(json.dumps({
    "question": QUESTION,
    "train_n": TRAIN_N,
    "train_seconds": round(train_s, 3),
    "calibrate_seconds": round(cal_s, 3),
    "valid_accuracy": round(ok / max(len(test), 1), 4),
    "sim_decisions": steps,
    "sim_ask_seconds": round(ask_s, 3),
    "decisions_per_second": round(steps / max(ask_s, 1e-9), 1),
    "survival_steps": steps,
    "score": score,
    "alive_at_cap": alive,
    "fallbacks": fallbacks,
    "temperature": metrics.get("temperature"),
    "ece": [metrics.get("ece_before"), metrics.get("ece_after")],
}, indent=1))
