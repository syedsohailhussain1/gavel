#!/usr/bin/env python3
"""DAgger harvest for Tetris: play with the MODEL deciding, label every
visited pair with the heuristic. Fixes closed-loop drift — the model trains
on states it actually visits, not just states good play visits.

Usage:
  python tetris_dagger.py [out] [pieces] [seeds] [question]
Defaults: tetris_dagger.jsonl 3000 12 tetris_pick
Needs gavel-server up with the question defined. Stdlib only.
"""
import json
import random
import sys
import urllib.request

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from tetris_game import Board, bag, fmt, landing_features

BASE = "http://127.0.0.1:7575"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tetris_dagger.jsonl"
PIECES = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
QUESTION = sys.argv[4] if len(sys.argv) > 4 else "tetris_pick"

import http.client as _hc
from urllib.parse import urlparse as _up

_CONN = None


def decide(text):
    global _CONN
    body = json.dumps({"question": QUESTION, "input": text})
    try:
        if _CONN is None:
            u = _up(BASE)
            _CONN = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=60)
        _CONN.request("POST", "/ask", body=body,
                      headers={"Content-Type": "application/json"})
        out = json.loads(_CONN.getresponse().read())
    except Exception:
        return None
    lab = (out.get("output") or {}).get("label")
    return lab if lab in ("A", "B") else None


rows = []
seed = 101
while len(rows) < PIECES and seed < 101 + SEEDS * 10:
    b = Board()
    rng = random.Random(seed)
    gen = bag(rng)
    guard = 0
    while len(rows) < PIECES and guard < 1500 and b.alive:
        p = next(gen)
        cands = sorted(landing_features(b, p),
                       key=lambda l: l["score"], reverse=True)
        if len(cands) < 2:
            break
        a = cands[0]
        c = next((l for l in cands[1:]
                  if a["score"] - l["score"] >= 3.0), None)
        if c is None:
            best = a
        else:
            flip = rng.random() < 0.5
            x, y = (c, a) if flip else (a, c)
            # Record the VISITED state with the HEURISTIC label (DAgger):
            # label = the side actually holding the heuristic-best landing.
            rows.append({"text": f"A {fmt(x, 'a')} || B {fmt(y, 'b')}",
                         "label": "B" if flip else "A"})
            pick = decide(rows[-1]["text"])
            if pick is None:
                best = a
            else:
                best = x if pick == "A" else y
        b.apply(best["cells"], best["ox"], best["y"])
        guard += 1
    seed += 1

json.dump(rows[:PIECES], open(OUT, "w"))
from collections import Counter
print(f"dagger={len(rows[:PIECES])} labels={dict(Counter(r['label'] for r in rows[:PIECES]))}",
      flush=True)
