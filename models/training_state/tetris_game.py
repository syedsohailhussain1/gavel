#!/usr/bin/env python3
"""Tetris engine + Dellacherie heuristic + pairwise choice data.

Engine: 10x20 board (bitmask rows), 7 tetrominoes, SRS-like rotations,
7-bag seeded randomizer, guideline-ish scoring. Deterministic per seed.

Data: heuristic self-play; each piece contributes one A-vs-B choice between
its two strongest legal landings (their pattern: one typed call per piece).
Near-ties skipped for clean labels. Writes JSON arrays for legacy_tfidf.py.

Usage:
  python tetris_game.py [out_base] [pieces] [seeds]
Defaults: tetris_pairs.jsonl 20000 40
"""
import json
import random
import sys

W, H = 10, 20
OUT = "tetris_pairs.jsonl"
PIECES = 20000
SEEDS = 40

BASE = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}


def _rot(cells):
    return sorted({(3 - y, x) for x, y in cells})


def _rots(name):
    out, cur = [], BASE[name]
    for _ in range(4):
        key = tuple(cur)
        if key not in [tuple(r) for r in out]:
            xs = [x for x, _ in cur]
            ys = [y for _, y in cur]
            norm = sorted((x - min(xs), y - min(ys)) for x, y in cur)
            out.append(norm)
        cur = _rot(cur)
    return out


ROTS = {n: _rots(n) for n in BASE}
FULL = (1 << W) - 1


class Board:
    def __init__(self):
        self.rows = [0] * H
        self.score = self.lines = self.pieces = 0
        self.alive = True

    def fits(self, cells, ox, oy):
        for x, y in cells:
            xx, yy = ox + x, oy + y
            if not (0 <= xx < W) or yy >= H:
                return False
            if yy >= 0 and (self.rows[yy] >> xx) & 1:
                return False
        return True

    def drop_y(self, cells, ox):
        y = -4
        while self.fits(cells, ox, y + 1):
            y += 1
        return y

    def landings(self, piece):
        out = []
        for ri, cells in enumerate(ROTS[piece]):
            maxx = max(x for x, _ in cells)
            for ox in range(W - maxx):
                y = self.drop_y(cells, ox)
                if self.fits(cells, ox, y):
                    out.append((ri, ox, y, cells))
        return out

    def apply(self, cells, ox, oy):
        for x, y in cells:
            yy = oy + y
            if yy < 0:
                self.alive = False
                return 0
            self.rows[yy] |= 1 << (ox + x)
        cleared = [r for r in self.rows if r == FULL]
        n = len(cleared)
        if n:
            self.rows = [r for r in self.rows if r != FULL]
            while len(self.rows) < H:
                self.rows.insert(0, 0)
        self.lines += n
        self.pieces += 1
        self.score += [0, 100, 300, 500, 800][n] if n <= 4 else 1200
        return n

    # ---- Dellacherie features ----
    def heights(self):
        h = []
        for x in range(W):
            c = 0
            for y in range(H):
                if (self.rows[y] >> x) & 1:
                    c = H - y
                    break
            h.append(c)
        return h

    def holes(self):
        n = 0
        for x in range(W):
            block = False
            for y in range(H):
                if (self.rows[y] >> x) & 1:
                    block = True
                elif block:
                    n += 1
        return n

    def features(self):
        h = self.heights()
        agg = sum(h)
        bump = sum(abs(a - b) for a, b in zip(h, h[1:]))
        wells = sum(1 for i in range(W)
                    if (h[i - 1] if i else 99) > h[i] < (h[i + 1] if i + 1 < W else 99))
        return {"agg": agg, "bump": bump, "holes": self.holes(), "wells": wells,
                "maxh": max(h)}

    @staticmethod
    def score_fit(f, lines):
        return (-0.51 * f["agg"] + 0.76 * lines * 10
                - 0.36 * f["holes"] * 10 - 0.18 * f["bump"])


def _clone(rows):
    b = Board()
    b.rows = list(rows)
    return b


def landing_features(board, piece):
    """All legal landings with heuristic-scored features (no mutation)."""
    out = []
    for ri, ox, y, cells in board.landings(piece):
        nb = _clone(board.rows)
        ok = True
        for x, dy in cells:
            yy = y + dy
            if yy < 0:
                ok = False
                break
            nb.rows[yy] |= 1 << (ox + x)
        if not ok:
            continue
        n = sum(1 for r in nb.rows if r == FULL)
        f = nb.features()
        out.append({"ri": ri, "ox": ox, "score": Board.score_fit(f, n),
                    "lines": n, "agg": f["agg"], "holes": f["holes"],
                    "bump": f["bump"], "maxh": f["maxh"],
                    "cells": cells, "y": y})
    return out


def fmt(l, side):
    # Side-bound tokens (ah11 vs bh11): the BOW engine must see WHICH side
    # each fact belongs to, or comparison is unlearnable (measured 61%).
    return (f"{side}h{l['maxh']} {side}lines{l['lines']} {side}holes{l['holes']} "
            f"{side}bump{l['bump']} {side}agg{l['agg']}")


def bag(rng):
    names = list(BASE)
    while True:
        rng.shuffle(names)
        for n in names:
            yield n


if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else OUT
    PIECES = int(sys.argv[2]) if len(sys.argv) > 2 else PIECES
    SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else SEEDS
    rows = []
    seed = 11
    while len(rows) < PIECES and seed < 11 + SEEDS * 8:
        b = Board()
        rng = random.Random(seed)
        gen = bag(rng)
        guard = 0
        while len(rows) < PIECES and guard < 3000 and b.alive:
            p = next(gen)
            cands = sorted(landing_features(b, p),
                           key=lambda l: l["score"], reverse=True)
            if len(cands) >= 2:
                a, c = cands[0], cands[1]
                if a["score"] - c["score"] >= 3.0:  # clean margin, no ties
                    flip = rng.random() < 0.5
                    x, y = (c, a) if flip else (a, c)
                    rows.append({"text": f"A {fmt(x, 'a')} || B {fmt(y, 'b')}",
                                 "label": "B" if flip else "A"})
            best = cands[0] if cands else None
            if best is None:
                break
            b.apply(best["cells"], best["ox"], best["y"])
            guard += 1
        seed += 1

    rng = random.Random(20260930)
    rng.shuffle(rows)
    rows = rows[:PIECES]
    n_val = max(500, PIECES // 10)
    base = OUT.rsplit(".jsonl", 1)[0]
    json.dump(rows[n_val:], open(base + "_train.jsonl", "w"))
    json.dump(rows[:n_val], open(base + "_val.jsonl", "w"))
    from collections import Counter
    print(f"train={len(rows)-n_val} val={n_val} "
          f"labels={dict(Counter(r['label'] for r in rows))}", flush=True)
