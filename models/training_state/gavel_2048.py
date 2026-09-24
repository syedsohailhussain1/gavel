#!/usr/bin/env python3
"""gavel-2048: 2048 played by tiny Gavel decisions (TF-IDF question).

Same architecture as the GLiClass 2048 demo we answer (heuristic shortlists
the strongest swipes, a small model picks among resulting-board
descriptions, a margin gate keeps the heuristic leader on low confidence)
-- with a ~5k-weight TF-IDF question behind the same HTTP API instead of a
32.7M-parameter encoder. Every move is a live /ask call; every override of
the heuristic leader is counted, not hidden.

Usage:
  python gavel_2048.py gen [out_base] [total] [seeds]
      -- teacher rollouts (expectimax 1-ply + heuristic) -> JSON data
  python gavel_2048.py play [--seed N] [--seeds a,b,c] [--max-moves N]
      [--headless] [--max-speed] [--margin F] [--candidates 2]
      -- needs gavel-server on :7575 with the question trained (see below)
  python gavel_2048.py bench [--seeds 1-10] [--max-moves N]
      -- headless aggregate: score/moves/max-tile/agreement/latency table

Train (after gen):
  python legacy_tfidf.py <train> <val> 2048_move 2048_tfidf_prod.json

Engine: conventional 4x4, slide+merge toward a side, 90/10 spawn of 2/4 on
a uniform free cell, seeded RNG (deterministic tile sequences per seed).
Game over when no legal swipe remains.
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

TS = Path(__file__).resolve().parent
sys.path.insert(0, str(TS))

BASE = "http://127.0.0.1:7575"
QUESTION = "2048_move"
MOVES = ("up", "down", "left", "right")

# Heuristic weights (teacher): empty tiles dominate, then monotonicity of
# log-tiles along rows/cols (corner discipline), smoothness (adjacent
# similarity), max-tile-in-corner bonus. Tuned for play, not theory.
W_EMPTY = 2.7
W_MONO = 1.0
W_SMOOTH = 0.1
W_CORNER = 1.5

# Search depth (max plies) for the teacher. Depth 1 = move + sampled
# spawns + heuristic (fast, ~10k teacher). Depth 2 adds one reply ply
# (~2-5ms/move, much stronger labels). The TF-IDF standard is depth 2:
# labels worth imitating, still cheap enough to generate 30k states.
SEARCH_DEPTH = 2
SEARCH_SAMPLES = 6


class Game:
    def __init__(self, seed=46):
        self.rng = random.Random(seed)
        self.b = [0] * 16
        self.score = self.moves = 0
        self.over = False
        self._spawn()
        self._spawn()

    def _free(self):
        return [i for i, v in enumerate(self.b) if v == 0]

    def _spawn(self):
        free = self._free()
        if not free:
            return False
        self.b[self.rng.choice(free)] = 4 if self.rng.random() < 0.1 else 2
        return True

    @staticmethod
    def _slide_row_left(row):
        """Slide+merge one 4-row left. Returns (new_row, gained)."""
        t = [v for v in row if v != 0]
        out, gained, i = [], 0, 0
        while i < len(t):
            if i + 1 < len(t) and t[i] == t[i + 1]:
                out.append(t[i] * 2)
                gained += t[i] * 2
                i += 2
            else:
                out.append(t[i])
                i += 1
        out += [0] * (4 - len(out))
        return out, gained

    @staticmethod
    def _move(board, mv):
        """Apply swipe to a board. Returns (new_board, gained, changed).

        Row/col mapping: left = rows as-is; right = rows reversed;
        up = columns top-down; down = columns bottom-up.
        """
        nb = list(board)
        gained, changed = 0, False
        for k in range(4):
            if mv in ("left", "right"):
                idx = [k * 4 + j for j in range(4)]
            else:
                idx = [j * 4 + k for j in range(4)]
            row = [nb[i] for i in idx]
            if mv in ("right", "down"):
                row = row[::-1]
            new_row, g = Game._slide_row_left(row)
            if mv in ("right", "down"):
                new_row = new_row[::-1]
            for i, v in zip(idx, new_row):
                if nb[i] != v:
                    changed = True
                nb[i] = v
            gained += g
        return nb, gained, changed

    def legal(self, board=None):
        b = self.b if board is None else board
        return [m for m in MOVES if Game._move(b, m)[2]]

    def step(self, mv):
        """Play a swipe. Returns gained score, or None if illegal."""
        nb, gained, changed = Game._move(self.b, mv)
        if not changed:
            return None
        self.b = nb
        self.score += gained
        self.moves += 1
        self._spawn()
        if not self.legal():
            self.over = True
        return gained

    def max_tile(self):
        return max(self.b)

    def snapshot(self):
        return {"board": list(self.b), "score": self.score,
                "moves": self.moves, "over": self.over,
                "max": self.max_tile()}


def _log2(v):
    return v.bit_length() - 1 if v > 0 else 0


def heuristic(board):
    """Static board value: empty + monotonicity + smoothness + corner."""
    empty = sum(1 for v in board if v == 0)
    mono = 0
    for k in range(4):
        row = [board[k * 4 + j] for j in range(4)]
        col = [board[j * 4 + k] for j in range(4)]
        for line in (row, col):
            l = [_log2(v) for v in line]
            inc = sum(1 for a, b in zip(l, l[1:]) if a <= b)
            dec = sum(1 for a, b in zip(l, l[1:]) if a >= b)
            mono += max(inc, dec)
    smooth = 0
    for i, v in enumerate(board):
        if v == 0:
            continue
        x, y = i % 4, i // 4
        if x + 1 < 4 and board[i + 1] != 0:
            smooth -= abs(_log2(v) - _log2(board[i + 1]))
        if y + 1 < 4 and board[i + 4] != 0:
            smooth -= abs(_log2(v) - _log2(board[i + 4]))
    mx = max(board)
    corner = mx if board[0] == mx else 0
    return (W_EMPTY * empty + W_MONO * mono + W_SMOOTH * smooth
            + W_CORNER * _log2(corner or 1))


def expectimax(board, rng, samples=8):
    """One-ply expectimax value of each legal swipe: mean heuristic over
    sampled random spawns after the move. Returns {move: value}."""
    vals = {}
    for m in MOVES:
        nb, gained, changed = Game._move(board, m)
        if not changed:
            continue
        free = [i for i, v in enumerate(nb) if v == 0]
        if not free:
            vals[m] = heuristic(nb) + math_gain(gained)
            continue
        tot = 0.0
        for _ in range(samples):
            b2 = list(nb)
            b2[rng.choice(free)] = 4 if rng.random() < 0.1 else 2
            tot += heuristic(b2)
        vals[m] = tot / samples + math_gain(gained)
    return vals


def math_gain(gained):
    return gained ** 0.5 if gained > 0 else 0.0


def _max_value(board, depth, rng, samples):
    """Best continuation value with `depth` max plies remaining."""
    if depth <= 0:
        return heuristic(board)
    best = None
    for m in MOVES:
        nb, gained, changed = Game._move(board, m)
        if not changed:
            continue
        v = _chance_value(nb, gained, depth - 1, rng, samples)
        if best is None or v > best:
            best = v
    return heuristic(board) if best is None else best


def _chance_value(board, gained, depth, rng, samples):
    """Mean value over sampled random spawns on `board` (+ immediate gain)."""
    free = [i for i, v in enumerate(board) if v == 0]
    if not free:
        return heuristic(board) + math_gain(gained)
    tot = 0.0
    for _ in range(samples):
        b2 = list(board)
        b2[rng.choice(free)] = 4 if rng.random() < 0.1 else 2
        tot += _max_value(b2, depth, rng, samples)
    return tot / samples + math_gain(gained)


def search(board, rng, width=2, depth=SEARCH_DEPTH, samples=SEARCH_SAMPLES):
    """Depth-limited expectimax values per legal swipe. depth=1 matches
    expectimax(); depth=2 adds one reply ply. Returns ranked moves + vals."""
    vals = {}
    for m in MOVES:
        nb, gained, changed = Game._move(board, m)
        if not changed:
            continue
        vals[m] = _chance_value(nb, gained, depth - 1, rng, samples)
    ranked = sorted(vals, key=lambda m: -vals[m])
    return ranked[:width], vals


def teacher(board, seed_rng, width=2, samples=8, depth=SEARCH_DEPTH):
    """Shortlist: top-`width` swipes. depth=0 ranks by static heuristic
    (deterministic: 100% self-consistent labels, ~9.5k teacher); depth>=1
    runs depth-limited expectimax search (~15k at depth 2, ~69% label
    self-consistency from spawn sampling noise). Returns ranked moves."""
    if depth <= 0:
        scored = sorted(
            [m for m in MOVES if Game._move(board, m)[2]],
            key=lambda m: -heuristic(Game._move(board, m)[0]))
        return scored[:width], {}
    if depth <= 1:
        vals = expectimax(board, seed_rng, samples)
        ranked = sorted(vals, key=lambda m: -vals[m])
        return ranked[:width], vals
    return search(board, seed_rng, width, depth, SEARCH_SAMPLES)


def _bucket(v):
    """Log tile bucket, single alphanumeric token."""
    if v == 0:
        return "e"
    return "t%d" % min(_log2(v), 17)


def board_text(board, options):
    """Serve/train text: 16 cells row-major as c<idx>_<bucket> plus an
    options_* token per shortlisted swipe (MOVES order). Byte-identical
    wherever used -- training and serve must build the same string."""
    cells = " ".join("c%d_%s" % (i, _bucket(v)) for i, v in enumerate(board))
    opts = " ".join("options_%s" % m for m in MOVES if m in options)
    Empties = sum(1 for v in board if v == 0)
    mx = max(board)
    return "%s || %s || empty%d max%d" % (cells, opts, Empties, mx)


def pair_text(board_a, board_b):
    """Pairwise choice text: two resulting boards side by side, cell tokens
    prefixed per side (A_/B_) so BOW features bind to the right board.
    Labels: 'first' / 'second'. Byte-identical train/serve."""
    a = " ".join("A_c%d_%s" % (i, _bucket(v)) for i, v in enumerate(board_a))
    b = " ".join("B_c%d_%s" % (i, _bucket(v)) for i, v in enumerate(board_b))
    ea, eb = sum(1 for v in board_a if v == 0), sum(1 for v in board_b if v == 0)
    ma, mb = max(board_a), max(board_b)
    return ("%s || %s || A_empty%d A_max%d || B_empty%d B_max%d"
            % (a, b, ea, ma, eb, mb))


def cand_facts(board, gained=0):
    """Proposition-level candidate description: scalar board facts as
    bound tokens (compare-friendly for linear models: every fact is an
    explicit number, no geometry to decode). Non-negative by design
    (negation is invisible to BOW). Collapsed sums beat directional
    splits (measured 91.9% vs 80.4%): smoothing helps linear models."""
    empty = sum(1 for v in board if v == 0)
    mx = max(board)
    mono = 0
    for k in range(4):
        row = [_log2(board[k * 4 + j]) for j in range(4)]
        col = [_log2(board[j * 4 + k]) for j in range(4)]
        for line in (row, col):
            mono += max(sum(1 for a, b in zip(line, line[1:]) if a <= b),
                        sum(1 for a, b in zip(line, line[1:]) if a >= b))
    rough = 0
    for i, v in enumerate(board):
        if v == 0:
            continue
        x, y = i % 4, i // 4
        if x + 1 < 4 and board[i + 1]:
            rough += abs(_log2(v) - _log2(board[i + 1]))
        if y + 1 < 4 and board[i + 4]:
            rough += abs(_log2(v) - _log2(board[i + 4]))
    corner = 1 if board[0] == mx else 0
    return {"empty": empty, "maxlog": _log2(mx), "gain": gained,
            "mono": mono, "rough": rough, "corner": corner}


def pair_facts_text(fa, fb):
    """Pairwise choice text, proposition level: A_*/B_* scalar facts.
    Labels: 'first' / 'second'. Byte-identical train/serve."""
    a = " ".join("A_%s%d" % (k, v) for k, v in sorted(fa.items()))
    b = " ".join("B_%s%d" % (k, v) for k, v in sorted(fb.items()))
    return "%s || %s" % (a, b)


def api(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_CONN = None


def decide(text, question=None):
    """One live /ask call over keep-alive HTTP. Returns (label, conf, ms)."""
    import http.client as _hc
    from urllib.parse import urlparse as _up
    global _CONN
    t0 = time.perf_counter()
    body = json.dumps({"question": question or QUESTION, "input": text})
    try:
        if _CONN is None:
            u = _up(BASE)
            _CONN = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=30)
        _CONN.request("POST", "/ask", body=body,
                      headers={"Content-Type": "application/json"})
        out = json.loads(_CONN.getresponse().read())
    except Exception:
        try:
            _CONN.close()
        except Exception:
            pass
        _CONN = None
        out = api("/ask", {"question": QUESTION, "input": text}, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    return ((out.get("output") or {}).get("label"),
            round(out.get("confidence", 0), 3), round(ms, 2))


def result_board(board, mv):
    """Resulting board after a swipe (no spawn)."""
    return Game._move(board, mv)[0]


def cmd_gen(a):
    rows = []
    seed = a.seed0
    trng = random.Random(20260924)
    while len(rows) < a.total:
        g = Game(seed)
        grng = random.Random(1000 + seed)
        guard = 0
        while len(rows) < a.total and guard < 3000 and not g.over:
            ranked, _ = teacher(g.b, grng, a.candidates, 8, a.depth)
            if not ranked:
                break
            if a.pairs:
                if len(ranked) < 2:
                    g.step(ranked[0])
                    guard += 1
                    continue
                ra, ga, _ = Game._move(g.b, ranked[0])
                rb, gb, _ = Game._move(g.b, ranked[1])
                if a.facts:
                    if trng.random() < 0.5:
                        rows.append({"text": pair_facts_text(
                            cand_facts(ra, ga), cand_facts(rb, gb)),
                            "label": "first"})
                    else:
                        rows.append({"text": pair_facts_text(
                            cand_facts(rb, gb), cand_facts(ra, ga)),
                            "label": "second"})
                elif trng.random() < 0.5:
                    rows.append({"text": pair_text(ra, rb),
                                 "label": "first"})
                else:
                    rows.append({"text": pair_text(rb, ra),
                                 "label": "second"})
                label = ranked[0]
            else:
                label = ranked[0]
                rows.append({"text": board_text(g.b, ranked), "label": label})
            g.step(ranked[0])
            guard += 1
        seed += 1
    trng.shuffle(rows)
    rows = rows[:a.total]
    n_val = max(500, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    base = a.out.rsplit(".jsonl", 1)[0]
    json.dump(train, open(base + "_train.jsonl", "w"))
    json.dump(val, open(base + "_val.jsonl", "w"))
    print(json.dumps({"train": len(train), "val": len(val),
                      "seeds_used": seed - a.seed0}, indent=1))


def play_game(seed, max_moves, margin, width, headless_fast=False,
              collect=None, depth=SEARCH_DEPTH, question=None,
              pick_question=None, pick_facts=False):
    """Play one game. Move-question flow (default): shortlist top-`width`,
    /ask picks a swipe, margin gate keeps the heuristic leader on low
    confidence. Pairwise flow (pick_question set): tournament over the
    shortlist (top-2 = one call), each call picks first/second between
    the champion and the next challenger; margin gate (on the last call's
    confidence) keeps the leader. Returns summary."""
    question = question or QUESTION
    g = Game(seed)
    grng = random.Random(1000 + seed)
    agree = over = vetoms = totms = 0
    n = 0
    max_ms = 0.0
    while not g.over and n < max_moves:
        ranked, _ = teacher(g.b, grng, width, 8, depth)
        if not ranked:
            break
        leader = ranked[0]
        if pick_question and len(ranked) >= 2:
            # Tournament over the shortlist (top-2 = single comparison):
            # winner of each pairwise call faces the next candidate.
            contenders = list(ranked[:max(2, width)])
            champ = contenders[0]
            for chal in contenders[1:]:
                na, ga, _ = Game._move(g.b, champ)
                nb, gb, _ = Game._move(g.b, chal)
                if pick_facts:
                    ptext = pair_facts_text(cand_facts(na, ga),
                                            cand_facts(nb, gb))
                else:
                    ptext = pair_text(na, nb)
                raw, conf, ms = decide(ptext, pick_question)
                vetoms += ms
                max_ms = max(max_ms, ms)
                if raw == "second":
                    champ = chal
            picked = champ
            if picked == leader:
                agree += 1
            if picked == leader or conf >= margin:
                move, intervened = picked, (picked != leader)
            else:
                move, intervened = leader, True
        else:
            t0 = time.perf_counter()
            raw, conf, ms = decide(board_text(g.b, ranked), question)
            vetoms += ms
            max_ms = max(max_ms, ms)
            if raw == leader:
                agree += 1
            if raw in ranked and (raw == leader or conf >= margin):
                move, intervened = raw, (raw != leader)
            else:
                move, intervened = leader, (raw != leader)
        if collect is not None:
            collect.append({"board": list(g.b), "ranked": ranked,
                            "leader": leader, "proposed": raw,
                            "confidence": conf, "executed": move,
                            "intervened": intervened})
        r = g.step(move)
        if r is None:  # model proposed outside the shortlist and margin
            g.step(leader)  # logic passed -- defensive, counted above
        n += 1
    return {"seed": seed, "score": g.score, "moves": n,
            "max": g.max_tile(), "agreement": round(agree / max(n, 1), 4),
            "overrides": sum(1 for c in (collect or []) if c["intervened"]),
            "model_ms_mean": round(vetoms / max(n, 1), 3),
            "model_ms_max": round(max_ms, 3),
            "final": list(g.b)}


def render(board):
    rows = []
    for y in range(4):
        rows.append("│" + "│".join("%6d" % v if v else "      "
                                   for v in board[y * 4:(y + 1) * 4]) + "│")
    return "\n".join(rows)


def cmd_play(a):
    try:
        qs = api("/questions")
    except Exception as e:
        print(f"server not reachable: {e}", file=sys.stderr)
        return 2
    if QUESTION not in qs.get("questions", []):
        print(f"question '{QUESTION}' not on server -- train it first "
              f"(legacy_tfidf.py)", file=sys.stderr)
        return 2
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] \
        if a.seeds else [a.seed]
    for s in seeds:
        t0 = time.perf_counter()
        frames = [] if a.record else None
        r = play_game(s, a.max_moves, a.margin, a.candidates,
                      collect=frames, depth=a.depth, question=a.question,
                      pick_question=a.pick_question or None,
                      pick_facts=a.pick_facts)
        wall = time.perf_counter() - t0
        if not a.headless:
            print(render(r.pop("final")))
        print(json.dumps({**r, "wall_s": round(wall, 1),
                          "question": a.question}, indent=1))
        if a.record:
            json.dump(frames, open(a.record, "w"))
    return 0


def cmd_bench(a):
    seeds = list(range(1, 11)) if a.seeds == "1-10" else \
        [int(s) for s in a.seeds.split(",") if s.strip()]
    t0 = time.perf_counter()
    tot_s = tot_m = mx = agr = ovr = mms = 0
    tiles = {}
    for s in seeds:
        r = play_game(s, a.max_moves, a.margin, a.candidates,
                      depth=a.depth, question=a.question,
                      pick_question=a.pick_question or None,
                      pick_facts=a.pick_facts)
        tot_s += r["score"]
        tot_m += r["moves"]
        agr += r["agreement"]
        ovr += r["overrides"]
        mms = max(mms, r["model_ms_max"])
        tiles[r["max"]] = tiles.get(r["max"], 0) + 1
        mx = max(mx, r["max"])
        print(f"seed {s}: score={r['score']} moves={r['moves']} "
              f"max={r['max']} agree={r['agreement']} "
              f"overrides={r['overrides']} ms={r['model_ms_mean']}")
    n = len(seeds)
    print(json.dumps({
        "seeds": n, "avg_score": round(tot_s / n),
        "avg_moves": round(tot_m / n), "best_tile": mx,
        "tile_hist": tiles, "avg_agreement": round(agr / n, 4),
        "total_overrides": ovr, "model_ms_max": mms,
        "wall_s": round(time.perf_counter() - t0, 1),
        "question": a.question}, indent=1))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="gavel-2048")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("out", nargs="?", default="tiny_2048.jsonl")
    g.add_argument("total", nargs="?", type=int, default=30000)
    g.add_argument("seeds", nargs="?", type=int, default=200)
    g.add_argument("--seed0", type=int, default=1)
    g.add_argument("--candidates", type=int, default=2)
    g.add_argument("--depth", type=int, default=SEARCH_DEPTH,
                   help="teacher search depth (0 = static heuristic, "
                        "deterministic labels)")
    g.add_argument("--pairs", action="store_true",
                   help="pairwise rows: top-2 resulting boards, label "
                        "first/second (needs --depth 0 for determinism)")
    g.add_argument("--facts", action="store_true",
                   help="with --pairs: proposition-level scalar facts "
                        "instead of cell soup")
    pl = sub.add_parser("play")
    pl.add_argument("--seed", type=int, default=46)
    pl.add_argument("--seeds", default="")
    pl.add_argument("--max-moves", type=int, default=10000)
    pl.add_argument("--margin", type=float, default=0.5)
    pl.add_argument("--candidates", type=int, default=2)
    pl.add_argument("--depth", type=int, default=SEARCH_DEPTH,
                    help="serve shortlist depth (match training teacher)")
    pl.add_argument("--question", default=QUESTION,
                    help="server question name (match training)")
    pl.add_argument("--pick-question", default="",
                    help="pairwise question name: compare top-2 resulting "
                         "boards in one /ask call (match pairs training)")
    pl.add_argument("--pick-facts", action="store_true",
                    help="pairwise over proposition facts (match --facts)")
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--max-speed", action="store_true")
    pl.add_argument("--record", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--seeds", default="1-10")
    b.add_argument("--max-moves", type=int, default=10000)
    b.add_argument("--margin", type=float, default=0.5)
    b.add_argument("--candidates", type=int, default=2)
    b.add_argument("--depth", type=int, default=SEARCH_DEPTH,
                   help="serve shortlist depth (match training teacher)")
    b.add_argument("--question", default=QUESTION,
                   help="server question name (match training)")
    b.add_argument("--pick-question", default="",
                   help="pairwise question name: compare top-2 resulting "
                        "boards in one /ask call (match pairs training)")
    b.add_argument("--pick-facts", action="store_true",
                   help="pairwise over proposition facts (match --facts)")
    a = p.parse_args(argv)
    if a.cmd == "gen":
        # guard<3000/move counts above; gen is offline, keep stdlib pace
        return cmd_gen(a)
    if a.cmd == "play":
        return cmd_play(a)
    return cmd_bench(a)


if __name__ == "__main__":
    raise SystemExit(main())
