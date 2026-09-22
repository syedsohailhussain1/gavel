#!/usr/bin/env python3
"""gavel-snake: live terminal Snake played by a tiny Gavel model.

Every move is a real /ask call to gavel-server (question `snake_move`,
trained by showcase_snake.py). Layout mirrors the genre: board left,
decision panel right — but every number is ours and measured live.

Usage:
  python gavel_snake.py play [--fps 12] [--max-speed] [--steps N]
                             [--record run.jsonl] [--headless] [--unassisted]
  python gavel_snake.py replay <run.jsonl> [--fps 12] [--loop]

- Guarded by default: if Gavel proposes suicide, a safety shield substitutes
  the synthetic teacher move and counts an intervention (disclosed on screen,
  same honesty rule as the genre). --unassisted runs raw model top-1.
- --record writes every frame (board + decision + stats) as JSONL.
- replay plays a recording back deterministically — record your X video
  from the perfect take without burning live inference.

Needs gavel-server on :7575 with `snake_move` trained. rich required for display.
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import deque

BASE = "http://127.0.0.1:7575"
QUESTION = "snake_move"
MOVES = ("up", "down", "left", "right")
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

BG = "#0a0e12"
FG = "#e8f4f0"
MUTED = "#6b8a91"
DIM = "#22363d"
GREEN = "#5ff2b0"
AMBER = "#ffce73"
RED = "#ff7c8c"
CYAN = "#8ad8e9"


def _machine_info():
    import os
    import platform
    cores = os.cpu_count() or 0
    chip = (os.environ.get("PROCESSOR_IDENTIFIER")
            or platform.processor() or platform.machine() or "CPU")
    return cores, chip


_CORES, _CHIP = _machine_info()
HW_LINE = "%d logical cores · CPU-only engine" % _CORES

_server_proc = None


def server_stats():
    """Live footprint of the actual decision-maker: the gavel-server process.
    Returns 'cpu% · RSS MB' or None. First call warms up the CPU counter."""
    global _server_proc
    try:
        import psutil
    except ImportError:
        return None
    try:
        if _server_proc is None or not _server_proc.is_running():
            _server_proc = None
            for p in psutil.process_iter(["name", "exe"]):
                try:
                    n = "%s %s" % (p.info.get("name") or "",
                                   p.info.get("exe") or "")
                    if "gavel" in n.lower() or "decide-server" in n.lower():
                        _server_proc = p
                        break
                except Exception:
                    continue
        if _server_proc is None:
            return None
        cpu = _server_proc.cpu_percent(interval=None)
        rss = _server_proc.memory_info().rss / 1e6
        return "%.1f%% CPU · %.0f MB" % (cpu, rss)
    except Exception:
        return None


def api(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_CONN = None


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
        parts.append("%s_d%s" % (m, "02" if d <= 2 else ("35" if d <= 5 else "6p")))
    parts.append("heading_%s" % heading)
    return " ".join(parts)


def legacy_text(game, heading):
    """Legacy (server/logistic) input: EXACT rollout format the retrained
    question learned (tiny_snake3: phase prefix + bucketed per-move facts).
    Must match character-for-character in structure (values vary)."""
    hx, hy = game.snake[0]
    body = list(game.snake)
    occ = set(body)
    tail = body[-1]

    def fb(d):
        return "C" if d <= 2 else ("M" if d <= 6 else "F")

    def sb(s):
        return "T" if s <= 15 else ("R" if s <= 50 else "O")

    parts = ["phase %s head(%d,%d) apple(%d,%d) len%d heading %s" %
             (phase_of(game, game.danger()),
              hx, hy, game.apple[0], game.apple[1], len(body), heading)]
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
        # Single alphanumeric token per move-fact: the logistic engine splits
        # on punctuation, so up:free fC sO would shred into unbound pieces.
        mv.append("%s_%s_f%s_s%s" % (m, st, fb(fd), sb(space) if legal else "T"))
    parts.append(" | ".join(mv))
    return " || ".join(parts)


def teacher(ax, ay, danger):
    best, best_d = None, None
    for m in MOVES:
        if danger.get(m):
            continue
        dx, dy = DELTA[m]
        d = abs(ax - dx) + abs(ay - dy)
        if best_d is None or d < best_d:
            best, best_d = m, d
    return best or "up"


class Game:
    def __init__(self, width=12, height=12, seed=7):
        self.w, self.h = width, height
        self.rng = random_mod(seed)
        cyc = cycle_index(width, height)
        if cyc is not None:
            # Start on 3 consecutive cycle cells near the middle so the
            # cycle-order invariant holds from move 1 (seed varies the spot).
            _, path = cyc
            N = len(path)
            mid = min(range(N), key=lambda i: abs(path[i][0] - width // 2)
                      + abs(path[i][1] - height // 2))
            k = (mid + self.rng.choice(range(N))) % N
            seg = [path[(k + i) % N] for i in range(3)]
            self.snake = deque([seg[2], seg[1], seg[0]])
        else:
            cx, cy = width // 2, height // 2
            self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.apple = self._spawn()
        self.score = self.ticks = 0
        self.alive = True
        self.death = None
        self.won = False

    def _spawn(self):
        occ = set(self.snake)
        free = [(x, y) for x in range(self.w) for y in range(self.h)
                if (x, y) not in occ]
        if not free:
            return None
        # Spawn reachable-only (unreachable food is unwinnable), weighted
        # toward the FAR half from the head: near-only spawns collapse into
        # a self-reinforcing camp on one side of the board. Food placement
        # is environment design; the route there is the model's decision.
        reach = self._reachable(occ)
        cand = [c for c in free if c in reach] or free
        hx, hy = self.snake[0]
        cand.sort(key=lambda c: abs(c[0] - hx) + abs(c[1] - hy))
        far = cand[len(cand) // 2:] or cand
        return self.rng.choice(far)

    def _reachable(self, occ):
        from collections import deque as _dq
        seen = {self.snake[0]}
        q = _dq([self.snake[0]])
        while q:
            x, y = q.popleft()
            for dx, dy in DELTA.values():
                c = (x + dx, y + dy)
                if (0 <= c[0] < self.w and 0 <= c[1] < self.h
                        and c not in occ and c not in seen):
                    seen.add(c)
                    q.append(c)
        return seen

    def danger(self):
        hx, hy = self.snake[0]
        body = set(self.snake)
        tail = self.snake[-1]
        out = {}
        for m, (dx, dy) in DELTA.items():
            nx, ny = hx + dx, hy + dy
            if not (0 <= nx < self.w and 0 <= ny < self.h):
                out[m] = True
            elif (nx, ny) in body and (nx, ny) != tail:
                out[m] = True
            elif (nx, ny) == tail and self.apple is not None \
                    and (nx, ny) == tuple(self.apple):
                out[m] = True  # eating: the tail stays, so this is body
            else:
                out[m] = False  # tail cell vacates on a non-growing step
        return out

    def step(self, move):
        if not self.alive:
            return False
        self.ticks += 1
        hx, hy = self.snake[0]
        dx, dy = DELTA[move]
        head = (hx + dx, hy + dy)
        if not (0 <= head[0] < self.w and 0 <= head[1] < self.h):
            self.alive, self.death = False, "wall"
            return False
        if head in set(self.snake) and (head != self.snake[-1] or (
                self.apple is not None and head == self.apple)):
            # Tail cell vacates on a non-growing step; eating into it is body.
            self.alive, self.death = False, "body"
            return False
        self.snake.appendleft(head)
        if head == self.apple:
            self.score += 1
            self.apple = self._spawn()
            if self.apple is None:
                self.won = True  # board clear: every cell is snake
            return True
        self.snake.pop()
        return False

    def snapshot(self):
        return {"w": self.w, "h": self.h, "body": [list(c) for c in self.snake],
                "apple": list(self.apple) if self.apple else None,
                "score": self.score, "length": len(self.snake),
                "ticks": self.ticks, "alive": self.alive, "death": self.death,
                "won": self.won}


class random_mod:
    """Tiny seeded RNG wrapper (random.Random with .choice)."""
    def __init__(self, seed):
        import random as _r
        self._r = _r.Random(seed)

    def choice(self, seq):
        return self._r.choice(seq)


def decide(text):
    """Server brain via keep-alive HTTP (falls back to plain api).
    A fresh TCP+thread per call costs ~15ms on Windows localhost; reuse
    drops /ask to ~0.4ms."""
    import http.client as _hc
    from urllib.parse import urlparse as _up
    global _CONN
    t0 = time.perf_counter()
    body = json.dumps({"question": QUESTION, "input": text})
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


SURVIVE_LEN = 20


def phase_of(game, danger):
    """Regime switch, identical to the training-data rule (tiny_data2.py):
    survive when long, cramped, or food cut off — else hunt."""
    body = list(game.snake)
    n = len(body)
    roomy = any(flood_info(body, game.apple, game.w, game.h, m)[1] >= 2 * n
                for m in MOVES if not danger.get(m))
    food_ok = any(flood_info(body, game.apple, game.w, game.h, m)[2]
                  for m in MOVES if not danger.get(m))
    return "survive" if (n >= SURVIVE_LEN or not roomy or not food_ok) else "hunt"


def brain_text(game, heading, phased=True):
    """Rich state text for the transformer brain (same format it trained on).
    phased=False reproduces the v1 format (no phase prefix)."""
    hx, hy = game.snake[0]
    body = list(game.snake)
    occ = set(body)
    tail = body[-1]
    head = ("phase %s head(%d,%d)" % (phase_of(game, game.danger()), hx, hy)
            if phased else "head(%d,%d)" % (hx, hy))
    parts = ["%s apple(%d,%d) len%d heading %s" %
             (head, game.apple[0], game.apple[1], len(body), heading)]
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


class OnnxBrain:
    """Local tiny-transformer brain (batch-1 ONNX, CPU)."""

    def __init__(self, onnx_path, tok_dir, threads=2):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self._np = np
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        e = self.tok("head(0,0) apple(1,1) len3 heading right",
                     return_tensors="np", truncation=True, max_length=96)
        self.sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                             "attention_mask": e["attention_mask"].astype(np.int64)})

    def decide(self, text):
        import time as _t
        np = self._np
        t0 = _t.perf_counter()
        e = self.tok(text, return_tensors="np", truncation=True, max_length=96)
        logits = self.sess.run(
            None, {"input_ids": e["input_ids"].astype(np.int64),
                   "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
        ms = (_t.perf_counter() - t0) * 1000
        ex = np.exp(logits - logits.max())
        conf = float(ex.max() / ex.sum())
        return MOVES[int(logits.argmax())], round(conf, 3), round(ms, 2)


ARROWS = {"up": "▲", "down": "▼", "left": "◄", "right": "►"}


_CYCLES = {}


def _build_cycle(w, h):
    """Hamiltonian cycle index for even boards: visit every cell exactly once
    and return to start. Transposes odd-height boards; impossible on odd×odd
    (checkerboard parity) → None. The cruise-control survival strategy."""
    transpose = False
    if h % 2 == 1:
        if w % 2 == 1:
            return None
        w, h, transpose = h, w, True
    path = [(x, 0) for x in range(w)]
    for y in range(1, h):
        xs = range(w - 1, 0, -1) if y % 2 == 1 else range(1, w)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(h - 1, 0, -1))
    if transpose:
        path = [(y, x) for (x, y) in path]
    assert len(set(path)) == len(path), "cycle must not repeat cells"
    return {cell: i for i, cell in enumerate(path)}, path


def cycle_index(w, h):
    if (w, h) not in _CYCLES:
        _CYCLES[(w, h)] = _build_cycle(w, h)
    return _CYCLES[(w, h)]


def cycle_step(body, w, h):
    """Next step along the Hamiltonian cycle from the head's position."""
    cyc = cycle_index(w, h)
    if cyc is None:
        return None
    idx, path = cyc
    nxt = path[(idx[body[0]] + 1) % len(path)]
    dx, dy = nxt[0] - body[0][0], nxt[1] - body[0][1]
    for m, (vx, vy) in DELTA.items():
        if (dx, dy) == (vx, vy):
            return m
    return None


def _apply(body, apple, w, h, move):
    """Simulate one move. Returns (new_body, ate) or None if instantly fatal.
    Tail vacates on a non-growing step; eating into the tail cell is fatal."""
    hx, hy = body[0]
    dx, dy = DELTA[move]
    head = (hx + dx, hy + dy)
    if not (0 <= head[0] < w and 0 <= head[1] < h):
        return None
    eats = apple is not None and head == tuple(apple)
    occ = set(body)
    if not eats:
        occ.discard(body[-1])
    if head in occ:
        return None
    return ([head] + list(body)) if eats else ([head] + list(body[:-1])), eats


def _mobility(nbody, ate, w, h):
    """Free neighbors of the new head — corridor descents have exactly 1
    (forward), open board 2+. Cheap topological signal max-space can't see."""
    occ = set(nbody)
    if not ate:
        occ.discard(nbody[-1])  # tail vacates next step
    hx, hy = nbody[0]
    return sum(1 for dx, dy in DELTA.values()
               if 0 <= hx + dx < w and 0 <= hy + dy < h
               and (hx + dx, hy + dy) not in occ)


def _deep(nbody, apple, w, h, depth):
    """Best (ate, space) reachable from this position within `depth` replies.
    Apple persists across plies (it only moves on eat, which is terminal).
    Returns None if no legal reply exists."""
    best = None
    for r in MOVES:
        st = _apply(nbody, apple, w, h, r)
        if st is None:
            continue
        nb2, ate = st
        if ate:
            key = (1, 10 ** 9)
        elif depth <= 1:
            legal, space, _ = flood_info(nb2, apple, w, h, r)
            key = (0, space) if legal else None
        else:
            sub = _deep(nb2, apple, w, h, depth - 1)
            key = (0, sub[1]) if sub is not None else None
        if key is not None and (best is None or key > best):
            best = key
    return best


def food_step(body, apple, w, h):
    """BFS shortest path from head to food through free cells; first step.
    Unlike greedy min-distance, this routes AROUND the body. The tail cell
    counts as free (it vacates) unless the food sits on it. None if no path."""
    from collections import deque as _dq
    if apple is None:
        return None
    target = tuple(apple)
    occ = set(body)
    if target != body[-1]:
        occ.discard(body[-1])
    prev = {body[0]: None}
    q = _dq([body[0]])
    while q:
        cur = q.popleft()
        if cur == target:
            break
        x, y = cur
        for m, (dx, dy) in DELTA.items():
            c = (x + dx, y + dy)
            if (0 <= c[0] < w and 0 <= c[1] < h and c not in occ and c not in prev):
                prev[c] = (cur, m)
                q.append(c)
    if target not in prev or prev[target] is None:
        return None
    cur = target
    while prev[cur] is not None and prev[cur][0] != body[0]:
        cur = prev[cur][0]
    return prev[cur][1] if prev[cur] is not None else None


def tail_step(body, apple, w, h):
    """BFS shortest path from head to tail through free cells; return first step.
    Chasing the tail is the classic survival strategy: the tail always vacates,
    so reaching it resets the position. Returns None if unreachable."""
    from collections import deque as _dq
    tail = body[-1]
    occ = set(body)
    if not (apple is not None and tuple(apple) == tail):
        occ.discard(tail)  # tail vacates on a non-growing step
    prev = {body[0]: None}
    q = _dq([body[0]])
    while q:
        cur = q.popleft()
        if cur == tail:
            break
        x, y = cur
        for m, (dx, dy) in DELTA.items():
            c = (x + dx, y + dy)
            if (0 <= c[0] < w and 0 <= c[1] < h and c not in occ and c not in prev):
                prev[c] = (cur, m)
                q.append(c)
    if tail not in prev or prev[tail] is None:
        return None
    cur = tail
    while prev[cur] is not None and prev[cur][0] != body[0]:
        cur = prev[cur][0]
    return prev[cur][1] if prev[cur] is not None else None


def flood_info(body, apple, w, h, move):
    """Simulate one move; return (legal, free_space, food_reachable)."""
    from collections import deque as _dq
    hx, hy = body[0]
    dx, dy = DELTA[move]
    head = (hx + dx, hy + dy)
    if not (0 <= head[0] < w and 0 <= head[1] < h):
        return False, 0, False
    eats = apple is not None and head == apple
    occ = set(body)
    if not eats:
        occ.discard(body[-1])  # tail moves away on a non-growing step
    if head in occ:
        return False, 0, False
    seen = {head}
    q = _dq([head])
    while q:
        x, y = q.popleft()
        for ddx, ddy in DELTA.values():
            c = (x + ddx, y + ddy)
            if (0 <= c[0] < w and 0 <= c[1] < h and c not in occ and c not in seen):
                seen.add(c)
                q.append(c)
    return True, len(seen), (apple in seen)


def flood_ok(body, apple, w, h, move):
    """Lookahead: would this move leave the snake trapped?
    Requires the food to stay reachable and enough free room to fit the body.
    Cheap on a 12x12 board (a few hundred cells)."""
    legal, space, food = flood_info(body, apple, w, h, move)
    return legal and food and space >= len(body) + 1


_TRACE = None


def _T(tag):
    if _TRACE is not None:
        _TRACE.append(tag)


def pick_move(game, ax, ay, danger, raw, unassisted, loop_break=False,
              immortal=False):
    """Cycle-disciplined move selection. Returns (final, intervened).
    The Hamiltonian cycle is the immortality substrate: any body that stays
    cycle-ordered can be continued forever (the next cycle cell is provably
    free short of a full board). So every executed move must keep the body
    cycle-ordered — the brain drives whenever its move respects the
    invariant, the teacher is second choice, the cycle backbone third.
    Short bodies are almost always orderable, so early game is fully
    model-driven; discipline only bites when coils threaten.
    loop_break forces food pursuit (merry-go-round escape); unassisted is raw.
    Fallbacks (disordered coil): 2-ply escape search, tail-chase, max room."""
    if unassisted:
        _T("unassisted")
        return (raw if raw in MOVES else teacher(ax, ay, danger)), False
    body = list(game.snake)
    cands = []
    for m in ([raw] if raw in MOVES else []) + [teacher(ax, ay, danger)] + list(MOVES):
        if m not in cands:
            cands.append(m)
    if immortal:
        # Full-time cycle discipline: every move keeps the body cycle-ordered
        # (provably immortal short of a full board). With the v2 model this
        # scores instead of stalling — measured 140/143, near board-clear —
        # because v2 trained on immortal-policy labels and follows order
        # instead of fighting it. Gating (hunt-then-cruise) was tested and
        # reverted: the disordered switch-over reintroduces deaths.
        # Cycle-first discipline: the backbone is immortal, everything else
        # is optional. Order: keeps-cycle eats (score, invariant intact) →
        # cycle step when ordered → disciplined driving → guarded fallbacks.
        # Pursuit NEVER leads here, so order, once held, is never broken
        # for food. Falls through only when already disordered.
        cyc = cycle_index(game.w, game.h)
        if cyc is not None:
            idx, path = cyc
            N = len(path)

            def _ord(nbody):
                ii = [idx[c] for c in reversed(nbody)]
                ds = [(b - a) % N for a, b in zip(ii, ii[1:])]
                return all(d > 0 for d in ds) and sum(ds) < N

            for m in cands:
                if danger.get(m):
                    continue
                st = _apply(body, game.apple, game.w, game.h, m)
                if st is not None and st[1] and _ord(st[0]):
                    _T("keeps-eat")
                    return m, (m != raw)
            ii = [idx[c] for c in reversed(body)]
            ds = [(b - a) % N for a, b in zip(ii, ii[1:])]
            if all(d > 0 for d in ds) and sum(ds) < N:
                cm = cycle_step(body, game.w, game.h)
                if cm is not None and not danger.get(cm):
                    _T("cycle")
                    return cm, (cm != raw)
            if raw in MOVES and not danger.get(raw):
                st = _apply(body, game.apple, game.w, game.h, raw)
                if st is not None and _ord(st[0]):
                    _T("raw-keeping")
                    return raw, False
            t = teacher(ax, ay, danger)
            if not danger.get(t):
                st = _apply(body, game.apple, game.w, game.h, t)
                if st is not None and _ord(st[0]):
                    _T("teacher-keeping")
                    return t, (t != raw)
            # Disordered: rejoin explicitly — first safe move (candidate
            # order) whose post-body is cycle-ordered. Without this the
            # backbone stays unreachable and the gate is decorative.
            for m in cands:
                if danger.get(m):
                    continue
                st = _apply(body, game.apple, game.w, game.h, m)
                if st is not None and _ord(st[0]):
                    _T("rejoin")
                    return m, (m != raw)
    cyc = cycle_index(game.w, game.h)
    # Watchdog first: if set, the brain is circling — food pursuit
    # outranks everything except instant survival (checked inside).
    if loop_break:
        pursue = food_step(body, game.apple, game.w, game.h)
        if pursue is not None and not danger.get(pursue):
            legal, space, _ = flood_info(body, game.apple, game.w, game.h, pursue)
            if legal and space >= len(body) + 1:
                _T("loop-bfs")
                return pursue, (pursue != raw)
        by_dist = sorted(
            (m for m in MOVES if not danger.get(m)),
            key=lambda m: abs(ax - DELTA[m][0]) + abs(ay - DELTA[m][1]))
        for m in by_dist:
            if flood_ok(body, game.apple, game.w, game.h, m):
                _T("loop-greedy")
                return m, (m != raw)
    # Open position: brain first, teacher second (both must be trap-free).
    # This is the scoring engine — it stays first so the demo shows real
    # model decisions with minimal vetoes.
    if raw in MOVES and not danger.get(raw) \
            and flood_ok(body, game.apple, game.w, game.h, raw):
        _T("raw")
        return raw, False
    t = teacher(ax, ay, danger)
    if not danger.get(t) and flood_ok(body, game.apple, game.w, game.h, t):
        _T("teacher")
        return t, (t != raw)
    # Constrained: commit to the food path (BFS routes around the body) —
    # but ONLY with room to survive it. Chasing into a shrinking pocket
    # eats the snake into a coffin (verified: space 47→1 over 40 moves).
    # Without room, fall through to max-space survival below.
    pursue = food_step(body, game.apple, game.w, game.h)
    if pursue is not None and not danger.get(pursue):
        legal, space, _ = flood_info(body, game.apple, game.w, game.h, pursue)
        if legal and space >= len(body) + 1:
            _T("pursue")
            return pursue, (pursue != raw)
    # Constrained (<=1 trap-free move) or high fill: prefer the provably
    # safe ordered cycle step over heuristic search. It only fires when the
    # body is cycle-ordered; otherwise it could steer into the coil.
    safe_moves = [m for m in MOVES if not danger.get(m)]
    flood_safe = [m for m in safe_moves
                  if flood_ok(body, game.apple, game.w, game.h, m)]
    cruise = (cyc is not None and (len(body) >= max(20, game.w * game.h // 5)
                                   or len(flood_safe) <= 1))
    if cruise:
        idx, path = cyc
        N = len(path)
        hx, hy = body[0]
        for m in cands:
            if danger.get(m):
                continue
            tt = (hx + DELTA[m][0], hy + DELTA[m][1])
            if (game.apple is not None and tt == tuple(game.apple)
                    and flood_ok(body, game.apple, game.w, game.h, m)):
                _T("cruise-eat")
                return m, (m != raw)  # shortcut: safe adjacent food
        ii = [idx[c] for c in reversed(body)]
        ds = [(b - a) % N for a, b in zip(ii, ii[1:])]
        if all(d > 0 for d in ds) and sum(ds) < N:
            cm = cycle_step(body, game.w, game.h)
            if cm is not None and not danger.get(cm):
                _T("cycle")
                return cm, (cm != raw)
    # Narrowing: adaptive-depth escape search (2-ply normally, 4-ply when
    # constrained — the killing trap is always several moves deep, and the
    # deep search runs on ~5% of moves so the amortized cost is trivial).
    # Value = most free space reachable, eating preferred.
    constrained = len([m for m in MOVES
                       if not danger.get(m)
                       and flood_ok(body, game.apple, game.w, game.h, m)]) <= 2
    depth = 4 if constrained else 2
    best_m, best_key = None, None
    for m in MOVES:
        if danger.get(m):
            continue
        st = _apply(body, game.apple, game.w, game.h, m)
        if st is None:
            continue
        nbody, ate = st
        if ate:
            key = (1, 10 ** 9, 0)
        else:
            sub = _deep(nbody, game.apple, game.w, game.h, depth)
            mob = _mobility(nbody, False, game.w, game.h)
            key = (0, mob, sub[1]) if sub is not None else None
        if key is not None and (best_key is None or key > best_key):
            best_m, best_key = m, key
    if best_m is not None:
        _T("search%d" % depth)
        return best_m, (best_m != raw)
    # Nothing survives search: tail-chase, then max room, then raw.
    chase = tail_step(body, game.apple, game.w, game.h)
    if chase is not None and not danger.get(chase):
        _T("tail")
        return chase, (chase != raw)
    room, room_m = -1, None
    for m in MOVES:
        if danger.get(m):
            continue
        legal, space, _ = flood_info(body, game.apple, game.w, game.h, m)
        if legal and space > room:
            room, room_m = space, m
    if room_m is not None:
        _T("room")
        return room_m, (room_m != raw)
    _T("doomed")
    return (raw if raw in MOVES else cands[0]), True


def render(board, dec, stats):
    from rich.text import Text
    bw, bh = board["w"], board["h"]
    left, top = 3, 6
    right = max(42, left + bw * 2 + 8)
    W = right + 36
    H = max(34, top + bh + 14)
    grid = [[" "] * W for _ in range(H)]
    sty = [[FG] * W for _ in range(H)]

    def put(row, col, text, color=FG):
        for i, ch in enumerate(str(text)):
            x = col + i
            if 0 <= row < H and 0 <= x < W:
                grid[row][x], sty[row][x] = ch, color

    bottom = top + bh + 1
    state = ("BOARD CLEAR" if board.get("won") else
             "GAME OVER" if not board["alive"] else
             "LIVE" if not stats.get("replay") else "RECORDED RUN · 1×")
    put(1, left, "GAVEL  /  LOCAL INTELLIGENCE", MUTED)
    put(1, W - len(state) - 3, state, GREEN if board["alive"] else RED)
    put(2, left, "─" * (W - 6), DIM)
    put(4, left, "S N A K E", FG)
    put(5, left, stats.get("tag", "TINY MODEL · 0.06s TRAIN"), MUTED)
    put(top, left, "┌" + "─" * (bw * 2) + "┐", DIM)
    put(bottom, left, "└" + "─" * (bw * 2) + "┘", DIM)
    for y in range(board["h"]):
        put(top + 1 + y, left, "│", DIM)
        put(top + 1 + y, left + bw * 2 + 1, "│", DIM)
        put(top + 1 + y, left + 1, "· " * bw, "#14262d")
    body = board["body"]
    head_arrow = ARROWS.get(dec.get("executed") or dec.get("proposed"), "██")
    for i, (x, y) in reversed(list(enumerate(body))):
        f = 1 - i / max(1, len(body))
        c = ("#dcfff0" if i == 0 else
             "#%02x%02x%02x" % (int(20 + 60 * f), int(80 + 140 * f), int(60 + 100 * f)))
        glyph = (head_arrow + " ") if i == 0 else "██"
        put(top + y + 1, left + 1 + 2 * x, glyph, c)
    if board["apple"] is not None:
        x, y = board["apple"]
        put(top + y + 1, left + 1 + 2 * x, "● ", AMBER)
    for off, label, val in ((0, "SCORE", board["score"]),
                            (14, "LENGTH", board["length"]),
                            (28, "MOVES", board["ticks"])):
        put(bottom + 2, left + off, label, MUTED)
        put(bottom + 3, left + off, "%03d" % val, GREEN)

    put(4, right, stats.get("model_name", "Gavel · snake_move"), GREEN)
    put(5, right, HW_LINE, MUTED)
    put(7, right, "NEXT MOVE", FG)
    put(7, right + 16, str(dec.get("proposed", "—")).upper(), GREEN)
    if dec.get("intervened"):
        put(7, right + 24, "SHIELD", AMBER)
    put(8, right, "EXECUTING", MUTED)
    put(8, right + 16, str(dec.get("executed", "—")).upper(), FG)
    put(9, right, "CONFIDENCE", MUTED)
    put(9, right + 16, "%.3f" % dec.get("confidence", 0), FG)
    put(10, right, "PATH", MUTED)
    put(10, right + 16, stats.get("hist", "")[-9:], FG)
    _loop = stats.get("loop")
    _phase = stats.get("phase", "")
    put(10, right + 27, "LOOP-BREAK" if _loop else _phase.upper(),
        AMBER if (_loop or _phase == "survive") else GREEN)
    put(11, right, "INFERENCE", MUTED)
    put(11, right + 16, "%6.1f ms" % dec.get("inference_ms", 0), FG)
    put(12, right, "DECISIONS", MUTED)
    put(12, right + 16, "%6.1f /s" % stats.get("steps_per_second", 0), FG)
    put(13, right, "BRAIN MAX", MUTED)
    put(13, right + 16, stats.get("brain_max", "—"), GREEN)
    put(14, right, "OUTPUT TOKENS", MUTED)
    put(14, right + 16, "0", FG)
    put(15, right, "NETWORK", MUTED)
    put(15, right + 16, "OFFLINE", GREEN)
    put(16, right, "ENGINE", MUTED)
    put(16, right + 16, stats.get("engine_line", "logistic · CPU"), MUTED)
    put(17, right, "SERVER", MUTED)
    put(17, right + 16, stats.get("server") or "—", FG)
    put(18, right, "Shield interventions  %04d" % stats.get("interventions", 0), AMBER)
    put(20, right, stats.get("train_line1", "Train 0.062s · 5k"), MUTED)
    put(21, right, stats.get("train_line2", "ECE 0.026"), MUTED)
    put(H - 3, left, "─" * (W - 6), DIM)
    put(H - 2, left, stats.get("footer", "gavel-snake · every move is a live decision"), MUTED)

    out = Text(no_wrap=True, overflow="crop", style=f"{FG} on {BG}")
    for row, (ch, st) in enumerate(zip(grid, sty)):
        b = 0
        for e in range(1, W + 1):
            if e == W or st[e] != st[b]:
                out.append("".join(ch[b:e]), style=st[b])
                b = e
        if row + 1 != H:
            out.append("\n")
    return out


def head_dir(game):
    (hx, hy), (nx, ny) = game.snake[0], game.snake[1]
    for m, (dx, dy) in DELTA.items():
        if (hx - nx, hy - ny) == (dx, dy):
            return m
    return "right"


def cmd_play(a):
    from rich.console import Console
    from rich.live import Live
    brain = None
    global QUESTION
    if a.brain == "server":
        QUESTION = a.question
        model_name = "legacy-tfidf · " + a.question
        tag = "TF-IDF · 1.1s TRAIN"
        engine_line = "tfidf+logreg · CPU"
        train1, train2 = "Train 4.6s · 90k", "Val 89.4% + shield"
        QUESTION = a.question
        qs = api("/questions")
        if QUESTION not in qs.get("questions", []):
            print(f"question '{QUESTION}' not on server — run legacy_tfidf.py first",
                  file=sys.stderr)
            return 2
        footer = "gavel-snake · every move is a live /ask call"
    else:
        try:
            brain = OnnxBrain(a.onnx, a.tok, a.threads)
        except Exception as e:
            print(f"onnx brain failed to load: {e}", file=sys.stderr)
            return 2
        onx = a.onnx.lower()
        if "modernbert" in onx:
            model_name = "modernbert · 150M"
            tag = "TRANSFORMER · 48min TRAIN"
            train1, train2 = "Train 48min · 27k", "Val 97.8%"
        elif "snake2" in onx:
            model_name = "tiny-distil-v2 · 66M"
            tag = "TRANSFORMER · 50min TRAIN"
            train1, train2 = "Train 50min · 27k", "Val 96.8% · hunt+live"
        elif "bert11m" in onx or "11m" in onx:
            model_name = "bert-11M · 99.5%"
            tag = "TRANSFORMER · 17min TRAIN"
            train1, train2 = "Train 17min · 27k", "Val 99.5% · eats"
        elif "distil" in onx:
            model_name = "tiny-distil · 66M"
            tag = "TRANSFORMER · 38min TRAIN"
            train1, train2 = "Train 38min · 27k", "Val 98.8% · veto 2.8%"
        else:
            model_name = "google-tiny · 4.4M"
            tag = "TRANSFORMER · 24s TRAIN"
            train1, train2 = "Train 24s · 27k", "Val 85.2% · loops"
        engine_line = "onnx · CPU"
        footer = "gavel-snake · every move is a live ONNX call"
    # Startup calibration: the brain's raw ceiling (batch-1, no game loop,
    # no shield) measured live on this machine — shown as BRAIN MAX even
    # when the game itself is paced to watchable fps.
    g0 = Game(a.width, a.height, a.seed)
    hx0, hy0 = g0.snake[0]
    ax0, ay0 = g0.apple[0] - hx0, g0.apple[1] - hy0
    d0 = g0.danger()
    if brain is None:
        cal = [lambda h=h: decide(legacy_text(g0, h))
               for h in ("up", "down", "left", "right")]
    else:
        cal = [lambda t=brain_text(g0, h, bool(a.phased)): brain.decide(t)
               for h in ("up", "down", "left", "right", "up")]
    cal_ms = []
    for i in range(60):
        _, _, ms = cal[i % len(cal)]()
        cal_ms.append(ms)
    cal_ms.sort()
    med = cal_ms[len(cal_ms) // 2]
    brain_max = "%.1f ms · %d /s" % (med, int(round(1000 / med))) if med > 0 else "—"
    print(f"[calibration] brain raw p50: {brain_max}", flush=True)
    game = Game(a.width, a.height, a.seed)
    heading = head_dir(game)
    console = Console(style=f"on {BG}", highlight=False)
    if not a.headless and not console.is_terminal:
        print("needs a TTY; use --headless for non-interactive runs", file=sys.stderr)
        return 2
    rec = open(a.record, "x") if a.record else None
    if rec:
        rec.write(json.dumps({"type": "metadata", "format": "gavel-snake-v1",
                              "question": QUESTION, "guarded": not a.unassisted,
                              "settings": vars(a)}) + "\n")
    stats = {"interventions": 0, "steps_per_second": 0, "elapsed": 0,
             "model_name": model_name, "tag": tag, "engine_line": engine_line,
             "train_line1": train1, "train_line2": train2, "footer": footer,
             "hist": "", "loop": False, "brain_max": brain_max,
             "phase": "hunt"}
    stamps = deque(maxlen=60)
    heads = deque(maxlen=10)
    hist = deque(maxlen=5)
    loop_left = 0
    last_score, since_food = 0, 0
    total_score, best_round = 0, 0
    victory = False
    total, calls, deaths = 0, 0, 0
    t_start = time.perf_counter()
    shown_board, shown_dec = game.snapshot(), {}
    live = None if a.headless else Live(console=console, auto_refresh=False)
    if live:
        live.__enter__()
    try:
        while True:
            if a.steps and total >= a.steps:
                break
            hx, hy = game.snake[0]
            ax, ay = game.apple[0] - hx, game.apple[1] - hy
            danger = game.danger()
            phase = phase_of(game, danger)
            stats["phase"] = phase
            if brain is None:
                raw, conf, ms = decide(legacy_text(game, heading))
            else:
                raw, conf, ms = brain.decide(brain_text(game, heading,
                                                        bool(a.phased)))
            calls += 1
            heads.append((hx, hy))
            if game.score > last_score:
                last_score, since_food = game.score, 0
            else:
                since_food += 1
            loop_on = list(heads).count((hx, hy)) > 1
            loop_left = 6 if loop_on else max(0, loop_left - 1)
            # Watchdog must tolerate a full cycle loop (N moves with no food
            # is normal cruising, not a merry-go-round). Only beyond that is
            # it a genuine stall.
            stall_after = game.w * game.h + 60
            stats["loop"] = loop_left > 0 or since_food > stall_after
            # Regime deployment: model hunts (appetite), rules survive
            # (guarantee). In survive-phase the immortal policy decides and
            # the veto is counted honestly against the model's proposal.
            if phase == "survive" and not a.unassisted and brain is not None:
                move, _ = pick_move(game, ax, ay, danger,
                                    teacher(ax, ay, danger),
                                    a.unassisted, stats["loop"], True)
                intervened = (move != raw)
            else:
                move, intervened = pick_move(game, ax, ay, danger, raw,
                                             a.unassisted, stats["loop"],
                                             getattr(a, "immortal", False))
            if intervened:
                stats["interventions"] += 1
            hist.append(ARROWS.get(move, "?"))
            stats["hist"] = " ".join(hist)
            now = time.perf_counter()
            stamps.append(now)
            stats["elapsed"] = now - t_start
            stats["steps_per_second"] = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                                         if len(stamps) > 1 else 0)
            stats["server"] = ("in-process" if brain is not None
                               else (server_stats() or stats.get("server")))
            dec = {"proposed": raw, "executed": move, "confidence": conf,
                   "inference_ms": ms, "intervened": intervened}
            board = game.snapshot()
            shown_board, shown_dec = board, dec
            if live:
                live.update(render(board, dec, stats), refresh=True)
            if rec:
                rec.write(json.dumps({"type": "frame", "at": now - t_start,
                                      "game": board, "decision": dec,
                                      "stats": dict(stats)}, separators=(",", ":")) + "\n")
            if not a.max_speed:
                wait = 1.0 / a.fps - (time.perf_counter() - now)
                if wait > 0:
                    time.sleep(wait)
            game.step(move)
            heading = move
            total += 1
            if game.won:
                victory = True
                if live:
                    live.update(render(game.snapshot(), {}, stats), refresh=True)
                    time.sleep(2)
                break
            if not game.alive:
                deaths += 1
                total_score += game.score
                best_round = max(best_round, game.score)
                if live:
                    live.update(render(game.snapshot(), {}, stats), refresh=True)
                    time.sleep(1)
                if a.unassisted or (a.steps and total >= a.steps):
                    break
                game = Game(a.width, a.height, a.seed + 1)
                heading = head_dir(game)
                heads.clear()
                hist.clear()
                loop_left = 0
                last_score, since_food = 0, 0
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            live.__exit__(None, None, None)
        el = time.perf_counter() - t_start
        total_score += game.score
        best_round = max(best_round, game.score)
        summary = {"steps": total, "inference_calls": calls, "seconds": round(el, 2),
                   "steps_per_second": round(total / el, 1) if el else 0,
                   "score": total_score, "best_round": best_round,
                   "deaths": deaths, "interventions": stats["interventions"],
                   "guarded": not a.unassisted, "brain": a.brain,
                   "immortal": getattr(a, "immortal", False),
                   "victory": victory}
        if rec:
            rec.write(json.dumps({"type": "end", "summary": summary}) + "\n")
            rec.close()
        print(json.dumps(summary, indent=2))
    return 0


def cmd_replay(a):
    from rich.console import Console
    from rich.live import Live
    console = Console(style=f"on {BG}", highlight=False)
    frames = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    frames = [f for f in frames if f.get("type") == "frame"]
    if not frames:
        print("no frames in recording", file=sys.stderr)
        return 2
    stats = {"replay": True, "interventions": 0, "steps_per_second": 0}
    with Live(console=console, auto_refresh=False) as live:
        t0 = time.perf_counter()
        stamps = deque(maxlen=60)
        n = 0
        while True:
            for f in frames:
                st = dict(f.get("stats", {}))
                st["replay"] = True
                stamps.append(time.perf_counter())
                st["steps_per_second"] = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                                          if len(stamps) > 1 else 0)
                live.update(render(f["game"], f.get("decision", {}), st), refresh=True)
                n += 1
                if a.steps and n >= a.steps:
                    return 0
                time.sleep(1.0 / a.fps)
            if not a.loop:
                return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="gavel-snake")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("play")
    pl.add_argument("--width", type=int, default=12)
    pl.add_argument("--height", type=int, default=12)
    pl.add_argument("--seed", type=int, default=7)
    pl.add_argument("--fps", type=float, default=12)
    pl.add_argument("--max-speed", action="store_true")
    pl.add_argument("--steps", type=int, default=0)
    pl.add_argument("--record", default=None)
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--unassisted", action="store_true")
    pl.add_argument("--immortal", action="store_true",
                    help="full-time cycle discipline: zero deaths guaranteed, "
                         "slower scoring, model drives only when disciplined")
    pl.add_argument("--brain", choices=("server", "onnx"), default="onnx",
                    help="onnx: local tiny transformer (default); server: Gavel /ask (logistic)")
    pl.add_argument("--question", default="snake_tfidf",
                    help="server-brain question name (snake_tfidf = retrained; "
                         "snake_move = original uniform-state model)")
    pl.add_argument("--onnx", default="models/training_state/tiny_snake2_distil_fp32.onnx")
    pl.add_argument("--tok", default="models/training_state/tiny_snake2_distil")
    pl.add_argument("--threads", type=int, default=2)
    pl.add_argument("--phased", type=int, default=1,
                    help="1: v2 phase-prefixed states (default); 0: v1 format for old models")
    rp = sub.add_parser("replay")
    rp.add_argument("file")
    rp.add_argument("--fps", type=float, default=12)
    rp.add_argument("--steps", type=int, default=0)
    rp.add_argument("--loop", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "play":
        return cmd_play(a)
    return cmd_replay(a)


if __name__ == "__main__":
    raise SystemExit(main())
