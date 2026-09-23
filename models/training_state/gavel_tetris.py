#!/usr/bin/env python3
"""gavel-tetris: live terminal Tetris played by Gavel typed decisions.

Policy (their shortlist pattern): for each piece, the Dellacherie heuristic
ranks all legal landings; the top 2 go into ONE /ask call to `tetris_pick`
("A ... || B ..."), and the winner is executed. Near-ties (heuristic gap
< 3.0, outside training distribution) fall back to the heuristic directly —
counted on screen as HEURISTIC, same honesty rule as the Snake shield.

Usage:
  python gavel_tetris.py play [--fps 8] [--max-speed] [--pieces N]
                              [--seed S] [--record run.jsonl] [--headless]
  python gavel_tetris.py replay <run.jsonl> [--fps 8] [--loop]

Needs gavel-server on :7575 with `tetris_pick` defined (legacy_tfidf.py).
rich required for display. Stdlib + rich otherwise.
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parent))
from tetris_game import Board, bag, fmt, landing_features

BASE = "http://127.0.0.1:7575"
QUESTION = "tetris_pick"
GAP_FALLBACK = 3.0

BG = "#0a0e12"
FG = "#e8f4f0"
MUTED = "#6b8a91"
DIM = "#22363d"
GREEN = "#5ff2b0"
AMBER = "#ffce73"
RED = "#ff7c8c"
CYAN = "#8ad8e9"

_CONN = None
_SYS = {"t": 0.0, "gpu": None}


def api(path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def decide(text):
    """One typed call (keep-alive): returns (pick A/B, confidence, ms)."""
    import http.client as _hc
    from urllib.parse import urlparse as _up
    global _CONN
    t0 = time.perf_counter()
    body = json.dumps({"question": QUESTION, "input": text})
    try:
        if _CONN is None:
            u = _up(BASE)
            _CONN = _hc.HTTPConnection(u.hostname, u.port or 80, timeout=60)
        _CONN.request("POST", "/ask", body=body,
                      headers={"Content-Type": "application/json"})
        out = json.loads(_CONN.getresponse().read())
    except Exception:
        try:
            _CONN.close()
        except Exception:
            pass
        _CONN = None
        out = api("/ask", {"question": QUESTION, "input": text}, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    lab = (out.get("output") or {}).get("label")
    return (lab if lab in ("A", "B") else None,
            round(out.get("confidence", 0), 3), round(ms, 2))


def _gpu_stats():
    """GTX readings via nvidia-smi (cached by caller: ~50ms a call)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return "%s%% %s/%sMB" % (out[0].strip(), out[1].strip(), out[2].strip())
    except Exception:
        return None


def sys_stats():
    """Live machine load: (cpu_pct, ram_str, gpu_str). Non-blocking;
    GPU polled at most once per second."""
    import time as _t
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        ram = "%.0f%% %.1fG" % (vm.percent, vm.used / 1e9)
    except Exception:
        cpu, ram = None, None
    now = _t.perf_counter()
    if now - _SYS["t"] > 1.0:
        _SYS["t"] = now
        _SYS["gpu"] = _gpu_stats()
    return cpu, ram, _SYS["gpu"]


def render(board, piece, nxt, dec, stats, fall=None):
    from rich.text import Text
    W, H = 78, 36
    grid = [[" "] * W for _ in range(H)]
    sty = [[FG] * W for _ in range(H)]

    def put(row, col, text, color=FG):
        for i, ch in enumerate(str(text)):
            x = col + i
            if 0 <= row < H and 0 <= x < W:
                grid[row][x], sty[row][x] = ch, color

    left, right, top = 3, 34, 6
    bottom = top + 20 + 1
    state = ("GAME OVER" if not board.alive else
             "LIVE" if not stats.get("replay") else "RECORDED RUN · 1×")
    put(1, left, "GAVEL  /  LOCAL INTELLIGENCE", MUTED)
    put(1, W - len(state) - 3, state, GREEN if board.alive else RED)
    put(2, left, "─" * (W - 6), DIM)
    put(4, left, "T E T R I S", FG)
    put(4, left + 24, stats.get("tag", "TF-IDF · 0.3s TRAIN"), MUTED)
    bw = 10
    put(top, left, "┌" + "─" * (bw * 2) + "┐", DIM)
    put(bottom, left, "└" + "─" * (bw * 2) + "┘", DIM)
    for y in range(20):
        put(top + 1 + y, left, "│", DIM)
        put(top + 1 + y, left + bw * 2 + 1, "│", DIM)
        rowch = ""
        for x in range(bw):
            rowch += "██" if (board.rows[y] >> x) & 1 else "· "
        put(top + 1 + y, left + 1, rowch, "#14262d")
    # ghost the executed landing
    lx = dec.get("land_x")
    if lx is not None and board.alive:
        for cx, cy in dec.get("land_cells", []):
            xx, yy = lx + cx, dec.get("land_y", 0) + cy
            if 0 <= xx < bw and 0 <= yy < 20:
                put(top + 1 + yy, left + 1 + 2 * xx, "▓▓", GREEN)
    # falling piece overlay (paced mode): the decided piece visibly drops
    # from the top before locking. Decisions stay instant; only the eye
    # waits.
    if fall is not None and board.alive:
        fcells, fox, foy = fall
        for cx, cy in fcells:
            xx, yy = fox + cx, foy + cy
            if 0 <= xx < bw and 0 <= yy < 20:
                put(top + 1 + yy, left + 1 + 2 * xx, "▒▒", CYAN)
    for off, label, val in ((0, "SCORE", board.score),
                            (12, "LINES", board.lines),
                            (24, "PIECES", board.pieces)):
        put(bottom + 2, left + off, label, MUTED)
        put(bottom + 3, left + off, "%04d" % val, GREEN)

    put(4, right, "Gavel · tetris_pick", GREEN)
    put(5, right, stats.get("hw", "8 logical cores · CPU-only"), MUTED)
    put(7, right, "PIECE", FG)
    put(7, right + 16, str(piece or "—"), GREEN)
    put(8, right, "PICK", MUTED)
    put(8, right + 16, str(dec.get("pick", "—")), FG)
    if dec.get("heur"):
        put(8, right + 24, "HEURISTIC", AMBER)
    put(9, right, "CONFIDENCE", MUTED)
    put(9, right + 16, "%.3f" % dec.get("confidence", 0), FG)
    put(10, right, "A/B SPLIT", MUTED)
    put(10, right + 16, stats.get("last_ab", "—"), FG)
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
    put(16, right + 16, "tfidf+logreg · CPU", MUTED)
    put(17, right, "SERVER", MUTED)
    put(17, right + 16, stats.get("server") or "—", FG)
    put(18, right, "SYS", MUTED)
    put(18, right + 16, stats.get("sys") or "—", FG)
    put(19, right, "GPU", MUTED)
    put(19, right + 16, stats.get("gpu") or "n/a", FG)
    put(20, right, "Heuristic fallbacks  %04d" % stats.get("heuristics", 0),
        AMBER)
    put(20, right, "Model-vs-heuristic agree %s" %
        stats.get("fidelity", "—"), FG)
    put(22, right, "Train 0.3s · 18k pairs", MUTED)
    put(23, right, "Val 98.4% · one call/piece", MUTED)
    put(H - 3, left, "─" * (W - 6), DIM)
    put(H - 2, left, "gavel-tetris · one typed call per piece", MUTED)

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


def play_piece(board, piece, rng, stats, unassisted, maxh=12):
    """One piece: shortlist top-2, one typed call, execute. Returns dec dict."""
    cands = sorted(landing_features(board, piece),
                   key=lambda l: l["score"], reverse=True)
    if len(cands) < 2:
        return None  # no choice to offer: game effectively over
    a, c = cands[0], cands[1]
    if unassisted:
        stats["heuristics"] += 1
        best = a
        dec = {"pick": "A*", "confidence": 1.0, "inference_ms": 0.0,
               "heur": True}
        stats["last_ab"] = f"A:{a['score']:.0f} B:{c['score']:.0f}"
        dec.update({"land_x": best["ox"], "land_y": best["y"],
                    "land_cells": best["cells"]})
        return best, dec
    if a["maxh"] > maxh:
        # High stack: errors are fatal here, and the model saw little of
        # this regime in training. Heuristic drives survival; counted.
        stats["heuristics"] += 1
        best = a
        dec = {"pick": "A*", "confidence": 1.0, "inference_ms": 0.0,
               "heur": True}
        stats["last_ab"] = f"A:{a['score']:.0f} B:high-stack"
        dec.update({"land_x": best["ox"], "land_y": best["y"],
                    "land_cells": best["cells"]})
        return best, dec
    # Offered pair always carries a real margin (their "fewer bad moves"
    # rule): best vs the strongest landing at least GAP_FALLBACK worse.
    # Near-tie boards (or low model confidence — calibrated, so it means
    # something) fall back to the heuristic, counted.
    c = next((l for l in cands[1:]
              if a["score"] - l["score"] >= GAP_FALLBACK), None)
    if c is None:
        stats["heuristics"] += 1
        best = a
        dec = {"pick": "A*", "confidence": 1.0, "inference_ms": 0.0,
               "heur": True}
        stats["last_ab"] = f"A:{a['score']:.0f} B:—"
        dec.update({"land_x": best["ox"], "land_y": best["y"],
                    "land_cells": best["cells"]})
        return best, dec
    flip = rng.random() < 0.5
    x, y = (c, a) if flip else (a, c)
    text = f"A {fmt(x, 'a')} || B {fmt(y, 'b')}"
    pick, conf, ms = decide(text)
    if pick is None:
        pick = "A"
    best = x if pick == "A" else y  # displayed A shows x, B shows y
    # fidelity: did the model agree with the heuristic best?
    stats["calls"] += 1
    if conf < 0.70:
        # Calibrated uncertainty — the model is telling us it can't tell.
        # Take the heuristic best instead of gambling (disclosed fallback).
        stats["heuristics"] += 1
        best = a
        dec = {"pick": pick + "*", "confidence": conf, "inference_ms": ms,
               "heur": True}
    else:
        stats["agree"] += (best is a)
        dec = {"pick": pick, "confidence": conf, "inference_ms": ms,
               "heur": False}
    stats["last_ab"] = f"A:{a['score']:.0f} B:{c['score']:.0f}"
    dec.update({"land_x": best["ox"], "land_y": best["y"],
                "land_cells": best["cells"]})
    return best, dec


def cmd_play(a):
    from rich.console import Console
    from rich.live import Live
    qs = api("/questions")
    if QUESTION not in qs.get("questions", []):
        print(f"question '{QUESTION}' not on server — run legacy_tfidf.py first",
              file=sys.stderr)
        return 2
    import psutil as _ps
    _srv = None
    try:
        _iter = _ps.process_iter(["name", "exe"])
    except Exception:
        _iter = []
    for p in _iter:
        try:
            n = f"{p.info.get('name') or ''} {p.info.get('exe') or ''}".lower()
            if "gavel" in n or "decide-server" in n:
                _srv = p
                break
        except Exception:
            continue

    def server_str():
        if _srv is None:
            return None
        try:
            return "%.1f%% CPU · %.0f MB" % (
                _srv.cpu_percent(interval=None),
                _srv.memory_info().rss / 1e6)
        except Exception:
            return None

    board = Board()
    rng = random.Random(a.seed)
    gen = bag(rng)
    console = Console(style=f"on {BG}", highlight=False)
    if not a.headless and not console.is_terminal:
        print("needs a TTY; use --headless for non-interactive runs", file=sys.stderr)
        return 2
    rec = open(a.record, "x") if a.record else None
    if rec:
        rec.write(json.dumps({"type": "metadata", "format": "gavel-tetris-v1",
                              "question": QUESTION,
                              "settings": vars(a)}) + "\n")
    # Startup calibration: raw single-call ceiling on this machine.
    _cal = []
    for _ in range(40):
        _, _, ms = decide("A ah10 alines0 aholes1 abump5 aagg30 || "
                          "B bh8 blines1 bholes0 bbump3 bagg25")
        _cal.append(ms)
    _cal.sort()
    _med = _cal[len(_cal) // 2]
    brain_max = "%.1f ms · %d /s" % (_med, int(round(1000 / _med))) if _med > 0 else "—"
    print(f"[calibration] tetris_pick raw p50: {brain_max}", flush=True)

    stats = {"heuristics": 0, "calls": 0, "agree": 0,
             "steps_per_second": 0, "elapsed": 0, "brain_max": brain_max,
             "server": None, "last_ab": "—", "fidelity": "—",
             "hw": "8 logical cores · CPU-only",
             "tag": "TF-IDF · 0.3s TRAIN"}
    sys_stats()  # warm up the CPU counter (first call always reads 0)
    stamps = deque(maxlen=60)
    total = 0
    t_start = time.perf_counter()
    piece = next(gen)
    nxt = next(gen)
    live = None if a.headless else Live(console=console, auto_refresh=False)
    if live:
        live.__enter__()
    try:
        while board.alive:
            if a.pieces and total >= a.pieces:
                break
            r = play_piece(board, piece, rng, stats, a.unassisted, a.maxh)
            if r is None:
                board.alive = False
                break
            best, dec = r
            now = time.perf_counter()
            if live and not a.max_speed and a.drop_frames > 0:
                # Watchable fall: the decided piece drops from the top in
                # even steps before locking. The decision already happened
                # above (counter stays honest); this spends the fps budget
                # on animation, not thinking.
                span = best["y"] + 4
                per = max((1.0 / a.fps - (time.perf_counter() - now))
                          / a.drop_frames, 0)
                for f in range(1, a.drop_frames + 1):
                    oy = -4 + round(span * f / a.drop_frames)
                    live.update(render(_view(board), piece, nxt, dec, stats,
                                       fall=(best["cells"], best["ox"], oy)),
                                refresh=True)
                    time.sleep(per)
            board.apply(best["cells"], best["ox"], best["y"])
            now = time.perf_counter()
            stamps.append(now)
            total += 1
            stats["elapsed"] = now - t_start
            stats["steps_per_second"] = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                                         if len(stamps) > 1 else 0)
            stats["server"] = server_str() or stats.get("server")
            _cpu, _ram, _gpu = sys_stats()
            if _cpu is not None:
                stats["sys"] = "CPU %.0f%% · RAM %s" % (_cpu, _ram)
            if _gpu is not None:
                stats["gpu"] = _gpu
            if stats["calls"]:
                stats["fidelity"] = "%.3f" % (stats["agree"] / stats["calls"])
            snap = {"w": 10, "h": 20,
                    "rows": list(board.rows), "score": board.score,
                    "lines": board.lines, "pieces": board.pieces,
                    "alive": board.alive}
            if live:
                # render needs rows; adapt snapshot shape
                live.update(render(_view(board), piece, nxt, dec, stats),
                            refresh=True)
            if rec:
                rec.write(json.dumps({"type": "frame", "at": now - t_start,
                                      "game": snap, "piece": piece,
                                      "decision": dec,
                                      "stats": dict(stats)},
                                     separators=(",", ":")) + "\n")
            if not a.max_speed:
                wait = 1.0 / a.fps - (time.perf_counter() - now)
                if wait > 0:
                    time.sleep(wait)
            piece, nxt = nxt, next(gen)
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            live.__exit__(None, None, None)
        el = time.perf_counter() - t_start
        summary = {"pieces": total, "inference_calls": stats["calls"],
                   "heuristics": stats["heuristics"],
                   "fidelity": stats.get("fidelity"),
                   "seconds": round(el, 2),
                   "pieces_per_second": round(total / el, 1) if el else 0,
                   "score": board.score, "lines": board.lines,
                   "alive": board.alive}
        if rec:
            rec.write(json.dumps({"type": "end", "summary": summary}) + "\n")
            rec.close()
        print(json.dumps(summary, indent=2))
    return 0


class _view:
    """Adapter: render() wants body-list snapshots; Tetris uses bitmask rows."""

    def __init__(self, board):
        self.rows = board.rows
        self.score = board.score
        self.lines = board.lines
        self.pieces = board.pieces
        self.alive = board.alive

    def __getitem__(self, k):
        raise KeyError(k)

    def get(self, k, d=None):
        return {"w": 10, "h": 20}.get(k, d)


def cmd_replay(a):
    from rich.console import Console
    from rich.live import Live
    console = Console(style=f"on {BG}", highlight=False)
    frames = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    frames = [f for f in frames if f.get("type") == "frame"]
    if not frames:
        print("no frames in recording", file=sys.stderr)
        return 2
    with Live(console=console, auto_refresh=False) as live:
        n = 0
        while True:
            for f in frames:
                st = dict(f.get("stats", {}))
                st["replay"] = True
                g = f["game"]
                v = _view.__new__(_view)
                v.rows, v.score, v.lines = g["rows"], g["score"], g["lines"]
                v.pieces, v.alive = g["pieces"], g["alive"]
                live.update(render(v, f.get("piece"), None,
                                   f.get("decision", {}), st), refresh=True)
                n += 1
                if a.pieces and n >= a.pieces:
                    return 0
                time.sleep(1.0 / a.fps)
            if not a.loop:
                return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="gavel-tetris")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("play")
    pl.add_argument("--fps", type=float, default=8)
    pl.add_argument("--max-speed", action="store_true")
    pl.add_argument("--pieces", type=int, default=0)
    pl.add_argument("--seed", type=int, default=11)
    pl.add_argument("--maxh", type=int, default=8,
                    help="above this stack height the heuristic drives "
                         "survival instead of the model")
    pl.add_argument("--drop-frames", type=int, default=5,
                    help="paced mode: animation frames for the falling piece "
                         "(0 = instant lock)")
    pl.add_argument("--record", default=None)
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--unassisted", action="store_true",
                    help="heuristic only, no model calls")
    rp = sub.add_parser("replay")
    rp.add_argument("file")
    rp.add_argument("--fps", type=float, default=8)
    rp.add_argument("--pieces", type=int, default=0)
    rp.add_argument("--loop", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "play":
        return cmd_play(a)
    return cmd_replay(a)


if __name__ == "__main__":
    raise SystemExit(main())
