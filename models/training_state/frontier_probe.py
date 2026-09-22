#!/usr/bin/env python3
"""Frontier probe: OpenAI-format API (DeepSeek) as Snake reference brain.

- eval: N rollout states -> move choice; measures latency p50/p99,
  agreement vs guarded teacher, tokens + cost. Key via DEEPSEEK_API_KEY
  env only (never disk, never logged).
- stress: concurrent clients hammering the endpoint for throughput,
  error rate, and p99 under load.

Usage:
  python frontier_probe.py eval [n] [model]
  python frontier_probe.py stress [clients] [calls_each] [model]
Defaults: eval 50 deepseek-flash | stress 8 40 deepseek-flash
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from gavel_snake import DELTA, MOVES, Game, pick_move, teacher

URL = "https://api.deepseek.com/chat/completions"


def _main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        model = sys.argv[3] if len(sys.argv) > 3 else "deepseek-flash"
        cmd_eval(n, model)
    elif cmd == "stress":
        clients = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        each = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        model = sys.argv[4] if len(sys.argv) > 4 else "deepseek-flash"
        cmd_stress(clients, each, model)


def call(state_text, model, timeout=120):
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set (env only)")
    payload = {"model": model,
               "messages": [{"role": "user", "content": state_text}],
               "temperature": 0, "max_tokens": 500}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    ms = (time.perf_counter() - t0) * 1000
    try:
        msg = out["choices"][0]["message"]
        # The final answer lives in content; reasoning_content is
        # deliberation (it mentions every option, so last-mention parsing
        # misreads it). Fall back to reasoning only if content is empty.
        for blob in ((msg.get("content") or ""),
                     (msg.get("reasoning_content") or "")):
            found = [m for m in MOVES if m in blob.lower()]
            if found:
                pick = found[-1]
                break
        else:
            pick = None
    except (KeyError, IndexError):
        return None, ms, {}, out
    return pick, ms, out.get("usage", {}), None


def describe(game):
    hx, hy = game.snake[0]
    ax, ay = game.apple
    body = list(game.snake)
    occ = set(body)
    segs = ", ".join(f"({x},{y})" for x, y in body[:14])
    if len(body) > 14:
        segs += f", ...({len(body) - 14} more)"
    legal = []
    for m, (dx, dy) in DELTA.items():
        t = (hx + dx, hy + dy)
        if (0 <= t[0] < game.w and 0 <= t[1] < game.h) and t not in occ:
            legal.append(m.upper())
    return (f"You are playing Snake on a {game.w}x{game.h} board. "
            f"Head at ({hx},{hy}), food at ({ax},{ay}). "
            f"Body segments (head first): {segs}. "
            f"Legal moves (not wall, not body): {', '.join(legal)}. "
            f"Move toward the food without dying. Reply with exactly one "
            f"word: UP, DOWN, LEFT or RIGHT.")


def cmd_eval(n, model):
    rng_states = []
    seed = 7
    g = Game(12, 12, seed)
    heading = "right"
    while len(rng_states) < n:
        hx, hy = g.snake[0]
        danger = g.danger()
        ax, ay = g.apple[0] - hx, g.apple[1] - hy
        raw = teacher(ax, ay, danger)
        final, _ = pick_move(g, ax, ay, danger, raw, False)
        rng_states.append((describe(g), final))
        g.step(final)
        if not g.alive:
            seed += 1
            g = Game(12, 12, seed)
    ok = skip = 0
    lat, tok_in, tok_out = [], 0, 0
    for i, (text, want) in enumerate(rng_states):
        try:
            got, ms, usage, err = call(text, model)
        except Exception as e:
            print(f"[{i}] ERROR {str(e)[:120]}", flush=True)
            skip += 1
            continue
        if err is not None:
            print(f"[{i}] API-ERROR {json.dumps(err)[:200]}", flush=True)
            skip += 1
            continue
        lat.append(ms)
        tok_in += int(usage.get("prompt_tokens", 0))
        tok_out += int(usage.get("completion_tokens", 0))
        hit = (got == want)
        ok += hit
        print(f"[{i}] got={got} want={want} {'HIT' if hit else 'miss'} {ms:.0f}ms",
              flush=True)
    lat.sort()
    done = len(lat)
    print(json.dumps({
        "model": model, "n": n, "agree": f"{ok}/{done}",
        "acc": round(ok / max(done, 1), 4), "skipped": skip,
        "p50_ms": round(lat[done // 2], 1) if done else None,
        "p99_ms": round(lat[int(done * 0.99)], 1) if done else None,
        "tokens_in": tok_in, "tokens_out": tok_out,
    }, indent=1))


def cmd_stress(clients, each, model):
    g = Game(12, 12, 7)
    text = describe(g)

    def one(_):
        try:
            got, ms, _, err = call(text, model)
            return ms, err is None and got is not None
        except Exception:
            return None, False

    t0 = time.perf_counter()
    lat, ok, fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=clients) as ex:
        for ms, good in ex.map(one, range(clients * each)):
            if good:
                ok += 1
                lat.append(ms)
            else:
                fail += 1
    el = time.perf_counter() - t0
    lat.sort()
    print(json.dumps({
        "model": model, "clients": clients, "calls_each": each,
        "ok": ok, "fail": fail, "seconds": round(el, 1),
        "per_sec": round(ok / max(el, 1e-9), 1),
        "p50_ms": round(lat[len(lat) // 2], 1) if lat else None,
        "p99_ms": round(lat[int(len(lat) * 0.99)], 1) if lat else None,
    }, indent=1))

if __name__ == "__main__":
    _main()
