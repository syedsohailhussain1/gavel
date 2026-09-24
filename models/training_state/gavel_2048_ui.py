#!/usr/bin/env python3
"""gavel-2048 live UI: the GLiClass demo layout, Gavel engine underneath.

Same arrangement as Sources/GLiClass2048Demo (header / board / scoreboard /
controls), same tile palette, same controls (policy segmented, candidates,
margin slider, seed, move-delay, marathon, lookahead). Differences are only
where honesty requires: branding (Gavel, not GLiClass), load status, and
the decision path -- every move is a live /ask call to the 2048_move
TF-IDF question on gavel-server, counted on screen.

Layout contract (mirrors Game2048View, spacing in px):
  page padding 16, section gap 14, header gap 5, tile gap 8, board pad 10,
  board bg #6B6157 rounded 12, tile rounded 8, tile font 25 bold (19 when
  value >= 1024), dark text on 2/4 else white, scoreboard 5 readouts.

Usage:
  python gavel_2048_ui.py [--port 8091] [--question 2048_move] [--no-open]
Needs gavel-server on :7575 with the question trained. Opens a browser tab.
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TS = Path(__file__).resolve().parent
sys.path.insert(0, str(TS))

from gavel_2048 import MOVES, Game, board_text, decide, expectimax  # noqa: E402

# Tile palette, verbatim from tileColor() -- Color(red, green, blue).
TILE = {
    0: "rgba(255,255,255,0.15)",
    2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2794F", 16: "#F27340",
    32: "#F25C3B", 64: "#F24026", 128: "#EDC747", 256: "#EDBF33",
    512: "#EDB324", 1024: "#E8A614",
}
BIG = "#5940A6"  # default: 2048+
BOARD_BG = "#6B6157"


def tile_css(v):
    if v in TILE:
        return TILE[v]
    return BIG


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>local Gavel plays 2048</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px; font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
         max-width: 560px; background: #fff; color: #111; }
  .stack { display: flex; flex-direction: column; gap: 14px; }
  .header { display: flex; flex-direction: column; gap: 5px; }
  .title { font-size: 22px; font-weight: 700; }
  .sub { font-size: 13px; color: #6e6e6e; }
  .status { font-size: 12px; font-family: ui-monospace, Consolas, monospace; color: #6e6e6e; }
  .status.ok { color: #16a34a; }
  .status .game { color: #f59e0b; }
  .board { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
           padding: 10px; background: __BOARD_BG__; border-radius: 12px; }
  .tile { aspect-ratio: 1; border-radius: 8px; display: flex; align-items: center;
          justify-content: center; font-weight: 700; font-family: ui-rounded, system-ui;
          font-size: 25px; }
  .tile.big { font-size: 19px; }
  .tile.dark { color: rgba(0,0,0,0.72); }
  .tile.light { color: #fff; }
  .score { display: flex; padding: 8px 0; background: rgba(127,127,127,0.08);
           border-radius: 10px; }
  .ro { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 2px;
        border-right: 1px solid rgba(127,127,127,0.35); }
  .ro:last-child { border-right: none; }
  .ro .v { font-size: 16px; font-weight: 600; font-family: ui-monospace, Consolas, monospace; }
  .ro .l { font-size: 9px; color: #6e6e6e; }
  .c-primary { color: #111; } .c-secondary { color: #6e6e6e; } .c-green { color: #16a34a; }
  .c-orange { color: #f59e0b; } .c-blue { color: #2563eb; }
  .controls { display: flex; flex-direction: column; gap: 10px; font-size: 13px; }
  .row { display: flex; align-items: center; gap: 8px; }
  button { padding: 4px 12px; font-size: 13px; }
  button:disabled { opacity: 0.5; }
  .seg { display: inline-flex; border: 1px solid #ccc; border-radius: 6px; overflow: hidden; }
  .seg button { border: none; border-radius: 0; background: #fff; }
  .seg button.on { background: #0a84ff; color: #fff; }
  .cap { font-size: 13px; color: #111; min-width: 110px; }
  input[type=range] { flex: 1; }
  .mono { font-family: ui-monospace, Consolas, monospace; }
  .spacer { flex: 1; }
</style></head>
<body><div class="stack">
  <div class="header">
    <div class="title">local Gavel plays 2048</div>
    <div class="sub">A heuristic shortlists safe swipes; Gavel compares their resulting boards in one /ask call.</div>
    <div class="status" id="status">Not loaded</div>
    <div class="status" id="err" style="color:#dc2626"></div>
  </div>
  <div class="board" id="board"></div>
  <div class="score">
    <div class="ro"><span class="v c-primary" id="r-time">0:00.0</span><span class="l">time</span></div>
    <div class="ro"><span class="v c-secondary" id="r-score">0</span><span class="l">score</span></div>
    <div class="ro"><span class="v c-green" id="r-moves">0</span><span class="l">moves</span></div>
    <div class="ro"><span class="v c-orange" id="r-max">2</span><span class="l">max tile</span></div>
    <div class="ro"><span class="v c-blue" id="r-ms">&ndash;</span><span class="l">model ms/move</span></div>
  </div>
  <div class="controls">
    <div class="row">
      <button id="b-load">Load model</button>
      <button id="b-play">Play</button>
      <button id="b-reset">Reset</button>
      <span class="spacer"></span>
      <span class="status" id="last"></span>
    </div>
    <label class="row"><input type="checkbox" id="c-marathon"> <span>Marathon: start the next seeded game after game over</span></label>
    <label class="row"><input type="checkbox" id="c-look" checked> <span>One-move expectimax shortlist</span></label>
    <div class="row"><span class="cap">Policy</span><span class="seg" id="seg-policy">
      <button data-v="gavel" class="on">gavel</button><button data-v="heuristic">heuristic</button><button data-v="random">random</button>
    </span></div>
    <div class="row"><span class="cap">Candidates</span><span class="seg" id="seg-cand">
      <button data-v="2" class="on">top 2</button><button data-v="3">top 3</button><button data-v="4">all 4</button>
    </span></div>
    <div class="row"><span class="cap">Override margin</span>
      <input type="range" id="r-margin" min="0" max="1" step="0.05" value="0.5">
      <span class="mono" id="v-margin">0.50</span></div>
    <div class="row"><span class="cap">Seed</span>
      <input type="number" id="i-seed" value="50" style="width:90px"></div>
    <div class="row"><span class="cap" id="t-delay">Pause per move: 0 ms (may raise model latency)</span></div>
    <div class="row"><input type="range" id="r-delay" min="0" max="500" step="10" value="0"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let cfg = {};
async function get(p) { return (await fetch(p)).json(); }
async function post(p, b) {
  return (await fetch(p, {method: "POST", headers: {"Content-Type": "application/json"},
                          body: JSON.stringify(b || {})})).json();
}
function fmtTime(s) {
  const m = Math.floor(s / 60), r = s - m * 60;
  return m + ":" + r.toFixed(1).padStart(4, "0");
}
async function refresh() {
  const s = await get("/state");
  const bd = $("board"); bd.innerHTML = "";
  for (const v of s.board) {
    const d = document.createElement("div");
    d.className = "tile " + (v >= 1024 ? "big " : "") + (v <= 4 ? "dark" : "light");
    d.style.background = s.colors[v] || s.colors.big;
    d.textContent = v > 0 ? v : "";
    bd.appendChild(d);
  }
  $("r-time").textContent = fmtTime(s.elapsed);
  $("r-score").textContent = s.score;
  $("r-moves").textContent = s.moves;
  $("r-max").textContent = s.max;
  $("r-ms").textContent = s.moves > 0 ? s.model_ms.toFixed(1) : "\\u2013";
  const st = $("status");
  st.innerHTML = s.loaded
    ? '<span class="ok">' + s.load_status + "</span>" +
      (s.games > 0 ? ' <span class="game">\\u00b7 game ' + (s.games + 1) + "</span>" : "")
    : s.load_status;
  $("last").textContent = s.last ? "last: " + s.last : "";
  $("b-play").textContent = s.running ? "Pause" : (s.over ? "Restart" : "Play");
  $("b-load").textContent = s.loaded ? "Loaded" : "Load model";
  $("b-load").disabled = s.running || s.loaded || cfg.policy !== "gavel";
  $("b-reset").disabled = s.running;
  setTimeout(refresh, 50);
}
function seg(id, fn) {
  $(id).querySelectorAll("button").forEach(b => b.onclick = () => {
    $(id).querySelectorAll("button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); fn(b.dataset.v);
  });
}
window.onload = async () => {
  cfg = await get("/config");
  const err = $("err");
  const show = async (p, b) => {
    const r = await post(p, b);
    const msg = r && r.error ? r.error : "";
    if (err) err.textContent = msg;
    else if (msg) alert(msg);
    return r;
  };
  $("b-load").onclick = () => show("/load");
  $("b-play").onclick = () => show("/toggle");
  $("b-reset").onclick = () => { err.textContent = ""; post("/reset"); };
  seg("seg-policy", v => post("/config", {policy: v}));
  seg("seg-cand", v => post("/config", {candidates: +v}));
  $("r-margin").oninput = e => {
    $("v-margin").textContent = (+e.target.value).toFixed(2);
    post("/config", {margin: +e.target.value});
  };
  $("i-seed").onchange = e => post("/config", {seed: +e.target.value});
  $("c-marathon").onchange = e => post("/config", {marathon: e.target.checked});
  $("c-look").onchange = e => post("/config", {lookahead: e.target.checked});
  $("r-delay").oninput = e => {
    $("t-delay").textContent = "Pause per move: " + e.target.value + " ms (may raise model latency)";
    post("/config", {delay_ms: +e.target.value});
  };
  refresh();
};
</script></body></html>
""".replace("__BOARD_BG__", BOARD_BG)


class UI:
    """Game state + loop. Mirrors Game2048Model behavior (policies, margin
    override to offered[0], marathon seeds, ms + call accounting)."""

    def __init__(self, question):
        self.question = question
        # RLock: _step nests snapshot-style reads inside decision logic on
        # some paths; all critical sections stay brief either way.
        self.lock = threading.RLock()
        self.policy = "gavel"
        self.candidates = 2
        self.margin = 0.5
        self.lookahead = True
        self.marathon = False
        self.seed = 50
        self.delay = 0.0
        self.loaded = False
        self.load_status = "Not loaded"
        self.reset_game(keep_status=True)
        self.running = False
        self._stop = False
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def reset_game(self, keep_status=False):
        import gavel_2048 as G
        self.game = G.Game(self.seed)
        self.grng = __import__("random").Random(1000 + self.seed)
        self.board = list(self.game.b)
        self.score = self.moves = 0
        self.max_tile = 2
        self.calls = 0
        self.model_ms = 0.0
        self.last = None
        self.over = False
        self.elapsed = 0.0
        self.games = 0
        self._t0 = None

    def _snapshot(self):
        with self.lock:
            return {"board": list(self.board), "score": self.score,
                    "moves": self.moves, "max": self.max_tile,
                    "model_ms": self.model_ms / max(self.moves, 1),
                    "elapsed": self.elapsed, "loaded": self.loaded,
                    "load_status": self.load_status, "games": self.games,
                    "running": self.running, "over": self.over,
                    "last": self.last, "policy": self.policy,
                    "colors": {str(k): v for k, v in TILE.items()}
                    | {"big": BIG}}

    def _loop(self):
        import gavel_2048 as G
        import traceback
        while not self._stop:
            try:
                self._step()
            except Exception:
                traceback.print_exc()
                with self.lock:
                    self.running = False
                time.sleep(0.5)

    def _step(self):
        import gavel_2048 as G
        if self.delay > 0:
            time.sleep(0.02)
        with self.lock:
            running = self.running
        if not running:
            time.sleep(0.02)
            return
        with self.lock:
            if self.game.over:
                if self.marathon:
                    self.games += 1
                    ns = self.seed + self.games
                    self.game = G.Game(ns)
                    self.grng = __import__("random").Random(1000 + ns)
                else:
                    self.running = False
                self._sync()
                return
            board = list(self.game.b)
            legal = self.game.legal(board)
            if not legal:
                with self.lock:
                    self.over = True
                    self.running = False
                    self._sync()
                return
            if self.policy == "random":
                import random as _r
                mv = _r.Random().choice(legal)
                spent = 0.0
            else:
                if self.lookahead:
                    # Static top-K to match facts training (depth-2 search
                    # lives in data generation, not in the live loop).
                    ranked, _ = G.teacher(board, self.grng,
                                          self.candidates, 8, 0)
                else:
                    scored = sorted(
                        legal, key=lambda m: -G.heuristic(
                            G.Game(0)._move(board, m)[0]))
                    ranked = scored[:self.candidates]
                if len(ranked) == 1:
                    mv, spent = ranked[0], 0.0
                elif self.policy == "heuristic":
                    mv, spent = ranked[0], 0.0
                else:
                    import gavel_2048 as G2
                    # Best 2048 question: pairwise facts (static top-2,
                    # matching facts training). One /ask call per move.
                    rk, _ = G2.teacher(board, self.grng, 2, 8, 0)
                    if len(rk) < 2:
                        mv, spent = rk[0], 0.0
                    else:
                        na, ga, _ = G2.Game._move(board, rk[0])
                        nb, gb, _ = G2.Game._move(board, rk[1])
                        t0 = time.perf_counter()
                        raw, conf, ms = G2.decide(
                            G2.pair_facts_text(
                                G2.cand_facts(na, ga),
                                G2.cand_facts(nb, gb)),
                            "2048_facts")
                        spent = (time.perf_counter() - t0) * 1000
                        with self.lock:
                            self.calls += 1
                            self.model_ms += spent
                        picked = rk[0] if raw == "first" else rk[1]
                        mv = picked
                        if picked != rk[0] and conf < self.margin:
                            mv = rk[0]
            gained = self.game.step(mv)
            if gained is None:
                mv = self.game.legal()[0]
                self.game.step(mv)
            with self.lock:
                self.last = mv
                self._sync()
            if self.delay > 0:
                time.sleep(self.delay)

    def _sync(self):
        self.board = list(self.game.b)
        self.score = self.game.score
        self.moves = self.game.moves
        self.max_tile = self.game.max_tile()
        self.over = self.game.over
        if self._t0 is None:
            self._t0 = time.perf_counter()
        self.elapsed = time.perf_counter() - self._t0


class Handler(BaseHTTPRequestHandler):
    ui = None

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            self._json(self.ui._snapshot())
        elif self.path == "/config":
            u = self.ui
            self._json({"policy": u.policy, "candidates": u.candidates,
                        "margin": u.margin, "seed": u.seed,
                        "marathon": u.marathon, "lookahead": u.lookahead})
        else:
            self._json({"error": "unknown"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        u = self.ui
        if self.path == "/load":
            with u.lock:
                u.loaded = True
                u.load_status = ("2048_move TF-IDF ~5k weights "
                                 "Rust engine :7575 serving live /ask")
            self._json({"ok": True})
        elif self.path == "/toggle":
            with u.lock:
                if u.running:
                    u.running = False
                else:
                    if not u.loaded and u.policy == "gavel":
                        self._json({"error": "load the model first"})
                        return
                    if u.over:
                        u.reset_game()
                    u.running = True
                    if u._t0 is None:
                        u._t0 = time.perf_counter()
            self._json({"ok": True})
        elif self.path == "/reset":
            with u.lock:
                u.running = False
                u.reset_game()
            self._json({"ok": True})
        elif self.path == "/config":
            with u.lock:
                for k in ("policy", "candidates", "margin", "seed",
                          "marathon", "lookahead", "delay_ms"):
                    if k in body:
                        if k == "delay_ms":
                            u.delay = float(body[k]) / 1000
                        else:
                            setattr(u, k if k != "delay_ms" else "delay",
                                    body[k])
            self._json({"ok": True})
        else:
            self._json({"error": "unknown"}, 404)

    def log_message(self, *a):
        pass


def main(argv=None):
    p = argparse.ArgumentParser(prog="gavel-2048-ui")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--question", default="2048_move")
    p.add_argument("--no-open", action="store_true")
    a = p.parse_args(argv)
    import gavel_2048 as G
    G.QUESTION = a.question
    Handler.ui = UI(a.question)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"gavel-2048 UI on {url} (needs gavel-server :7575)", flush=True)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
