#!/usr/bin/env python3
"""Run JevBench public items through OUR system using THEIR runner.

Uses jevbench.runner.Runner + Ledger directly (no edits to their repo):
canonical results.jsonl + raw evidence, then their summarize for official
numbers. Subsettable for dev (--limit), full public for milestones.

Usage:
  python run_jevbench.py [tasks-csv] [outdir] [--limit N]
Defaults: easy+original+hard(public) ./JB_RUN
Ledger cap $25 (local run bills nothing; cap is a formality).
Raw evidence + ledger live OUTSIDE both repos (their house rule).
"""
import json
import os
import sys
import time
import types as _t

# Windows has no fcntl; jevbench.budget uses it only for multi-process file
# locking. Single-process runs stub it out (our WinLedger keeps no locks).
_f = _t.ModuleType("fcntl")
_f.LOCK_EX, _f.LOCK_UN, _f.LOCK_SH, _f.LOCK_NB = 2, 8, 1, 4
_f.flock = lambda *a, **k: None
_f.lockf = lambda *a, **k: None
sys.modules.setdefault("fcntl", _f)

sys.path.insert(0, r"D:\jevbench")
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())

from jevbench.runner import Runner  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

from gavel_adapter import GavelLocalAdapter  # noqa: E402


class WinLedger:
    """Minimal duck-type of jevbench.budget.Ledger (which needs Unix fcntl).
    Single-process runs only — no file locking. Same JSON-lines shape."""

    def __init__(self, path, cap_usd=25):
        self.path = path
        self.cap = cap_usd
        self.spent = 0.0
        self.n = 0
        open(path, "a", encoding="utf-8").close()

    def _write(self, obj):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    def reserve(self, amount, info):
        self.n += 1
        rid = f"r{self.n}"
        self._write({"op": "reserve", "rid": rid, "amount": amount,
                     "info": info})
        return rid

    def settle(self, rid, cost, info):
        self.spent += cost or 0.0
        self._write({"op": "settle", "rid": rid, "cost": cost,
                     "spent": self.spent, "info": info})

TASKS = (sys.argv[1] if len(sys.argv) > 1
         else r"D:\jevbench\datasets\public\easy.jsonl,"
              r"D:\jevbench\datasets\public\original.jsonl,"
              r"D:\jevbench\datasets\public\hard.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\Sohail\AppData\Local\Temp\opencode\JB_RUN"
limit = None
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])

os.makedirs(OUT, exist_ok=True)
tasks = []
for p in TASKS.split(","):
    tasks.extend(load_jsonl(p))
if limit:
    tasks = tasks[:limit]
print(f"tasks={len(tasks)} -> {OUT}", flush=True)

adapter = GavelLocalAdapter()
t0 = time.perf_counter()
adapter.load()
print(f"[warm load] {time.perf_counter()-t0:.0f}s", flush=True)
ledger = WinLedger(os.path.join(OUT, "ledger.json"), cap_usd=25)
runner = Runner(adapter, ledger,
                raw_dir=os.path.join(OUT, "raw"),
                default_reserve_usd=0.0)
t0 = time.perf_counter()
records = runner.run_all(tasks, results_path=os.path.join(OUT, "results.jsonl"))
el = time.perf_counter() - t0
ok = sum(1 for r in records if r.get("correct"))
tot = sum(1 for r in records if r.get("status") == "ok")
print(f"done: {ok}/{tot} correct in {el/60:.1f} min "
      f"({el/max(tot,1):.1f}s/decision)", flush=True)
print(f"results: {os.path.join(OUT, 'results.jsonl')}", flush=True)
print("score officially with: "
      "python -m jevbench.cli summarize --tasks <same> --results <results>",
      flush=True)
