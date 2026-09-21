#!/usr/bin/env python3
"""Eval gate: a candidate question promotes to production ONLY if all green.
Checks: probe 16/16 labels, aug_val >= 15/16, held-out test sample >= 0.985.
Usage: python eval_gate.py <candidate>   (exit 0 = PROMOTE, 1 = BLOCK)
On pass: writes manifest production pointer. No Jev calls (local only).
"""
import json
import os
import sys
import urllib.request

BASE = "http://127.0.0.1:7575"
TS = "D:/gavel/models/training_state/"
CAND = sys.argv[1] if len(sys.argv) > 1 else "phishing_tfidf"

_src = open(os.path.join(TS, "negation_probe.py"), encoding="utf-8").read()
_ns: dict = {}
exec(_src[_src.index("def E("):_src.index("def g_ask")], _ns)
PROBE = _ns["PROBE"]


def api(p, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + p, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(text, timeout=20):
    return api("/ask", {"question": CAND, "input": text}, timeout=timeout)


def serialize(e):
    return (f"from: {e['from']} sender: {e['sender']} subject: {e['subject']} "
            f"link_text: {e['link_display_text']} link_url: {e['link_url']} body: {e['body']}")


fails = []


def check(name, items, need, inp_of, lab_of):
    ok, skip = 0, 0
    for x in items:
        try:
            r = ask(inp_of(x), timeout=20)
            if (r.get("output") or {}).get("label") == lab_of(x):
                ok += 1
        except Exception:
            skip += 1
    done = len(items) - skip
    rate = ok / max(done, 1)
    verdict = "PASS" if ok >= need and skip == 0 else "FAIL"
    print(f"[{verdict}] {name}: {ok}/{done} (need {need}) skipped={skip}", flush=True)
    if verdict == "FAIL":
        fails.append(name)
    return rate


check("probe", PROBE, 16, serialize, lambda e: e["expect"])
aug_val = json.load(open(TS + "negation_aug_val.json"))
check("aug_val", aug_val, 15, lambda x: x["input"], lambda x: x["label"])
te = [json.loads(l) for l in
      open("C:/Users/Sohail/AppData/Local/Temp/opencode/gavel-phish/test.jsonl",
           encoding="utf-8")][:150]
ok = skip = 0
for x in te:
    try:
        r = ask(x["input"], timeout=20)
        if (r.get("output") or {}).get("label") == x["label"]:
            ok += 1
    except Exception:
        skip += 1
done = len(te) - skip
rate = ok / max(done, 1)
v = "PASS" if rate >= 0.985 and skip == 0 else "FAIL"
print(f"[{v}] test150: {ok}/{done}={rate:.4f} (need >=0.985) skipped={skip}", flush=True)
if v == "FAIL":
    fails.append("test150")

if fails:
    print(f"GATE BLOCKED: {fails}", flush=True)
    raise SystemExit(1)
man_path = TS + "manifest.json"
man = json.loads(open(man_path).read()) if __import__("os").path.exists(man_path) else {}
man["production_phishing_tfidf"] = CAND
json.dump(man, open(man_path, "w"), indent=1)
print(f"GATE GREEN — production -> {CAND}", flush=True)
