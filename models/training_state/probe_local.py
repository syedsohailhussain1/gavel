#!/usr/bin/env python3
"""Local-only probe rerun (post-augmentation) with energies. No Jev calls."""
import json
import os
import time
import urllib.request

TS = "D:/gavel/models/training_state/"
_src = open(os.path.join(TS, "negation_probe.py"), encoding="utf-8").read()
_ns: dict = {}
exec(_src[_src.index("def E("):_src.index("def g_ask")], _ns)
PROBE = _ns["PROBE"]

BASE = "http://127.0.0.1:7575"


def ask(q, text, timeout=30):
    body = json.dumps({"question": q, "input": text}).encode()
    req = urllib.request.Request(
        BASE + "/ask", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def serialize(e):
    return (f"from: {e['from']} sender: {e['sender']} subject: {e['subject']} "
            f"link_text: {e['link_display_text']} link_url: {e['link_url']} body: {e['body']}")


for q in ("phishing_tfidf", "phishing"):
    correct, flips_ok, flips = 0, 0, 0
    print(f"===== {q} =====", flush=True)
    for e in PROBE:
        try:
            r, ms = ask(q, serialize(e))
            got = (r.get("output") or {}).get("label", "?")
            ok = got == e["expect"]
            correct += ok
            if e["flip"]:
                flips += 1
                flips_ok += ok
            mark = "OK " if ok else "MISS"
            print(f"[{mark} {e['id']}] want={e['expect']} got={got} "
                  f"conf={r.get('confidence', 0):.3f} act={r.get('action')} "
                  f"energy={r.get('energy')}", flush=True)
        except Exception as ex:
            print(f"[SKIP {e['id']}] {type(ex).__name__}", flush=True)
    print(f"{q}: {correct}/16 flips={flips_ok}/{flips}", flush=True)
print("DONE", flush=True)
