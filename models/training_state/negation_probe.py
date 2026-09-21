#!/usr/bin/env python3
"""Contrastive negation probe: Jev Live vs local TF-IDF (+logistic ref).
8 pairs (base + negated counterpart). Key via TYPESAFE_API_KEY env only.
Run: $env:TYPESAFE_API_KEY='<key>'; python -u models/training_state/negation_probe.py
"""
import json
import os
import time
import urllib.request

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import GAVEL_URL as GAVEL, TS
TS = str(TS)
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
VERDICT = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "This email arrived in the user's inbox and contains a link (`link_url`). "
            "Is it a phishing attempt, or a legitimate message whose link is safe to click?"),
        "criteria": {
            "phishing": "The email or its link is a phishing attempt or otherwise malicious. The user should not click the link.",
            "legitimate": "A normal, legitimate message. The link is safe to click."}}}


from probe_items import E, PROBE


def g_ask(question, text, timeout=30):
    body = json.dumps({"question": question, "input": text}).encode()
    req = urllib.request.Request(
        GAVEL + "/ask", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def serialize(e):
    return (f"from: {e['from']} sender: {e['sender']} subject: {e['subject']} "
            f"link_text: {e['link_display_text']} link_url: {e['link_url']} body: {e['body']}")


def jev_ask(state):
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set")
    payload = {"state": {k: state[k] for k in
                         ("from", "sender", "subject", "body",
                          "link_display_text", "link_url")},
               "model": JEV_MODEL, "questions": VERDICT}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        JEV_URL, data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            out = json.loads(r.read())
    except Exception as e:
        return {"error": str(e)[:200]}, -1
    ms = (time.perf_counter() - t0) * 1000
    try:
        v = out["answers"]["verdict"]
        return {"choice": v["choice"], "probabilities": v.get("probabilities"),
                "confidence": round(v.get("confidence", 0), 3),
                "usage": out.get("usage", {})}, ms
    except KeyError:
        return {"error": json.dumps(out)[:200]}, ms


if __name__ == "__main__":
    rows = []
    for e in PROBE:
        text = serialize(e)
        try:
            lt, lt_ms = g_ask("phishing_tfidf", text)
            lt = {"label": (lt.get("output") or {}).get("label"), "conf": round(lt.get("confidence", 0), 3),
                  "action": lt.get("action"), "ms": round(lt_ms, 1)}
        except Exception as ex:
            lt = {"error": str(ex)[:100]}
        try:
            ll, ll_ms = g_ask("phishing", text)
            ll = {"label": (ll.get("output") or {}).get("label"), "conf": round(ll.get("confidence", 0), 3),
                  "action": ll.get("action"), "ms": round(ll_ms, 1)}
        except Exception as ex:
            ll = {"error": str(ex)[:100]}
        jv, j_ms = jev_ask(e)
        jv["ms"] = round(j_ms, 1) if j_ms >= 0 else -1
        rows.append({"id": e["id"], "expect": e["expect"], "flip": e["flip"],
                     "tfidf": lt, "logistic": ll, "jev": jv})
        ok_t = "OK " if lt.get("label") == e["expect"] else "MISS"
        ok_l = "OK " if ll.get("label") == e["expect"] else "MISS"
        ok_j = "OK " if jv.get("choice") == e["expect"] else ("MISS" if "choice" in jv else "ERR ")
        print(f"[{e['id']} expect={e['expect']}] tfidf:{ok_t}{lt} logistic:{ok_l}{ll} jev:{ok_j}{jv}",
              flush=True)

    json.dump(rows, open(TS + "/negation_probe.json", "w"), indent=1)


    def score(rows, get):
        items = [(r["expect"], get(r)) for r in rows]
        ok = sum(1 for e, g in items if e == g)
        tot = sum(1 for _, g in items if g is not None)
        flips = [r for r in rows if r["flip"]]
        fok = sum(1 for r in flips if get(r) == r["expect"])
        return f"{ok}/{tot} overall, flips {fok}/{len(flips)}"


    print("TF-IDF:  ", score(rows, lambda r: r["tfidf"].get("label")), flush=True)
    print("Logistic:", score(rows, lambda r: r["logistic"].get("label")), flush=True)
    print("Jev:     ", score(rows, lambda r: r["jev"].get("choice")), flush=True)
    print("DONE", flush=True)

