#!/usr/bin/env python3
"""Decompose sidecar (Step: decomposition): one POST -> verdict + 5 signals + rules.
POST /decompose {"email": {from,sender,subject,body,link_display_text,link_url}}
  -> {"verdict": {...}, "signals": [{id,label,confidence,action}...],
      "rules": {free_hosting, domain_mismatch}, "agree": bool, "ms": N}
Also accepts {"input": "<serialized string>"} (rules skipped).
GET /health, GET / (demo UI). Port 7586. No model weights here (all in Gavel).
"""
import json
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import GAVEL_URL as GAVEL, BENCH
sys.path.insert(0, str(BENCH))
SIGNALS = ["sig_free_hosting", "sig_domain_mismatch", "sig_lure",
           "sig_urgency", "sig_generic_sender"]
from bench.heuristics import features as heur_rules  # noqa: E402


def g_ask(question, text, timeout=30):
    body = json.dumps({"question": question, "input": text}).encode()
    req = urllib.request.Request(
        GAVEL + "/ask", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def serialize(e):
    return (f"from: {e.get('from','')} sender: {e.get('sender','')} "
            f"subject: {e.get('subject','')} link_text: {e.get('link_display_text','')} "
            f"link_url: {e.get('link_url','')} body: {e.get('body','')}")


def decompose(email=None, text=None):
    t0 = time.perf_counter()
    rules = {}
    if email is not None:
        text = serialize(email)
        try:
            h = heur_rules({k: email.get(k, "") for k in
                            ("from", "sender", "subject", "body",
                             "link_display_text", "link_url")})
            rules = {"free_hosting_rule": bool(h["hosting_or_shortener"]),
                     "domain_mismatch_rule": bool(h["etld1_mismatch"])}
        except Exception as e:
            rules = {"error": str(e)[:100]}
    v = g_ask("phishing_tfidf", text)
    verdict = {"label": (v.get("output") or {}).get("label"),
               "confidence": round(v.get("confidence", 0), 3),
               "action": v.get("action"), "energy": v.get("energy")}
    signals = []
    for s in SIGNALS:
        try:
            r = g_ask(s, text)
            signals.append({"id": s,
                            "label": (r.get("output") or {}).get("label"),
                            "confidence": round(r.get("confidence", 0), 3),
                            "action": r.get("action")})
        except Exception as e:
            signals.append({"id": s, "error": str(e)[:100]})
    fired = [s["id"] for s in signals if s.get("label") == "true"]
    agree = (verdict["label"] == "phishing") == (len(fired) >= 2)
    return {"verdict": verdict, "signals": signals, "rules": rules,
            "signals_fired": fired, "agree": agree,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Decompose: verdict+signals</title>
<style>body{font-family:system-ui;max-width:760px;margin:2em auto;padding:0 1em}
#log{border:1px solid #ccc;border-radius:8px;padding:1em}pre{background:#f4f4f4;padding:.5em;overflow:auto}</style>
</head><body><h2>Verdict + 5 signals (one call)</h2>
<textarea id="b" rows="4" cols="80">from: billing-support@notice-payments.com sender: billing-support subject: Action required: verify your account link_text: Verify now link_url: http://bit.ly/3xqv9r body: Dear customer, you were charged twice. Verify your account now to claim your refund.</textarea><br>
<button onclick="go()">Decompose</button><div id="log"></div>
<script>async function go(){const l=document.getElementById('log');l.innerHTML='…';
const t=document.getElementById('b').value;
const r=await fetch('/decompose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:t})});
l.innerHTML='<pre>'+JSON.stringify(await r.json(),null,1)+'</pre>';}</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        b = (json.dumps(obj) if ctype == "application/json" else obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"ok": True, "verdict": "phishing_tfidf", "signals": SIGNALS})
        return self._send(HTML, ctype="text/html")

    def do_POST(self):
        if self.path != "/decompose":
            return self._send({"error": "use POST /decompose"}, 400)
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send({"error": str(e)[:150]}, 400)
        try:
            if "email" in d:
                return self._send(decompose(email=d["email"]))
            elif "input" in d:
                return self._send(decompose(text=str(d["input"])))
            return self._send({"error": "need email{} or input"}, 400)
        except Exception as e:
            return self._send({"error": str(e)[:200]}, 500)


print("decompose on :7586 (verdict + 5 signals + rules)", flush=True)
ThreadingHTTPServer(("0.0.0.0", 7586), H).serve_forever()
