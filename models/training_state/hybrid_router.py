#!/usr/bin/env python3
"""Hybrid router (Step 2): Gavel fast-path + Qwen3-4B fallback.

  POST /route {"question": "<gavel question>", "input": "<text or {...>}"}
    1. Asks Gavel (:7575) for a decision.
    2. action==act  -> returns Gavel's answer immediately (~ms).
    3. else         -> asks Qwen3-4B (:local) to classify into the question's
                       labels, returns both opinions.

  GET /health, GET / (tiny demo UI). Port 7586. Stdlib + torch/transformers.
"""
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import GAVEL_URL as GAVEL, QWEN as _QWEN
MODEL_DIR = str(_QWEN)


def gavel(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        GAVEL + path, data=data,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print("loading Qwen3-4B fallback...", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
qwen = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float16, device_map="auto",
    low_cpu_mem_usage=True, trust_remote_code=False)
qwen.eval()
print("Qwen ready.", flush=True)
_lock = threading.Lock()


def qwen_classify(labels, text, max_new_tokens=16):
    prompt = ("Classify the input into exactly one of these labels: "
              + ", ".join(labels)
              + ". Reply with only the label.\n\nInput: "
              + (text if isinstance(text, str) else json.dumps(text))
              + "\nLabel:")
    enc = tok(prompt, return_tensors="pt")
    try:
        dev = next(qwen.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    enc = {k: v.to(dev) for k, v in enc.items()}
    t0 = time.perf_counter()
    with _lock:
        with torch.no_grad():
            out = qwen.generate(**enc, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tok.pad_token_id)
    ms = (time.perf_counter() - t0) * 1000
    cont = tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    low = cont.lower()
    pick = next((l for l in labels if l.lower() in low), None)
    return {"reply": cont, "pick": pick, "latency_ms": round(ms, 1)}


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Hybrid: Gavel+Qwen</title>
<style>body{font-family:system-ui;max-width:720px;margin:2em auto;padding:0 1em}
#log{border:1px solid #ccc;border-radius:8px;padding:1em;min-height:200px}
.fast{color:green}.slow{color:#b45309}</style></head><body>
<h2>Hybrid router — Gavel fast-path, Qwen fallback</h2>
<div><input id="q" value="domain_router" size="16"><input id="m" size="50" placeholder="input text">
<button onclick="go()">Route</button></div><div id="log"></div>
<script>async function go(){const l=document.getElementById('log');
l.innerHTML+='<p>…routing…</p>';
const r=await fetch('/route',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:document.getElementById('q').value,input:document.getElementById('m').value})});
const j=await r.json();l.lastChild.remove();
l.innerHTML+=`<p class="${j.route==='gavel'?'fast':'slow'}"><b>${j.route}</b> — `+JSON.stringify(j).slice(0,400)+'</p>';}</script>
</body></html>"""


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
            return self._send({"ok": True, "gavel": True, "frontier": "qwen3-4b"})
        return self._send(HTML, ctype="text/html")

    def do_POST(self):
        if self.path != "/route":
            return self._send({"error": "use POST /route"}, 400)
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            q, inp = d["question"], d["input"]
        except Exception as e:
            return self._send({"error": str(e)}, 400)
        try:
            g = gavel("/ask", {"question": q, "input": inp})
        except Exception as e:
            return self._send({"error": f"gavel: {e}"}, 502)
        if g.get("action") == "act":
            return self._send({"route": "gavel", "decision": g})
        try:
            labels = gavel(f"/metrics?question={q}").get("classes", [])
        except Exception:
            labels = []
        if not labels:
            return self._send({"route": "gavel", "decision": g,
                               "note": "no labels for fallback"})
        f = qwen_classify(labels, inp)
        return self._send({"route": "frontier", "gavel": g, "frontier": f})


print("hybrid on :7586  (gavel :7575 fast-path -> Qwen fallback)", flush=True)
ThreadingHTTPServer(("0.0.0.0", 7586), H).serve_forever()
