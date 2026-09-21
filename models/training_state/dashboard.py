#!/usr/bin/env python3
"""Gavel dashboard: see everything. Port 7587. Stdlib only. Auto-refresh 30s.
Panels: services, questions+metrics, production pointers, flywheel state,
live ask box. Data: Gavel :7575 API + training_state files. No writes."""
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GAVEL = "http://127.0.0.1:7575"
TS = "D:/gavel/models/training_state/"


def g(path, timeout=15):
    try:
        with urllib.request.urlopen(GAVEL + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)[:120]}


def ask(q, text):
    body = json.dumps({"question": q, "input": text}).encode()
    req = urllib.request.Request(
        GAVEL + "/ask", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def read(name):
    try:
        return open(TS + name, encoding="utf-8").read()
    except Exception as e:
        return f"(unreadable: {e})"


def page(notice=""):
    qs = g("/questions")
    names = qs.get("questions", []) if isinstance(qs, dict) else []
    rows = []
    for n in sorted(names):
        m = g(f"/metrics?question={n}")
        if "_error" in m:
            rows.append(f"<tr><td>{n}</td><td colspan=5>error</td></tr>")
            continue
        rows.append(
            f"<tr><td><b>{n}</b></td><td>{','.join(m.get('classes', []))[:60]}</td>"
            f"<td>{m.get('temperature')}</td>"
            f"<td>{m.get('ece_before')}→{m.get('ece_after')}</td>"
            f"<td>{m.get('trained')}</td></tr>")
    opts = "".join(f"<option>{n}</option>" for n in sorted(names))
    try:
        man = json.dumps(json.loads(read("manifest.json")), indent=1)
    except Exception:
        man = read("manifest.json")
    fly = read("FLYWHEEL.md").replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Gavel dashboard</title><meta http-equiv="refresh" content="30">
<style>body{{font-family:system-ui;max-width:1000px;margin:1em auto;padding:0 1em}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.3em .5em;font-size:.85em}}
.panels{{display:grid;grid-template-columns:1fr 1fr;gap:1em}}pre{{background:#f4f4f4;padding:.5em;overflow:auto;font-size:.8em}}
.note{{background:#fff8e1;padding:.5em;border-radius:6px}}</style></head><body>
<h1>Gavel dashboard <small>(refresh 30s)</small></h1>
<div class="note">{notice} Services: gavel :7575 · decompose :7586 ·
<a href="http://127.0.0.1:7586/">decompose UI</a></div>
<h2>Questions ({len(names)})</h2>
<table><tr><th>question</th><th>classes</th><th>T</th><th>ECE before→after</th><th>trained</th></tr>
{''.join(rows)}</table>
<div class="panels"><div><h2>Production manifest</h2><pre>{man}</pre></div>
<div><h2>Live ask</h2><select id="q">{opts}</select><br>
<textarea id="t" rows="3" cols="48">from: a@b.c sender: s subject: t link_text:  link_url:  body: routine notice</textarea><br>
<button onclick="go()">Ask</button><pre id="o"></pre>
<script>async function go(){{const r=await fetch('/ask',{{method:'POST',headers:{{'Content-Type':'application/json'}},
body:JSON.stringify({{question:document.getElementById('q').value,input:document.getElementById('t').value}})}});
document.getElementById('o').textContent=JSON.stringify(await r.json(),null,1);}}</script></div></div>
<h2>Flywheel (rules of the system)</h2><pre>{fly[:4000]}</pre>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        b = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{{}}")
            out = ask(d["question"], d["input"])
            b = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except Exception as e:
            b = json.dumps({"error": str(e)[:200]}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)


print("dashboard on :7587", flush=True)
ThreadingHTTPServer(("0.0.0.0", 7587), H).serve_forever()
