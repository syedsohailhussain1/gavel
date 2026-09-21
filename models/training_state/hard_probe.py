#!/usr/bin/env python3
"""HARD adversarial probe: 20 fresh items neither side trained on.
BEC/linkless traps, homoglyphs, QR lures, double negation, meta-phishing talk,
prize-worded office fun. Key via TYPESAFE_API_KEY env only. Never written to disk.
Saves hard_probe_results.json (answers only, no key)."""
import json
import os
import time
import urllib.request

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from gavel_paths import GAVEL_URL as GAVEL, TS as _TS
JEV_URL = "https://api.typesafe.ai/v1/systemone"
VERDICT = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "This email arrived in the user's inbox and contains a link (`link_url`). "
            "Is it a phishing attempt, or a legitimate message whose link is safe to click?"),
        "criteria": {
            "phishing": "The email or its link is a phishing attempt or otherwise malicious. The user should not click the link.",
            "legitimate": "A normal, legitimate message. The link is safe to click."}}}


def E(pid, expect, note, **kw):
    d = {"id": pid, "expect": expect, "note": note,
         "from": "x@y.z", "sender": "s", "subject": "t", "body": "",
         "link_display_text": "", "link_url": ""}
    d.update(kw)
    return d


HARD = [
    E("H1", "phishing", "linkless BEC",
      subject="Invoice", body="Hi, please process the attached invoice today. Thanks.",
      sender="Chief Executive", **{"from": "chief.executive87@gmail.com"}),
    E("H2", "phishing", "homoglyph domain",
      subject="Microsoft account alert",
      body="Unusual activity was found. Confirm your identity now to keep access.",
      link_display_text="Confirm", link_url="http://login.micros0ft-support.com/verify",
      sender="Microsoft", **{"from": "security@micros0ft-support.com"}),
    E("H3", "phishing", "QR lure, no URL",
      subject="Parcel held",
      body="Your parcel is held at the depot. Scan the attached QR code to release it.",
      sender="Depot", **{"from": "depot@parcel-release.info"}),
    E("H4", "phishing", "double negation trap",
      subject="Verification exception",
      body=("It is not uncommon for accounts to never require verification, "
            "but yours is not one of them. You cannot skip this: verify now, not later."),
      link_display_text="Verify", link_url="http://verify-exception.ga/go",
      sender="Accounts", **{"from": "accounts@verify-exception.ga"}),
    E("H5", "phishing", "benign preamble + trap",
      subject="Office move + gift",
      body=("The office move to floor 3 finishes Friday. Boxes are in the hall; "
            "kitchen opens Monday. Also: a gift card waits for you, claim it here."),
      link_display_text="Claim", link_url="http://gift-claim.ml/c",
      sender="Office", **{"from": "office@company.com"}),
    E("H6", "phishing", "fake thread lure",
      subject="Re: Re: Re: document you missed",
      body="Following up on the file you never opened. Everyone else signed already. Open it now.",
      link_display_text="Open file", link_url="http://doc-sign.tk/f",
      sender="A colleague", **{"from": "colleague123@outlook.com"}),
    E("H7", "phishing", "dotted obfuscation",
      subject="Account notice",
      body="Please v.e.r.i.f.y your account details on the secure page to avoid closure.",
      link_display_text="Secure page", link_url="http://secure-page.ga/v",
      sender="Support", **{"from": "support@secure-page.ga"}),
    E("H8", "phishing", "encrypted-zip payroll",
      subject="Payroll details",
      body="Payroll details are in the attached encrypted archive. The password is 1234. Open it on your work machine.",
      sender="Payroll", **{"from": "payroll@company-payroll.org"}),
    E("H9", "phishing", "legit-looking renewal + harvest link",
      subject="Subscription renewed",
      body="Your yearly subscription renewed normally. Review the invoice in your account.",
      link_display_text="My account", link_url="http://account-review.tk/login",
      sender="Streaming", **{"from": "billing@stream-co-billing.com"}),
    E("H10", "phishing", "fake meeting recording",
      subject="Missed Zoom meeting",
      body="You missed the all-hands. Watch the recording by signing in below.",
      link_display_text="Watch", link_url="http://zoom-recording.ml/signin",
      sender="Comms", **{"from": "comms@zoom-meetings.info"}),
    E("H11", "legitimate", "meta talk about phishing",
      subject="Jev benchmark discussion",
      body=("For the meeting: phishing signals, calibration error and scam detection. "
            "Bring the report on verify-link scams and credential harvesting numbers."),
      sender="Research", **{"from": "research@company.com"}),
    E("H12", "legitimate", "prize-worded office fun",
      subject="Lottery pool + pizza",
      body="We won the office lottery pool! $12 split six ways. Winnings buy pizza Friday. No claim needed.",
      sender="Office", **{"from": "office@company.com"}),
    E("H13", "legitimate", "vault notice, call-don't-click",
      subject="New device on vault",
      body=("Your password vault was opened on a new device. If this was not you, "
            "call the number on your card. Do not use any link for this."),
      sender="Vault", **{"from": "vault@company.com"}),
    E("H14", "legitimate", "bank statement, no link",
      subject="Statement ready",
      body="Your monthly statement is ready. Open the app to view it. This message has no links.",
      sender="Bank", **{"from": "statements@bank.com"}),
    E("H15", "legitimate", "double negation, nothing needed",
      subject="Nothing needed",
      body="Not only is no action required, nothing is expiring and nobody needs anything from you. Ignore all such requests.",
      sender="IT", **{"from": "it@company.com"}),
    E("H16", "legitimate", "external vendor zoom",
      subject="Demo tomorrow",
      body="Demo with the vendor tomorrow at 2pm. Same link as last time.",
      link_display_text="Join", link_url="https://zoom.us/j/555111222",
      sender="Sales", **{"from": "sales@company.com"}),
    E("H17", "legitimate", "forwarded joke scam",
      subject="Fwd: look at this loser",
      body="Got this gem today: claim your refund now at bit.ly junk. Almost clicked. Coffee?",
      sender="Priya", **{"from": "priya@company.com"}),
    E("H18", "legitimate", "pure advisory, no quotes",
      subject="Codes stay secret",
      body="Reminder: never share login codes with anyone, not even IT, not ever, under no circumstance.",
      sender="IT Security", **{"from": "security@company.com"}),
    E("H19", "legitimate", "newsletter, many benign links",
      subject="Weekly digest",
      body="Digest: build green, picnic photos, new hires. Links: dashboard, gallery, directory.",
      link_display_text="dashboard", link_url="https://intranet.company.com/digest",
      sender="Comms", **{"from": "comms@company.com"}),
    E("H20", "legitimate", "refund posted, no link",
      subject="Refund posted",
      body="The refund of $18.40 posted to your card. Nothing to click and no form to fill.",
      sender="Store", **{"from": "support@shop-mart.com"}),
]


def serialize(e):
    return (f"from: {e['from']} sender: {e['sender']} subject: {e['subject']} "
            f"link_text: {e['link_display_text']} link_url: {e['link_url']} body: {e['body']}")


def g_ask(question, text, timeout=30):
    body = json.dumps({"question": question, "input": text}).encode()
    req = urllib.request.Request(
        GAVEL + "/ask", data=body,
        headers={"Content-Type": "application/json", "Connection": "close"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.perf_counter() - t0) * 1000


def jev_ask(state):
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set")
    payload = {"state": {k: state[k] for k in
                         ("from", "sender", "subject", "body",
                          "link_display_text", "link_url")},
               "model": "jev-latest", "questions": VERDICT}
    req = urllib.request.Request(
        JEV_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            out = json.loads(r.read())
    except Exception as e:
        return {"error": str(e)[:150]}, -1
    ms = (time.perf_counter() - t0) * 1000
    try:
        v = out["answers"]["verdict"]
        return {"choice": v["choice"], "conf": round(v.get("confidence", 0), 3),
                "usage": out.get("usage", {})}, ms
    except KeyError:
        return {"error": json.dumps(out)[:150]}, ms


rows, tok_in = [], 0
for e in HARD:
    text = serialize(e)
    try:
        lt, lt_ms = g_ask("phishing_tfidf", text)
        lt = {"label": (lt.get("output") or {}).get("label"),
              "conf": round(lt.get("confidence", 0), 3), "action": lt.get("action")}
    except Exception as ex:
        lt = {"error": type(ex).__name__}
    try:
        ll, _ = g_ask("phishing", text)
        ll = {"label": (ll.get("output") or {}).get("label"),
              "conf": round(ll.get("confidence", 0), 3), "action": ll.get("action")}
    except Exception as ex:
        ll = {"error": type(ex).__name__}
    jv, j_ms = jev_ask(e)
    jv["ms"] = round(j_ms, 1)
    tok_in += int((jv.get("usage") or {}).get("input_tokens", 0))
    rows.append({"id": e["id"], "expect": e["expect"], "note": e["note"],
                 "tfidf": lt, "logistic": ll, "jev": jv})
    f = lambda g, k: "OK " if g.get(k) == e["expect"] else ("MISS" if k in g else "ERR ")
    print(f"[{e['id']} {e['note']} expect={e['expect']}] "
          f"tfidf:{f(lt,'label')}{lt} logistic:{f(ll,'label')}{ll} jev:{f(jv,'choice')}{jv}",
          flush=True)

json.dump(rows, open(str(_TS) + "/hard_probe_results.json", "w"), indent=1)


def score(get):
    ok = sum(1 for r in rows if get(r) == r["expect"])
    return f"{ok}/{len(rows)}"


print("TF-IDF:  ", score(lambda r: r["tfidf"].get("label")), flush=True)
print("Logistic:", score(lambda r: r["logistic"].get("label")), flush=True)
print("Jev:     ", score(lambda r: r["jev"].get("choice")), flush=True)
print(f"Jev input tokens={tok_in} (~${tok_in/1e6*0.042:.4f})", flush=True)
print("DONE", flush=True)
