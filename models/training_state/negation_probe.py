#!/usr/bin/env python3
"""Contrastive negation probe: Jev Live vs local TF-IDF (+logistic ref).
8 pairs (base + negated counterpart). Key via TYPESAFE_API_KEY env only.
Run: $env:TYPESAFE_API_KEY='<key>'; python -u models/training_state/negation_probe.py
"""
import json
import os
import time
import urllib.request

GAVEL = "http://127.0.0.1:7575"
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


def E(pid, expect, flip, **kw):
    d = {"id": pid, "expect": expect, "flip": flip,
         "from": "unknown@example.com", "sender": "unknown",
         "subject": "(no subject)", "body": "",
         "link_display_text": "", "link_url": ""}
    d.update(kw)
    return d


PROBE = [
    E("P1b", "phishing", False, subject="Action required: verify your account",
      body="Dear customer, you were charged twice on your card. Verify your account now to claim your refund.",
      link_display_text="Verify now", link_url="http://bit.ly/3xqv9r",
      sender="billing-support", **{"from": "billing-support@notice-payments.com"}),
    E("P1n", "legitimate", True, subject="Security notice: ignore refund-link emails",
      body=("We will never ask you to verify your account through links. "
            "Do not click refund links like http://bit.ly/3xqv9r claiming double charges. "
            "Delete such emails; your account is not locked and no refund is owed."),
      sender="IT Security", **{"from": "security@company.com"}),
    E("P2b", "phishing", False, subject="Your password expires today",
      body="Urgent: your password expires today. Reset immediately or lose access.",
      link_display_text="Reset password", link_url="http://acct-reset.tk/login",
      sender="IT Helpdesk", **{"from": "it-helpdesk@mail-support.net"}),
    E("P2n", "legitimate", True, subject="Passwords do not expire",
      body=("No action needed: your password does not expire today and your account is not locked. "
            "Never reset your password through email links asking you to reset immediately."),
      sender="IT Helpdesk", **{"from": "helpdesk@company.com"}),
    E("P3b", "legitimate", False, subject="Team lunch tomorrow",
      body="Team lunch tomorrow at noon in the cafeteria. Menu: pasta and salad.",
      link_display_text="menu", link_url="https://intranet.company.com/menu",
      sender="Office Manager", **{"from": "office@company.com"}),
    E("P3n", "legitimate", False, subject="Team lunch cancelled",
      body="Team lunch is cancelled and will not happen tomorrow. Do not come to the cafeteria.",
      sender="Office Manager", **{"from": "office@company.com"}),
    E("P4b", "phishing", False, subject="You won a prize!",
      body="Congratulations! You won a $500 gift card. Claim your prize now before it expires.",
      link_display_text="Claim prize", link_url="http://prize-claim.ml/win",
      sender="Rewards Team", **{"from": "rewards@promo-winner.org"}),
    E("P4n", "legitimate", True, subject="Prize emails are scams",
      body=("You did not win any prize and no gift card is owed to you. "
            "Emails telling you to claim a prize through a link are scams. Delete this kind of message."),
      sender="IT Security", **{"from": "security@company.com"}),
    E("P5b", "legitimate", False, subject="Your package arrives Friday",
      body="Your package arrives Friday. Track it here.",
      link_display_text="Track package", link_url="https://www.ups.com/track/1Z999",
      sender="UPS", **{"from": "no-reply@ups.com"}),
    E("P5n", "legitimate", False, subject="Package delayed, no action needed",
      body=("Your package will not arrive Friday and tracking is temporarily unavailable. "
            "No action is needed; do not reply to this message."),
      sender="UPS", **{"from": "no-reply@ups.com"}),
    E("P6b", "phishing", False, subject="Verify today, not tomorrow",
      body=("This is not a drill and not tomorrow but today: your account shows an unusual sign-in "
            "that is not yours. Verify immediately, do not wait."),
      link_display_text="Verify now", link_url="http://secure-verify.ga/auth",
      sender="Account Security", **{"from": "security@account-alerts.info"}),
    E("P6n", "legitimate", True, subject="No verification needed",
      body=("No verification is needed and your account is not locked. "
            "There was no unusual sign-in. Ignore messages urging you to verify immediately."),
      sender="Account Security", **{"from": "security@company.com"}),
    E("P7b", "phishing", False, subject="Shared document: Q3 report",
      body="Someone shared the Q3 financial report with you. Open the document to review.",
      link_display_text="Open document", link_url="http://docs-share.tk/q3-report",
      sender="A colleague", **{"from": "colleague@gmail.com"}),
    E("P7n", "legitimate", True, subject="We never share documents this way",
      body=("We never share documents through free file hosts. Do not open document links "
            "from generic webmail addresses presenting as a colleague."),
      sender="IT Security", **{"from": "security@company.com"}),
    E("P8b", "legitimate", False, subject="Sprint review meeting invite",
      body="Sprint review meeting at 10am. Join with the link below.",
      link_display_text="Join meeting", link_url="https://teams.company.com/meet/review",
      sender="Engineering", **{"from": "eng@company.com"}),
    E("P8n", "legitimate", False, subject="Sprint review cancelled",
      body="The sprint review meeting is cancelled and will not take place. Do not join.",
      sender="Engineering", **{"from": "eng@company.com"}),
]


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

json.dump(rows, open("D:/gavel/models/training_state/negation_probe.json", "w"), indent=1)


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
