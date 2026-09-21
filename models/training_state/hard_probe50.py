#!/usr/bin/env python3
"""50-item hard probe (25 phish / 25 legit). Fresh patterns, disjoint from all
prior probes/aug/contrastive sets. PURE TEST — never train on these.
Key via TYPESAFE_API_KEY env only."""
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


P = [
    E("A01", "phishing", "BEC wire change", subject="Updated payment details",
      body="Hi, our bank details changed for the March invoice. Please send the wire to the new account below. Thanks!",
      sender="Vendor", **{"from": "accounts@vendor-supplies.com"}),
    E("A02", "phishing", "BEC gift cards", subject="Quick favor",
      body="Are you free? I need gift cards for client gifts. Scratch and send photos of the codes. Do not call, I am in meetings.",
      sender="Director", **{"from": "director.jane88@outlook.com"}),
    E("A03", "phishing", "BEC landlord", subject="Rent",
      body="This is your landlord. Rent goes to a new address starting next month. Confirm you got this.",
      sender="Landlord", **{"from": "landlord.mike23@yahoo.com"}),
    E("A04", "phishing", "homoglyph bank", subject="Card locked",
      body="Your card was locked after failed attempts. Unlock it now to restore access.",
      link_display_text="Unlock", link_url="http://secure.arnazon-pay.com/unlock",
      sender="Bank", **{"from": "alerts@arnazon-pay.com"}),
    E("A05", "phishing", "homoglyphapple", subject="Receipt",
      body="A purchase of $899 was made with your account. If this was not you, dispute it here.",
      link_display_text="Dispute", link_url="http://appleid.app1e-support.com/dispute",
      sender="Apple", **{"from": "no-reply@app1e-support.com"}),
    E("A06", "phishing", "QR parking fine", subject="Parking fine",
      body="Unpaid parking fine of $45. Scan the QR sticker on the notice to pay within 48 hours.",
      sender="City", **{"from": "fines@city-violations.org"}),
    E("A07", "phishing", "voicemail attachment v2", subject="Voice message (2)",
      body="Two voice messages could not be delivered to your inbox. Extract the attached archive to listen.",
      sender="Telecom", **{"from": "voip@telecom-msgs.info"}),
    E("A08", "phishing", " Macros doc", subject="Q1 numbers, enable content",
      body="Q1 numbers are in the attached sheet. Click Enable Content at the top so the figures load.",
      sender="Finance", **{"from": "finance@company-docs.net"}),
    E("A09", "phishing", "double-neg wire", subject="Not optional: transfer",
      body=("It is not true that no transfer is needed. The old account is not valid anymore. "
            "Do not delay: move the funds, not next week but now."),
      sender="Treasury", **{"from": "treasury@funds-move.tk"}),
    E("A10", "phishing", "double-neg login", subject="Never say never: login",
      body=("Never say your login never needs checks. A device that is not recognized is not "
            "cleared. Do not skip: approve it here."),
      link_display_text="Approve", link_url="http://device-approve.ga/ok",
      sender="Security", **{"from": "security@device-approve.ga"}),
    E("A11", "phishing", "double-neg delivery fee", subject="Fee never waived",
      body=("The storage fee was never waived and is not small. It is not possible to hold the box "
            "without it. Do not abandon your parcel: pay."),
      link_display_text="Pay", link_url="http://box-fee.ml/hold",
      sender="Depot", **{"from": "depot@box-fee.ml"}),
    E("A12", "phishing", "recipe preamble + trap", subject="Cake + survey prize",
      body=("The lemon cake recipe: zest, butter, sugar, bake 35 min. "
            "Separately: you were picked for a survey prize, claim it now."),
      link_display_text="Claim", link_url="http://survey-prize.ga/c",
      sender="Club", **{"from": "club@company.com"}),
    E("A13", "phishing", "weather preamble + trap", subject="Storm + account",
      body=("Storm warning tonight with hail after 9pm. Charge devices. "
            "Also your streaming account shows a failed payment; update it here."),
      link_display_text="Update", link_url="http://billing-fix.tk/u",
      sender="Neighbor", **{"from": "neighborhood@community-mail.org"}),
    E("A14", "phishing", "reply-chain HR", subject="Re: benefits",
      body="Per my last mail, everyone must re-enroll or lose coverage Friday. Use the enrollment link.",
      link_display_text="Enroll", link_url="http://benefits-re.tk/e",
      sender="HR", **{"from": "hr-benefits@outlook.com"}),
    E("A15", "phishing", "reply-chain legal", subject="Re: Re: contract",
      body="As discussed, sign the attached addendum today. Legal is waiting and the deadline cannot move.",
      link_display_text="Sign", link_url="http://contract-add.tk/s",
      sender="Legal", **{"from": "legal@firm-docs.info"}),
    E("A16", "phishing", "spaced obfuscation", subject="Hello",
      body="Dear user, p l e a s e c o n f i r m your d e t a i l s on our s e c u r e page.",
      link_display_text="Secure page", link_url="http://secure-page2.ga/x",
      sender="Service", **{"from": "service@secure-page2.ga"}),
    E("A17", "phishing", "leetspeak lure", subject="Fr33 g1ft",
      body="Cl41m your fr33 g1ft c4rd now, 0nly t0day, cl1ck h3re.",
      link_display_text="Here", link_url="http://free-g.ml/g",
      sender="Promo", **{"from": "promo@free-g.ml"}),
    E("A18", "phishing", "encrypted tax zip", subject="Tax file",
      body="Your tax file is encrypted for safety. Password is your birth year. Open the attachment on a computer.",
      sender="Tax preparer", **{"from": "tax@prepare-fast.org"}),
    E("A19", "phishing", "photo album zip", subject="Weekend photos",
      body="Weekend photos are in the zip. It asks for a password: use 0000. Enjoy!",
      sender="Cousin", **{"from": "cousin92@gmail.com"}),
    E("A20", "phishing", "legit invoice + harvest link", subject="Invoice #4417",
      body="Invoice 4417 for $210 is attached as PDF and is correct. Pay from your bank as usual, or view it online.",
      link_display_text="View online", link_url="http://invoice-view.tk/4417",
      sender="Supplier", **{"from": "billing@supplier-invoices.com"}),
    E("A21", "phishing", "legit tracking + harvest", subject="Shipped",
      body="Your parcel shipped this morning with tracking. Everything is normal; details below.",
      link_display_text="Details", link_url="http://track-info.ga/p",
      sender="Shop", **{"from": "ship@shop-track.info"}),
    E("A22", "phishing", "webinar recording lure", subject="Webinar replay",
      body="Yesterday's webinar replay is ready. Log in with your work email to watch.",
      link_display_text="Replay", link_url="http://webinar-replay.ml/in",
      sender="Events", **{"from": "events@webinar-host.co"}),
    E("A23", "phishing", "shared slides lure", subject="Deck shared",
      body="I shared the board deck with you. It expires in 24 hours, open it soon.",
      link_display_text="Open deck", link_url="http://deck-share.ga/b",
      sender="A manager", **{"from": "manager@hotmail.com"}),
    E("A24", "phishing", "smishing-style short", subject="(none)",
      body="BANK: suspicious $900 charge. Reply YES to block or tap here.",
      link_display_text="here", link_url="http://bank-block.tk/y",
      sender="SMS", **{"from": "sms@bank-block.tk"}),
    E("A25", "phishing", "calendar invite lure", subject="Interview invite",
      body="You are invited to interview Thursday. Accept the calendar hold and confirm attendance here.",
      link_display_text="Confirm", link_url="http://hire-confirm.ml/i",
      sender="Hiring", **{"from": "hiring@hire-confirm.ml"}),
    E("B01", "legitimate", "meta: detector chat", subject="Detector tuning",
      body=("In the meeting we will tune the detector: phishing scores, false positives, "
            "scam recall. Bring charts of credential-theft rates and verify-link traps."),
      sender="ML team", **{"from": "ml@company.com"}),
    E("B02", "legitimate", "meta: postmortem scam", subject="Postmortem",
      body=("Postmortem of yesterday's incident: a scam invoice with a harvest link reached "
            "two inboxes. Nobody clicked. Reviewing filters Monday."),
      sender="Security", **{"from": "security@company.com"}),
    E("B03", "legitimate", "won argument", subject="We won!",
      body="We won the contract! Celebration Thursday. You earned this win, team.",
      sender="Sales", **{"from": "sales@company.com"}),
    E("B04", "legitimate", "free lunch", subject="Free lunch",
      body="Free lunch in the lobby today while supplies last. First come, first served.",
      sender="Office", **{"from": "office@company.com"}),
    E("B05", "legitimate", "call6829 security", subject="Call us, no links",
      body=("Suspicious card activity may have occurred. Call the number on the back of your "
            "card at once. This email contains no links by design."),
      sender="Bank", **{"from": "alerts@bank.com"}),
    E("B06", "legitimate", "in-person verify", subject="Come to desk 4",
      body="To reset passwords, come to desk 4 with your badge. We never do resets by email or link.",
      sender="IT", **{"from": "it@company.com"}),
    E("B07", "legitimate", "statement portal", subject="Bill ready",
      body="Your bill is ready in the customer portal as always. Log in through your bookmark.",
      sender="Utility", **{"from": "billing@utility.com"}),
    E("B08", "legitimate", "app piggyback", subject="Update the app",
      body="Update the banking app from your phone's official store. Never sideload it from links.",
      sender="Bank", **{"from": "mobile@bank.com"}),
    E("B09", "legitimate", "double-neg all-clear", subject="All clear",
      body=("Not a single alert is unresolved and no account is at risk. There is nothing you "
            "must not ignore because there is nothing at all."),
      sender="NOC", **{"from": "noc@company.com"}),
    E("B10", "legitimate", "double-neg holiday", subject="Holiday",
      body=("It is not the case that the office is not closed Friday: it is closed, no one is "
            "expected, nothing is due."),
      sender="Office", **{"from": "office@company.com"}),
    E("B11", "legitimate", "contractor zoom", subject="Kickoff",
      body="Kickoff with the contractor at 3pm. They sent their own bridge.",
      link_display_text="Bridge", link_url="https://meet.google.com/abc-defg-hij",
      sender="PM", **{"from": "pm@company.com"}),
    E("B12", "legitimate", "webex client", subject="Support session",
      body="Support session at noon. Use the desktop client, meeting number in the invite.",
      link_display_text="Invite", link_url="https://company.webex.com/meet/supp",
      sender="Support", **{"from": "support@company.com"}),
    E("B13", "legitimate", "joke invoice meme", subject="Fwd: invoice meme",
      body="This meme (invoice for $0.00, pay now or else!!!) killed me. Almost forwarded to finance.",
      sender="Dev", **{"from": "dev@company.com"}),
    E("B14", "legitimate", "joke IT crowd", subject="Have you tried",
      body="Have you tried turning it off and on again? Asking for the ticket queue. Free coffee if you guess the caller.",
      sender="Helpdesk", **{"from": "helpdesk@company.com"}),
    E("B15", "legitimate", "advisory: locks", subject="Door codes",
      body="Never prop the stairwell doors and never share your badge, not with visitors, not with anyone.",
      sender="Facilities", **{"from": "facilities@company.com"}),
    E("B16", "legitimate", "advisory: tails", subject="No tailgating",
      body="Do not hold the turnstile for strangers, no matter the story, no exceptions, ever.",
      sender="Security", **{"from": "security@company.com"}),
    E("B17", "legitimate", "release notes links", subject="v2.4 notes",
      body="v2.4 notes: faster sync, dark mode, bug fixes. Full log and downloads inside.",
      link_display_text="Notes", link_url="https://docs.company.com/v2-4",
      sender="Eng", **{"from": "eng@company.com"}),
    E("B18", "legitimate", "status page", subject="All systems go",
      body="All systems operational. Past incidents and uptime graphs are on the status page.",
      link_display_text="Status", link_url="https://status.company.com/",
      sender="SRE", **{"from": "sre@company.com"}),
    E("B19", "legitimate", "delivery DHL", subject="DHL delivery",
      body="DHL delivers your documents tomorrow before noon. Track with the airbill number.",
      link_display_text="Track", link_url="https://www.dhl.com/track/123456",
      sender="DHL", **{"from": "noreply@dhl.com"}),
    E("B20", "legitimate", "pickup locker", subject="Locker pickup",
      body="Your order waits in locker 14. The code expires Sunday. Bring the barcode.",
      link_display_text="Barcode", link_url="https://shop-mart.com/pickup/14",
      sender="Shop", **{"from": "orders@shop-mart.com"}),
    E("B21", "legitimate", "training moved", subject="Training moved",
      body="Security training moves to room B. It will not be recorded, so do not miss it.",
      sender="Training", **{"from": "training@company.com"}),
    E("B22", "legitimate", "no deploy freeze", subject="No freeze",
      body="There is no deploy freeze this week. Ship normally; nothing is blocked and no approval is needed.",
      sender="Eng", **{"from": "eng@company.com"}),
    E("B23", "legitimate", "payslip ready", subject="Payslip",
      body="Your payslip is in the HR portal. Review deductions there; reply here with questions.",
      sender="Payroll", **{"from": "payroll@company.com"}),
    E("B24", "legitimate", "garden rota", subject="Garden rota",
      body="Garden rota for June is pinned. Watering cans are by the shed; no signup needed.",
      sender="Club", **{"from": "club@company.com"}),
    E("B25", "legitimate", "book club", subject="Book club pick",
      body="Next pick is a mystery novel, 300 pages, easy. Wine is on Maya. Non-readers welcome too.",
      sender="Club", **{"from": "club@company.com"}),
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


rows, tok_in, t_loc = [], 0, 0.0
for e in P:
    text = serialize(e)
    try:
        lt, ms = g_ask("phishing_tfidf", text)
        t_loc += ms
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
    tok_in += int((jv.get("usage") or {}).get("input_tokens", 0))
    rows.append({"id": e["id"], "expect": e["expect"], "note": e["note"],
                 "tfidf": lt, "logistic": ll,
                 "jev": {k: v for k, v in jv.items() if k != "usage"},
                 "jev_ms": round(j_ms, 1)})
    f = lambda g, k: "OK " if g.get(k) == e["expect"] else ("MISS" if k in g else "ERR ")
    print(f"[{e['id']} {e['note']} expect={e['expect']}] "
          f"tfidf:{f(lt,'label')}{lt.get('label')}/{lt.get('action')} "
          f"log:{f(ll,'label')}{ll.get('label')}/{ll.get('action')} "
          f"jev:{f(jv,'choice')}{jv.get('choice')}/{jv.get('conf')}", flush=True)

json.dump(rows, open(str(_TS) + "/hard50_results.json", "w"), indent=1)


def score(get):
    ok = sum(1 for r in rows if get(r) == r["expect"])
    return ok, f"{ok}/{len(rows)}"


t_ok, t_s = score(lambda r: r["tfidf"].get("label"))
l_ok, l_s = score(lambda r: r["logistic"].get("label"))
j_ok, j_s = score(lambda r: r["jev"].get("choice"))
print(f"TF-IDF:{t_s} Logistic:{l_s} Jev:{j_s} JevTokens={tok_in} (~${tok_in/1e6*0.042:.4f})",
      flush=True)
print("DONE", flush=True)
