#!/usr/bin/env python3
"""probe_items - 8-pair contrastive negation probe."""

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


