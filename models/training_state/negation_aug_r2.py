#!/usr/bin/env python3
"""Round-2 augmentation: kill P5b (legit delivery/tracking) and P6b (negation-dressed verify phish).
All wording fresh — disjoint from probe AND round-1. 16 train + 4 val."""
import json

R2_TRAIN = [
    # legit delivery / tracking / shipment notifications with benign links
    ("legitimate", "Order shipped",
     "Your order has shipped via FedEx. Follow its journey with the button below.",
     "Store", "orders@shop-mart.com", "Track shipment", "https://www.fedex.com/track/48291011"),
    ("legitimate", "Delivery attempt made",
     "We attempted delivery today but nobody answered. The parcel returns tomorrow; see the slip for details.",
     "Post Office", "noreply@post-office.com", "Delivery options", "https://www.post-office.com/redelivery"),
    ("legitimate", "Your ride receipt",
     "Thanks for riding. Your receipt and trip map are ready in the app.",
     "Rides", "receipts@ride-app.com", "View receipt", "https://ride-app.com/receipts/88aq"),
    ("legitimate", "Prescription ready",
     "Your prescription is ready for pickup at the counter. Bring your card; no payment is due.",
     "Pharmacy", "pharmacy@care-plus.com", "Store hours", "https://care-plus.com/stores"),
    ("legitimate", "Flight check-in open",
     "Check-in for your Friday flight is open. Seats and boarding passes are in the app.",
     "Airline", "noreply@sky-air.com", "Check in", "https://sky-air.com/checkin/XY42"),
    ("legitimate", "Subscription renewed",
     "Your yearly subscription renewed at the regular price. The invoice is attached for your records.",
     "Streaming", "billing@stream-co.com", "", ""),
    ("legitimate", "Appointment reminder",
     "Reminder: dental checkup Tuesday at 9am. Reply C to confirm or call to move it.",
     "Dental Clinic", "reminders@dental-care.com", "", ""),
    ("legitimate", "Refund processed",
     "The $18.40 overcharge was refunded to your original payment method. It posts within five business days.",
     "Store Support", "support@shop-mart.com", "", ""),
    # negation-dressed verify/account phish
    ("phishing", "Not spam: sign-in alert",
     "Not spam, do not delete. A sign-in that was not you needs review. This is not optional: confirm it was not you.",
     "Sign-in Protection", "protect@login-alerts.tk",
     "Review sign-in", "http://login-alerts.tk/review"),
    ("phishing", "No action? Wrong. Verify.",
     "Think no action is needed? Wrong. Your vault shows exports that never happened. Do not brush this off: verify now, not later.",
     "Vault Security", "vault@secure-vaults.ga",
     "Verify now", "http://secure-vaults.ga/verify"),
    ("phishing", "Don't ignore: storage locked",
     "Don't ignore this and don't wait. Your files are not accessible until the hold, which is not a mistake, is lifted.",
     "Cloud Drive", "drive@cloud-hold.ml",
     "Lift hold", "http://cloud-hold.ml/lift"),
    ("phishing", "Not a promotion: card on hold",
     "Not a promotion and not advertising. Your card is on hold for a charge you did not approve. This cannot wait: release it here.",
     "Card Services", "cards@card-release.tk",
     "Release card", "http://card-release.tk/go"),
    ("phishing", "Never too late, except now",
     "It is never too late to secure things, except this time. An intruder session that is still not closed must be ended by you.",
     "Session Guard", "guard@session-end.ga",
     "End session", "http://session-end.ga/stop"),
    ("phishing", "No mistake: invoice unpaid",
     "There is no mistake on our side. The invoice is not paid and the late fee is not waived. Settle it, do not dispute.",
     "Accounts", "accounts@invoice-settle.ml",
     "Settle now", "http://invoice-settle.ml/pay"),
    ("phishing", "Don't let them in: approve device",
     "Don't let a stranger in. A new device that is not yours asks for approval. If it was not you, deny it here at once.",
     "Device Trust", "trust@device-ok.tk",
     "Review device", "http://device-ok.tk/device"),
    ("phishing", "Not routine: password audit",
     "Not routine at all. The audit found sharing that is not allowed on your account. Re-secure it now; do not postpone.",
     "Audit Desk", "audit@pass-audit.ga",
     "Re-secure", "http://pass-audit.ga/fix"),
]

R2_VAL = [
    ("legitimate", "Bus pass renewed",
     "Your monthly bus pass renewed automatically. Show the code in the app when boarding.",
     "Transit", "noreply@city-transit.com", "Show pass", "https://city-transit.com/pass"),
    ("legitimate", "Warranty registered",
     "Your appliance warranty is registered through next year. Keep the receipt; nothing further is required.",
     "Appliances", "support@home-appliances.com", "", ""),
    ("phishing", "Not junk: prize shipment",
     "Not junk mail. Your prize shipment is not released because a fee is not paid. Do not abandon it: pay the small fee.",
     "Prize Desk", "desk@prize-ship.ml",
     "Release shipment", "http://prize-ship.ml/fee"),
    ("phishing", "Don't freeze my account, please act",
     "Don't let them freeze what is yours. A freeze that is not lifted today becomes permanent. Act here, not later.",
     "Account Desk", "desk@freeze-stop.tk",
     "Stop freeze", "http://freeze-stop.tk/act"),
]


def serialize(label, subject, body, sender, frm, link_text, link_url):
    return {"input": (f"from: {frm} sender: {sender} subject: {subject} "
                      f"link_text: {link_text} link_url: {link_url} body: {body}"),
            "label": label}


if __name__ == "__main__":
    tr = [serialize(*a) for a in R2_TRAIN]
    va = [serialize(*a) for a in R2_VAL]
    t0 = json.load(open("D:/gavel/models/training_state/negation_aug_train.json"))
    v0 = json.load(open("D:/gavel/models/training_state/negation_aug_val.json"))
    json.dump(t0 + tr, open("D:/gavel/models/training_state/negation_aug_train.json", "w"))
    json.dump(v0 + va, open("D:/gavel/models/training_state/negation_aug_val.json", "w"))
    from collections import Counter
    print("aug_train:", len(t0) + len(tr), Counter(x["label"] for x in t0 + tr), flush=True)
    print("aug_val:", len(v0) + len(va), flush=True)
