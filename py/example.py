"""gavel-server demo: train a ticket router, calibrate it, ask it questions.

Run with the server up first:
    ./target/release/gavel-server   # in another terminal
    python3 py/example.py
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from gavel_client import GavelClient, GavelError

# Hardcoded templates per class; each expands deterministically into 2
# training examples (raw + a lightly reworded variant) -> 5 x 2 x 3 = 30.
TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription",
        "Please refund the invoice from last month",
        "My credit card was billed the wrong amount",
        "Why is there an extra charge on my bill",
        "I need a receipt for my recent payment",
    ],
    "support": [
        "How do I reset my password",
        "I cannot log in to my account",
        "Where do I update my profile settings",
        "How do I cancel my subscription",
        "Help me export my data as CSV",
    ],
    "technical": [
        "The app crashes when I upload a file",
        "The API returns a 500 error on webhook calls",
        "Sync gets stuck and never finishes",
        "The dashboard shows a blank screen",
        "The websocket connection keeps dropping",
    ],
}

VARIANTS = [
    lambda t: t,
    lambda t: "Hi, " + t[0].lower() + t[1:] + " -- please advise",
]

# Held-out validation phrasings, 3 per class -> 9.
VALIDATION = [
    ("My statement shows a duplicate charge from you", "billing"),
    ("Can you reverse the payment you took yesterday", "billing"),
    ("The invoice total does not match what I agreed to", "billing"),
    ("I forgot my login credentials, what do I do", "support"),
    ("Where is the setting to change my email address", "support"),
    ("Walk me through deleting my account", "support"),
    ("Uploading a photo makes the whole thing freeze", "technical"),
    ("Getting timeout errors from the REST endpoint", "technical"),
    ("The page never loads past the spinner", "technical"),
]

ASKS = [
    # (ticket, expected label, expected action-ish)
    ("I was charged twice on my credit card, please refund me", "billing", "act"),
    ("The app crashes every time I try to upload a photo", "technical", "act"),
    ("I cannot access my account and I think I was overcharged", None, "review"),
    ("xqzt blorple fnord wobble quux", None, "escalate"),
]


def main():
    client = GavelClient()

    print("== define ==")
    client.define(
        "route_ticket",
        {"type": "object", "properties": {"subject": {"type": "string"}}},
        {"type": "object", "properties": {"route": {"type": "string"}}},
        policy={"act_above": 0.9, "review_low": 0.6, "review_high": 0.9},
    )
    print("defined route_ticket")

    print("== train ==")
    train = [
        (variant(t), label)
        for label, ts in TEMPLATES.items()
        for t in ts
        for variant in VARIANTS
    ]
    client.train("route_ticket", train)
    print(f"trained on {len(train)} examples")

    print("== calibrate ==")
    client.calibrate("route_ticket", VALIDATION)
    print(f"calibrated on {len(VALIDATION)} validation examples")

    print("== metrics ==")
    m = client.metrics("route_ticket")
    print(
        f"trained={m['trained']} classes={m['classes']} "
        f"temperature={m['temperature']:.3f} "
        f"ece_before={m['ece_before']} ece_after={m['ece_after']}"
    )

    print("== ask ==")
    for ticket, want_label, want_action in ASKS:
        d = client.ask("route_ticket", {"subject": ticket})
        label = d["output"].get("label")
        conf, action = d["confidence"], d["action"]
        marks = []
        if want_label is not None:
            marks.append("label-ok" if label == want_label else f"label-MISS (want {want_label})")
        if want_action is not None:
            marks.append(
                "action-ok" if action == want_action else f"action-note (want ~{want_action})"
            )
        print(f'  "{ticket}"')
        print(f"    -> {label}  confidence={conf:.3f}  action={action}  [{' '.join(marks)}]")

    print("done")


if __name__ == "__main__":
    try:
        main()
    except GavelError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
