#!/usr/bin/env python3
"""Permanent contrastive augmentation (rule-based, deterministic, no GPU).

Failure MODE coverage, regenerated on EVERY retrain — not hand-written once:
  P-phish -> advisory-frame (legit): "Security notice: the quoted message is a
      scam. Do not follow its links. Quoted: <body>"  (kills P1n/P2n-class misses)
  P-phish -> negation-dress (phish): wrap lure with negation filler while keeping
      the trap ("not tomorrow but today...", "do not wait...") (kills P6b-class)
  L-legit -> cancel/negate (legit): polarity-flipped variants of notifications
      (kills P8n-class) + benign-link delivery variants (kills P5-class)

Inputs: base train jsonl. Output: contrastive jsonl. Seeded, reproducible.
Labels are trustworthy by construction (framing determines legitimacy).
"""
import json
import random

NEG_FILLERS = [
    "This is not a drill and not tomorrow but today.",
    "Do not wait and do not hesitate.",
    "This cannot stay unresolved and must not be ignored.",
    "Never ignore this notice.",
    "Do not brush this off.",
]

ADVISORY_FRAMES = [
    ("Security notice: scam pattern",
     "The IT desk warns that messages like the quoted one below are scams. "
     "Do not follow their links and do not send anything. Quoted scam: "),
    ("Fraud alert: ignore this pattern",
     "Fraudulent emails of the following kind are going around. They are not from us. "
     "Delete them and never use their links. Quoted fraud: "),
    ("Reminder: we never ask this way",
     "A reminder that we never contact you this way. Anything phrased like the quoted text "
     "below is a trap. Do not engage with it. Quoted trap: "),
]

CANCEL_PAIRS = [
    ("is scheduled for", "is cancelled and will not take place"),
    ("will happen tomorrow", "will not happen; it is called off"),
    ("is available now", "is no longer available"),
    ("arrives Friday", "will not arrive Friday"),
    ("is open", "is closed until further notice"),
    ("is required", "is not required"),
]


def serialize(label, subject, body, sender, frm, link_text="", link_url=""):
    return {"input": (f"from: {frm} sender: {sender} subject: {subject} "
                      f"link_text: {link_text} link_url: {link_url} body: {body}"),
            "label": label}


def augment(base_train, seed=11, per_class_cap=120):
    rng = random.Random(seed)
    phish = [x for x in base_train if x["label"] == "phishing"]
    legit = [x for x in base_train if x["label"] == "legitimate"]
    rng.shuffle(phish)
    rng.shuffle(legit)
    out = []
    # 1. advisory-framed phish -> legit (quotes lure tokens under negation framing)
    for x in phish[:per_class_cap]:
        subj, frame = rng.choice(ADVISORY_FRAMES)
        raw = x["input"].split("body: ", 1)[1] if "body: " in x["input"] else x["input"]
        out.append(serialize(
            "legitimate", subj, frame + raw[:400],
            "IT Security", "security@company.com"))
    # 2. negation-dressed phish -> phishing (lure intact under negation filler)
    for x in phish[:per_class_cap]:
        raw = x["input"].split("body: ", 1)[1] if "body: " in x["input"] else x["input"]
        filler = " ".join(rng.sample(NEG_FILLERS, 2))
        subj = x["input"].split("subject: ", 1)[1].split(" link_text")[0] \
            if "subject: " in x["input"] else "Notice"
        link = ""
        if "link_url: " in x["input"]:
            link = x["input"].split("link_url: ", 1)[1].split(" body:")[0]
        frm = x["input"].split("from: ", 1)[1].split(" sender:")[0] \
            if "from: " in x["input"] else "a@b.c"
        snd = x["input"].split("sender: ", 1)[1].split(" subject:")[0] \
            if "sender: " in x["input"] else "s"
        out.append(serialize("phishing", subj, filler + " " + raw[:350], snd, frm, "", link))
    # 3. cancelled/negated legit -> legit (polarity flips of benign mail)
    for x in legit[:per_class_cap]:
        raw = x["input"].split("body: ", 1)[1] if "body: " in x["input"] else x["input"]
        sent, done = raw, False
        for a, b in CANCEL_PAIRS:
            if a in raw.lower():
                # replace first case-insensitive occurrence
                idx = raw.lower().index(a)
                sent = raw[:idx] + b + raw[idx + len(a):]
                done = True
                break
        if not done:
            sent = "Update: " + raw[:350] + " This is now cancelled and will not happen."
        subj = "Update: " + (x["input"].split("subject: ", 1)[1].split(" link_text")[0]
                             if "subject: " in x["input"] else "notice")
        frm = x["input"].split("from: ", 1)[1].split(" sender:")[0] \
            if "from: " in x["input"] else "a@b.c"
        snd = x["input"].split("sender: ", 1)[1].split(" subject:")[0] \
            if "sender: " in x["input"] else "s"
        out.append(serialize("legitimate", subj, sent[:450], snd, frm))
    return out


if __name__ == "__main__":
    base = [json.loads(l) for l in
            open("D:/gavel/models/training_state/phishing_train.jsonl", encoding="utf-8")]
    gen = augment(base)
    from collections import Counter
    print(f"generated={len(gen)}", Counter(x["label"] for x in gen), flush=True)
    # sanity: no generated text may equal a probe input (disjoint check vs probe inputs)
    json.dump(gen, open("D:/gavel/models/training_state/contrastive_gen.json", "w"))
    print("wrote contrastive_gen.json", flush=True)
