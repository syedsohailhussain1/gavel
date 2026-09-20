/**
 * Example: ticket routing as a typed decision — with a REAL trained engine.
 *
 * Trains a logistic-regression classifier on ~40 synthetic labeled tickets,
 * calibrates it, then routes unseen tickets. Watch the policy in action:
 * high confidence -> act, middling -> review, low -> escalate.
 *
 * Run:  npm run example
 * (from ts/decide, after building the native binding)
 */
import { Engine, TrainingExample } from "../src/index";

// Deterministic PRNG so the demo is reproducible.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const VOCAB: Record<string, string[]> = {
  billing: ["invoice", "billed", "charge", "refund", "payment", "subscription", "receipt", "overcharged", "billing", "credit card"],
  support: ["password", "login", "account", "reset", "signin", "username", "locked", "access", "profile", "email"],
  technical: ["crash", "bug", "error", "stack", "trace", "fails", "exception", "timeout", "server", "broken"],
};
const FILLER = ["please", "help", "issue", "problem", "thanks", "urgent", "hello", "hi"];
const LABELS = Object.keys(VOCAB);

function pick(rng: () => number, xs: string[]): string {
  return xs[Math.floor(rng() * xs.length)];
}

function makeTicket(rng: () => number, label: string, noisy: boolean): { subject: string; body: string } {
  const words: string[] = [];
  const n = 6 + Math.floor(rng() * 5);
  for (let i = 0; i < n; i++) {
    const r = rng();
    if (r < 0.55) words.push(pick(rng, VOCAB[label]));
    else if (r < 0.85) words.push(pick(rng, FILLER));
    else words.push(pick(rng, VOCAB[pick(rng, LABELS.filter((l) => l !== label))])); // cross-class noise
  }
  void noisy;
  const subject = words.slice(0, 3).join(" ");
  return { subject: subject.charAt(0).toUpperCase() + subject.slice(1), body: words.join(" ") + "." };
}

const rng = mulberry32(20260920);
const train: TrainingExample[] = [];
const validation: TrainingExample[] = [];
for (const label of LABELS) {
  for (let i = 0; i < 14; i++) train.push({ input: makeTicket(rng, label, true), label });
  for (let i = 0; i < 25; i++) validation.push({ input: makeTicket(rng, label, true), label });
}

const engine = new Engine();
engine.define({
  name: "route_ticket",
  inputSchema: {
    type: "object",
    required: ["subject", "body"],
    properties: { subject: { type: "string" }, body: { type: "string" } },
  },
  outputSchema: {
    type: "object",
    required: ["label"],
    properties: { label: { enum: LABELS } },
  },
  policy: { actAbove: 0.9, reviewLow: 0.6, reviewHigh: 0.9 },
});

console.log(`Training on ${train.length} synthetic tickets...`);
engine.train("route_ticket", train);
engine.calibrate("route_ticket", validation);
const m = engine.metrics("route_ticket");
console.log(
  `Trained. classes=[${m.classes.join(", ")}] temperature=${m.temperature.toFixed(3)} ` +
    `ECE before=${m.eceBefore?.toFixed(4)} after=${m.eceAfter?.toFixed(4)}\n`
);

const unseen: Array<{ note: string; ticket: unknown }> = [
  {
    note: "clear billing case",
    ticket: { subject: "Charged twice", body: "I was billed twice on my invoice, please refund the duplicate charge to my credit card." },
  },
  {
    note: "clear technical case",
    ticket: { subject: "App crash", body: "The app throws an exception and crashes with a stack trace on every timeout." },
  },
  {
    note: "ambiguous (billing + account words)",
    ticket: { subject: "Weird issue", body: "Please help, my invoice receipt login password account problem is urgent thanks." },
  },
  {
    note: "gibberish / out of domain",
    ticket: { subject: "asdf", body: "zxqv wobble fnord blorp." },
  },
];

for (const { note, ticket } of unseen) {
  const d = engine.ask<{ label: string }>("route_ticket", ticket);
  const arrow = d.action === "act" ? "→ AUTO-ROUTE" : d.action === "review" ? "→ HUMAN REVIEW" : "→ ESCALATE";
  console.log(`[${note}]`);
  console.log(`  route=${d.output.label} confidence=${d.confidence.toFixed(3)} action=${d.action} ${arrow}`);
}
