import { test } from "node:test";
import assert from "node:assert/strict";
import { Engine, DEFAULT_POLICY } from "../src/index";

function trainedEngine(): Engine {
  const engine = new Engine();
  engine.define({
    name: "q",
    inputSchema: { type: "object" },
    outputSchema: { type: "object" },
    policy: { actAbove: 0.9, reviewLow: 0.6, reviewHigh: 0.9 },
  });
  const examples = [
    { input: "invoice billed twice refund", label: "billing" },
    { input: "charge on my credit card statement", label: "billing" },
    { input: "please refund my invoice", label: "billing" },
    { input: "duplicate charge on the invoice", label: "billing" },
    { input: "billing statement shows wrong amount", label: "billing" },
    { input: "reset my password login", label: "account" },
    { input: "cannot sign in to account", label: "account" },
    { input: "locked out of my login", label: "account" },
    { input: "change my account password", label: "account" },
    { input: "account sign in not working", label: "account" },
  ];
  engine.train("q", examples);
  return engine;
}

test("define/train/ask round trip", () => {
  const engine = trainedEngine();
  const d = engine.ask<{ label: string }>("q", "please refund my invoice");
  assert.equal(d.output.label, "billing");
  assert.ok(d.confidence >= 0 && d.confidence <= 1);
  assert.equal(d.action, "act");
});

test("calibrate records temperature and ECE", () => {
  const engine = trainedEngine();
  engine.calibrate("q", [
    { input: "duplicate charge on invoice", label: "billing" },
    { input: "forgot password need reset", label: "account" },
  ]);
  const m = engine.metrics("q");
  assert.ok(m.trained);
  assert.deepEqual(m.classes, ["account", "billing"]);
  assert.ok(m.temperature > 0);
  assert.ok(m.eceBefore !== null);
  assert.ok(m.eceAfter !== null);
});

test("questions lists defined names", () => {
  const engine = trainedEngine();
  assert.deepEqual(engine.questions(), ["q"]);
});

test("asking an unknown question throws", () => {
  const engine = new Engine();
  assert.throws(() => engine.ask("nope", {}));
});

test("default policy matches the documented thresholds", () => {
  assert.deepEqual(DEFAULT_POLICY, { actAbove: 0.9, reviewLow: 0.6, reviewHigh: 0.9 });
});
