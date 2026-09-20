/**
 * @gavel/decide — the open typed-decision layer for AI agents.
 *
 * Structured decisions in, calibrated answers out. M1: the native binding
 * trains a real logistic-regression engine per question (`train`), with
 * temperature-scaling calibration (`calibrate`) in M2. The `define` / `ask`
 * API is stable.
 */

import * as path from "node:path";

// Resolve the native binding. `DECIDE_BINDING` overrides for custom layouts;
// default assumes the decide-napi package was built in the monorepo.
function loadBinding(): any {
  const override = process.env.DECIDE_BINDING;
  const candidates = override
    ? [override]
    : [
        // ts/decide/dist/src -> gavel root is 4 levels up
        path.resolve(__dirname, "..", "..", "..", "..", "crates", "decide-napi", `decide.${process.platform}-${process.arch}.node`),
        path.resolve(__dirname, "..", "..", "..", "..", "crates", "decide-napi", "decide.linux-x64-gnu.node"),
      ];
  let lastErr: unknown = null;
  for (const c of candidates) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      return require(c);
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error(
    `Could not load decide native binding. Tried:\n  ${candidates.join("\n  ")}\n` +
      `Build it with: cd crates/decide-napi && npm run build\nOriginal error: ${lastErr}`
  );
}

const binding = loadBinding();

export type Action = "act" | "review" | "escalate";

export interface Policy {
  /** Act when confidence >= actAbove (default 0.9). */
  actAbove: number;
  /** Lower bound of the review band (default 0.6). */
  reviewLow: number;
  /** Upper bound of the review band (default 0.9). */
  reviewHigh: number;
}

export const DEFAULT_POLICY: Policy = { actAbove: 0.9, reviewLow: 0.6, reviewHigh: 0.9 };

export interface QuestionDef {
  name: string;
  /** JSON Schema for the input. */
  inputSchema: Record<string, unknown>;
  /** JSON Schema for the output (decision shape). */
  outputSchema: Record<string, unknown>;
  policy?: Partial<Policy>;
}

export interface Decision<TOutput = unknown> {
  output: TOutput;
  /** Calibrated confidence in [0, 1]. */
  confidence: number;
  action: Action;
}

export interface EngineOptions {
  /** M0 stub: fixed decision payload returned for every question. */
  mockOutput?: unknown;
  /** M0 stub: fixed confidence returned for every question. */
  mockConfidence?: number;
}

export interface TrainingExample {
  /** String or object; object inputs contribute their string values. */
  input: unknown;
  label: string;
}

export interface EngineMetrics {
  trained: boolean;
  classes: string[];
  temperature: number;
  /** Uncalibrated ECE (train set at train time; validation set after calibrate). */
  eceBefore: number | null;
  /** Calibrated ECE on the validation set (null until calibrated). */
  eceAfter: number | null;
}

/** A decision engine: register typed questions, then ask them. */
export class Engine {
  private native: any;

  constructor(opts: EngineOptions = {}) {
    this.native = new binding.DecideEngine(
      opts.mockOutput !== undefined ? JSON.stringify(opts.mockOutput) : undefined,
      opts.mockConfidence
    );
  }

  /** Register a typed question with the engine. */
  define(q: QuestionDef): void {
    const policy: Policy = { ...DEFAULT_POLICY, ...(q.policy ?? {}) };
    this.native.defineQuestion(
      q.name,
      JSON.stringify(q.inputSchema),
      JSON.stringify(q.outputSchema),
      JSON.stringify({
        act_above: policy.actAbove,
        review_low: policy.reviewLow,
        review_high: policy.reviewHigh,
      })
    );
  }

  /** Ask a registered question. Returns the typed decision. */
  ask<TOutput = unknown>(questionName: string, input: unknown): Decision<TOutput> {
    const raw: string = this.native.ask(questionName, JSON.stringify(input));
    return JSON.parse(raw) as Decision<TOutput>;
  }

  /** Train a question's engine on labeled examples. */
  train(questionName: string, examples: TrainingExample[]): void {
    this.native.train(questionName, JSON.stringify(examples));
  }

  /** Fit temperature scaling on held-out validation examples. */
  calibrate(questionName: string, validation: TrainingExample[]): void {
    this.native.calibrate(questionName, JSON.stringify(validation));
  }

  /** Training/calibration status for a question. */
  metrics(questionName: string): EngineMetrics {
    const raw: string = this.native.metrics(questionName);
    const m = JSON.parse(raw);
    return {
      trained: m.trained,
      classes: m.classes,
      temperature: m.temperature,
      eceBefore: m.ece_before ?? null,
      eceAfter: m.ece_after ?? null,
    };
  }

  /** Names of registered questions. */
  questions(): string[] {
    return this.native.questions() as string[];
  }
}

/** Convenience: create an engine with a single question pre-registered. */
export function define(q: QuestionDef, engineOpts: EngineOptions = {}): {
  engine: Engine;
  ask: (input: unknown) => Decision;
} {
  const engine = new Engine(engineOpts);
  engine.define(q);
  return { engine, ask: (input: unknown) => engine.ask(q.name, input) };
}
