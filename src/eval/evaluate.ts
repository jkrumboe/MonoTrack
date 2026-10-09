import type { BallReconstructionOptions } from "../utils/ballReconstruction";
import type { EvalCase } from "./dataset";
import { runFixture, type PipelineRun } from "./pipeline";
import { scoreClip, type TouchMatch } from "./score";
import type { ClipMetrics } from "./types";

export interface CaseResult {
  case: EvalCase;
  run: PipelineRun | null;
  /** Null for unlabelled real clips. */
  metrics: ClipMetrics | null;
  matches: TouchMatch[];
}

export type RallyParamsOverride = BallReconstructionOptions["rallyParams"];

export function evaluateCase(c: EvalCase, params?: RallyParamsOverride): CaseResult {
  const run = runFixture(c.fixture, params, false, c.truth?.court);
  if (!c.truth) return { case: c, run, metrics: null, matches: [] };
  const { metrics, matches } = scoreClip(c.fixture, c.truth, run);
  return { case: c, run, metrics, matches };
}

/**
 * Mean score over labelled cases. Documented known limits are left out
 * unless asked for, so they don't drag the regression gate.
 */
export function aggregateScore(results: CaseResult[], includeKnownLimits = false): number {
  const s = results
    .filter((r) => r.metrics && (includeKnownLimits || !r.case.knownLimit))
    .map((r) => r.metrics!.score);
  return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
}
