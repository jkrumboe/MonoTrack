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

/**
 * Real clips are solved on their hand-placed court by default, which isolates
 * ball tracking from court detection. EVAL_COURT=detected uses the court the
 * analyzer found instead, as the app does, so court errors count.
 */
export function evaluateCase(c: EvalCase, params?: RallyParamsOverride): CaseResult {
  const detected = process.env.EVAL_COURT === "detected" && !c.synthetic;
  const run = runFixture(c.fixture, params, false, detected ? undefined : c.truth?.court);
  if (!c.truth) return { case: c, run, metrics: null, matches: [] };
  const { metrics, matches } = scoreClip(c.fixture, c.truth, run);
  return { case: c, run, metrics, matches };
}

export function realSummary(results: CaseResult[], resamples = 4000) {
  const real = results.filter((r) => r.metrics && !r.case.synthetic).map((r) => r.metrics!);
  if (!real.length) return null;
  const scores = real.map((m) => m.score);
  const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  // Fixed-seed bootstrap over clips: how far the mean moves with which clips happen to be labelled.
  let seed = 12345;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  const means = Array.from({ length: resamples }, () => avg(scores.map(() => scores[Math.floor(rnd() * scores.length)]))).sort((a, b) => a - b);
  const sum = (f: (m: (typeof real)[number]) => number) => real.reduce((s, m) => s + f(m), 0);
  const courts = real.flatMap((m) => (m.courtErrPx != null ? [m.courtErrPx] : []));
  return {
    clips: real.length,
    mean: avg(scores),
    ci95: [means[Math.floor(resamples * 0.025)], means[Math.floor(resamples * 0.975)]] as [number, number],
    // Pooled over touches: a clip with 16 touches counts for more than one with 2.
    recall: sum((m) => m.touchesMatched) / Math.max(1, sum((m) => m.touchesTrue)),
    precision: sum((m) => m.touchesMatched) / Math.max(1, sum((m) => m.touchesFound)),
    courtErrPx: courts.length ? avg(courts) : null,
  };
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

const med = (a: number[]) => {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const nums = (a: (number | null | undefined)[]) => a.filter((v): v is number => v != null && Number.isFinite(v));

/**
 * The numbers the acceptance criteria in EVAL.md are written in, over a group
 * of clips (gated synthetic scenarios, the tuning clips, the holdout clips).
 */
export function groupSummary(ms: ClipMetrics[]) {
  const sum = (f: (m: ClipMetrics) => number) => ms.reduce((s, m) => s + f(m), 0);
  const avg = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
  const max = (a: number[]) => (a.length ? Math.max(...a) : null);
  const min = (a: number[]) => (a.length ? Math.min(...a) : null);
  const T = sum((m) => m.touchesTrue), F = sum((m) => m.touchesFound), M = sum((m) => m.touchesMatched);
  return {
    clips: ms.length,
    score: avg(ms.map((m) => m.score)),
    touchesTrue: T, touchesFound: F, touchesMatched: M,
    recall: T ? M / T : null,
    precision: F ? M / F : null,
    /** Touches found at the right time with the right type, out of all true touches. */
    typeCorrect: T ? sum((m) => m.typeCorrect ?? 0) / T : null,
    touchPosMedianM: med(nums(ms.map((m) => m.touchPosErrM))),
    touchFrameErrMean: avg(nums(ms.map((m) => m.touchFrameErr))),
    touchFrameErrWorst: max(nums(ms.map((m) => m.touchFrameErr))),
    reprojMedianPx: med(nums(ms.map((m) => m.reprojMedianPx))),
    reprojWorstPx: max(nums(ms.map((m) => m.reprojMedianPx))),
    reprojCoverage: avg(nums(ms.map((m) => m.reprojCoverage))),
    trajCoverageMin: min(nums(ms.map((m) => m.trajCoverage))),
    pathRmsMedianM: med(nums(ms.map((m) => m.pathRmsM))),
    pathRmsWorstM: max(nums(ms.map((m) => m.pathRmsM))),
    path3dMedianM: med(nums(ms.map((m) => m.path3dMedianM))),
  };
}
export type GroupSummary = ReturnType<typeof groupSummary>;

export function groupSummaries(results: CaseResult[], holdout: string[]) {
  const pick = (f: (r: CaseResult) => boolean) => results.filter((r) => r.metrics && f(r)).map((r) => r.metrics!);
  return {
    synthetic: groupSummary(pick((r) => r.case.synthetic && !r.case.knownLimit)),
    tuning: groupSummary(pick((r) => !r.case.synthetic && !holdout.includes(r.case.name))),
    holdout: groupSummary(pick((r) => !r.case.synthetic && holdout.includes(r.case.name))),
  };
}
