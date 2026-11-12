/**
 * npm run eval [-- --only name1,name2] [-- --real] [-- --params file.json] [-- --save-baseline]
 * EVAL_COURT=detected scores real clips on the analyzer's court instead of the labelled one.
 *
 * Runs the 3D solver on every synthetic scenario and every real fixture,
 * prints a score table, and writes eval/report.html and eval/last.json.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { allCases, EVAL_DIR, HOLDOUT } from "./dataset";
import { aggregateScore, evaluateCase, groupSummaries, realSummary, type CaseResult, type GroupSummary, type RallyParamsOverride } from "./evaluate";
import { writeReport } from "./report";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

const only = arg("only")?.split(",");
const paramsFile = arg("params");
const params: RallyParamsOverride | undefined = paramsFile
  ? JSON.parse(fs.readFileSync(paramsFile, "utf8"))
  : undefined;

const cases = allCases().filter((c) => (!only || only.includes(c.name)) && !(process.argv.includes("--real") && c.synthetic));
const results: CaseResult[] = [];
for (const c of cases) {
  const r = evaluateCase(c, params);
  results.push(r);
  const m = r.metrics;
  const fmt = (v: number | null | undefined, d = 2) => (v == null || !Number.isFinite(v) ? "  -  " : v.toFixed(d).padStart(5));
  const tag = c.synthetic ? (c.knownLimit ? "limit" : "synth") : (!c.truth ? "real*" : HOLDOUT.includes(c.name) ? "hold " : "real ");
  console.log(
    `${tag} ${c.name.padEnd(18)} ` +
    (m
      ? `score ${m.score.toFixed(1).padStart(5)}  touches ${m.touchesMatched}/${m.touchesTrue} (found ${m.touchesFound})` +
        `  type ${fmt(m.typeAccuracy)}  pos ${fmt(m.touchPosErrM)} m  h ${fmt(m.touchHeightErrM)} m  dt ${fmt(m.touchFrameErr)} f` +
        `  path ${fmt(m.pathRmsM)} m  med3d ${fmt(m.path3dMedianM)} m  line ${fmt(m.trajCoverage)}  reproj ${fmt(m.reprojMedianPx, 1)} px on ${fmt(m.reprojCoverage)}  court ${fmt(m.courtErrPx, 1)} px`
      : `unlabelled: ${r.run?.result.touches?.length ?? 0} touches found`) +
    `  (${r.run?.millis ?? 0} ms)`,
  );
}
const total = aggregateScore(results);
console.log(`\nOverall score (labelled, excluding known limits): ${total.toFixed(1)}`);
const real = realSummary(results);
if (real) {
  console.log(
    `Real clips (${process.env.EVAL_COURT === "detected" ? "analyzer's court" : "labelled court"}): ${real.mean.toFixed(1)}` +
    `  95% CI ${real.ci95[0].toFixed(1)}-${real.ci95[1].toFixed(1)} over ${real.clips} clips` +
    `  touches: recall ${real.recall.toFixed(2)} precision ${real.precision.toFixed(2)}` +
    (real.courtErrPx != null ? `  court error ${real.courtErrPx.toFixed(1)} px` : ""),
  );
}

const groups = groupSummaries(results, HOLDOUT);
const pct = (v: number | null) => (v == null ? "-" : `${(100 * v).toFixed(0)}%`);
const n2 = (v: number | null, d = 2) => (v == null ? "-" : v.toFixed(d));
const line = (name: string, g: GroupSummary) => {
  if (!g.clips) return;
  console.log(
    `${name.padEnd(18)} ${g.clips} clips  score ${n2(g.score, 1)}  type ${pct(g.typeCorrect)}  recall ${pct(g.recall)}  precision ${pct(g.precision)}` +
    `  touch pos ${n2(g.touchPosMedianM)} m  dt ${n2(g.touchFrameErrMean)} (worst ${n2(g.touchFrameErrWorst)}) f` +
    `  reproj ${n2(g.reprojMedianPx, 1)} (worst ${n2(g.reprojWorstPx, 1)}) px on ${pct(g.reprojCoverage)}  line min ${pct(g.trajCoverageMin)}` +
    (g.pathRmsMedianM != null ? `  path RMS ${n2(g.pathRmsMedianM)} (worst ${n2(g.pathRmsWorstM)}) m  median 3D ${n2(g.path3dMedianM)} m` : ""),
  );
};
console.log("");
line("Synthetic (gated)", groups.synthetic);
line("Real, tuning set", groups.tuning);
line("Real, holdout", groups.holdout);

fs.mkdirSync(EVAL_DIR, { recursive: true });
const summary = {
  when: new Date().toISOString(),
  params: params ?? "defaults",
  overall: total,
  real,
  groups,
  cases: results.map((r) => ({ name: r.case.name, synthetic: r.case.synthetic, knownLimit: r.case.knownLimit, metrics: r.metrics })),
};
fs.writeFileSync(path.join(EVAL_DIR, "last.json"), JSON.stringify(summary, null, 2));
const reportPath = writeReport(results, total, path.join(EVAL_DIR, "report.html"));
if (process.argv.includes("--save-baseline")) {
  // The regression gate in eval.test.ts compares every case against this.
  const baseline = Object.fromEntries(results.filter((r) => r.metrics).map((r) => [r.case.name, r.metrics!.score]));
  fs.writeFileSync(path.join(EVAL_DIR, "baseline.json"), JSON.stringify(baseline, null, 2));
  console.log("Saved eval/baseline.json");
}
console.log(`Report: ${reportPath}`);
