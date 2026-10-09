/**
 * npm run eval [-- --only name1,name2] [-- --params file.json] [-- --save-baseline]
 *
 * Runs the 3D solver on every synthetic scenario and every real fixture,
 * prints a score table, and writes eval/report.html and eval/last.json.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { allCases, EVAL_DIR } from "./dataset";
import { aggregateScore, evaluateCase, type CaseResult, type RallyParamsOverride } from "./evaluate";
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

const cases = allCases().filter((c) => !only || only.includes(c.name));
const results: CaseResult[] = [];
for (const c of cases) {
  const r = evaluateCase(c, params);
  results.push(r);
  const m = r.metrics;
  const fmt = (v: number | null | undefined, d = 2) => (v == null || !Number.isFinite(v) ? "  -  " : v.toFixed(d).padStart(5));
  const tag = c.synthetic ? (c.knownLimit ? "limit" : "synth") : (c.truth ? "real " : "real*");
  console.log(
    `${tag} ${c.name.padEnd(18)} ` +
    (m
      ? `score ${m.score.toFixed(1).padStart(5)}  touches ${m.touchesMatched}/${m.touchesTrue} (found ${m.touchesFound})` +
        `  type ${fmt(m.typeAccuracy)}  pos ${fmt(m.touchPosErrM)} m  h ${fmt(m.touchHeightErrM)} m` +
        `  path ${fmt(m.pathRmsM)} m  cov ${fmt(m.pathCoverage)}  reproj ${fmt(m.reprojMedianPx, 1)} px`
      : `unlabelled: ${r.run?.result.touches?.length ?? 0} touches found`) +
    `  (${r.run?.millis ?? 0} ms)`,
  );
}
const total = aggregateScore(results);
console.log(`\nOverall score (labelled, excluding known limits): ${total.toFixed(1)}`);

fs.mkdirSync(EVAL_DIR, { recursive: true });
const summary = {
  when: new Date().toISOString(),
  params: params ?? "defaults",
  overall: total,
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
