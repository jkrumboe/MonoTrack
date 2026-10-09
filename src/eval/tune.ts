/**
 * npm run tune [-- --iters 40] [-- --sensitivity] [-- --seed 1]
 *
 * Searches the rally solver's parameters for the best combined score.
 *   train   = synthetic scenarios + every other labelled real clip
 *   holdout = the remaining labelled real clips (reported, never tuned on)
 * Known-limit scenarios are reported only. Writes eval/tuned-params.json
 * (use with `npm run eval -- --params eval/tuned-params.json`) and
 * eval/tune-report.json. Adopting new defaults is a manual step.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { DEFAULT_RALLY_PARAMS, type RallyParams } from "../utils/ballReconstruction";
import { EVAL_DIR, realCases, syntheticCases, type EvalCase } from "./dataset";
import { aggregateScore, evaluateCase, type RallyParamsOverride } from "./evaluate";

interface Spec { key: string; min: number; max: number; log?: boolean; int?: boolean }

/** The parameters worth searching, with ranges that stay physically sensible. */
const SPECS: Spec[] = [
  { key: "touchMinDvMps", min: 1.8, max: 6.4 },
  { key: "touchSplitM", min: 0.04, max: 0.12 },
  { key: "rallyMaxGapS", min: 0.5, max: 2.0 },
  { key: "touchImpulseWeight", min: 0.05, max: 1, log: true },
  { key: "sizeDepthWeight", min: 2, max: 30, log: true },
  { key: "anchorWeight", min: 4, max: 30, log: true },
  { key: "anchorRobustM", min: 1.5, max: 6 },
  { key: "knotImageWeight", min: 0.1, max: 1 },
  { key: "robustReprojPx", min: 2, max: 10 },
  { key: "segMaxMedianErrPx", min: 6, max: 16 },
  { key: "softMaxSpeed", min: 24, max: 36 },
  { key: "rules.attack.maxAbsZ", min: 0.8, max: 3 },
  { key: "rules.set.maxAbsZ", min: 2.5, max: 5 },
  { key: "rules.reception.minAbsZ", min: 1, max: 4 },
];

type Values = Record<string, number>;

function getDefault(key: string): number {
  return key.split(".").reduce((o: any, k) => o[k], DEFAULT_RALLY_PARAMS) as number;
}

/** Flat values -> the override shape solveBallTrajectories accepts. */
function toOverride(v: Values): RallyParamsOverride {
  const out: any = {};
  const rules: any = {};
  for (const [key, val] of Object.entries(v)) {
    const parts = key.split(".");
    if (parts[0] === "rules") {
      const touch = parts[1] as keyof RallyParams["rules"];
      rules[touch] = { ...(rules[touch] ?? DEFAULT_RALLY_PARAMS.rules[touch]), [parts[2]]: val };
    } else {
      out[key] = val;
    }
  }
  if (Object.keys(rules).length) out.rules = rules;
  return out;
}

function arg(name: string, dflt: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? Number(process.argv[i + 1]) : dflt;
}

// Seeded RNG so a tuning run can be reproduced.
let seed = arg("seed", 1) >>> 0;
const uni = () => {
  seed = (seed + 0x6d2b79f5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const sample = (s: Spec) => (s.log ? Math.exp(Math.log(s.min) + uni() * (Math.log(s.max) - Math.log(s.min))) : s.min + uni() * (s.max - s.min));
const clamp = (s: Spec, x: number) => Math.min(s.max, Math.max(s.min, x));

const synth = syntheticCases();
const real = realCases().filter((c) => c.truth);
const train: EvalCase[] = [...synth.filter((c) => !c.knownLimit), ...real.filter((_, i) => i % 2 === 0)];
const holdout: EvalCase[] = real.filter((_, i) => i % 2 === 1);
const limits: EvalCase[] = synth.filter((c) => c.knownLimit);

const scoreOn = (cases: EvalCase[], v: Values) => aggregateScore(cases.map((c) => evaluateCase(c, toOverride(v))), true);

const defaults: Values = Object.fromEntries(SPECS.map((s) => [s.key, getDefault(s.key)]));
const t0 = Date.now();
const baseTrain = scoreOn(train, defaults);
console.log(`train: ${train.length} cases, holdout: ${holdout.length}, known limits: ${limits.length}`);
console.log(`defaults: train ${baseTrain.toFixed(2)}  (${((Date.now() - t0) / 1000).toFixed(1)} s per evaluation)`);

let best = { v: defaults, score: baseTrain };
const iters = arg("iters", 40);
// Half random exploration, half local refinement around the best so far.
for (let i = 0; i < iters; i++) {
  const v: Values = {};
  const local = i >= iters / 2;
  for (const s of SPECS) {
    if (!local) v[s.key] = sample(s);
    else {
      const span = (s.max - s.min) * 0.15;
      v[s.key] = uni() < 0.35 ? clamp(s, best.v[s.key] + (uni() * 2 - 1) * span) : best.v[s.key];
    }
  }
  const score = scoreOn(train, v);
  if (score > best.score) {
    best = { v, score };
    console.log(`  #${i + 1} ${local ? "refine" : "random"}: train ${score.toFixed(2)}  ← new best`);
  }
}

const changed = SPECS.filter((s) => Math.abs(best.v[s.key] - defaults[s.key]) > 1e-9)
  .map((s) => `  ${s.key}: ${defaults[s.key]} → ${+best.v[s.key].toFixed(3)}`);
console.log(`\nbest train ${best.score.toFixed(2)} (defaults ${baseTrain.toFixed(2)})`);
console.log(changed.length ? changed.join("\n") : "  defaults were best");
const hold = holdout.length ? { defaults: scoreOn(holdout, defaults), tuned: scoreOn(holdout, best.v) } : null;
if (hold) console.log(`holdout: defaults ${hold.defaults.toFixed(2)} → tuned ${hold.tuned.toFixed(2)}`);
else console.log("holdout: none yet (label more real clips to get an honest check)");
const lim = limits.length ? { defaults: scoreOn(limits, defaults), tuned: scoreOn(limits, best.v) } : null;
if (lim) console.log(`known limits: defaults ${lim.defaults.toFixed(2)} → tuned ${lim.tuned.toFixed(2)}`);

// Sensitivity: how much the train score moves when one parameter goes to an end of its range.
let sensitivity: { key: string; atMin: number; atMax: number }[] | undefined;
if (process.argv.includes("--sensitivity")) {
  sensitivity = SPECS.map((s) => ({
    key: s.key,
    atMin: scoreOn(train, { ...best.v, [s.key]: s.min }) - best.score,
    atMax: scoreOn(train, { ...best.v, [s.key]: s.max }) - best.score,
  })).sort((a, b) => Math.min(a.atMin, a.atMax) - Math.min(b.atMin, b.atMax));
  console.log("\nsensitivity (score change at range ends, most damaging first):");
  for (const s of sensitivity) console.log(`  ${s.key.padEnd(26)} min ${s.atMin.toFixed(1).padStart(6)}  max ${s.atMax.toFixed(1).padStart(6)}`);
}

fs.mkdirSync(EVAL_DIR, { recursive: true });
fs.writeFileSync(path.join(EVAL_DIR, "tuned-params.json"), JSON.stringify(toOverride(best.v), null, 2));
fs.writeFileSync(path.join(EVAL_DIR, "tune-report.json"), JSON.stringify({
  when: new Date().toISOString(), iters, seed: arg("seed", 1),
  train: { defaults: baseTrain, tuned: best.score }, holdout: hold, knownLimits: lim,
  values: best.v, sensitivity,
}, null, 2));
console.log(`\nWrote eval/tuned-params.json and eval/tune-report.json (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
