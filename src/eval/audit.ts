/**
 * npm run eval:audit
 *
 * Can the real-clip score be trusted? Answers from the labelled clips alone:
 *  - chance level: what a pipeline with no information scores;
 *  - sensitivity: does the score fall as ball detections get worse?
 *  - self-flattery: the reprojection term rewards fitting fewer points, so its
 *    share of any change is printed separately.
 */
import type { BallTouch } from "../utils/ballReconstruction";
import { realCases, type EvalCase } from "./dataset";
import { evaluateCase, realSummary, type CaseResult } from "./evaluate";
import { scoreClip } from "./score";
import type { Fixture } from "./types";

const labelled = realCases().filter((c) => c.truth);
const BALL_FIELDS = ["ball_img_pos", "ball_ref_pos", "ball_interpolated", "ball_conf", "ball_bbox",
  "ball_diameter_px", "ball_ref_diameter_px", "players_near_ball", "players_ref"];

function rng(seed: number) {
  return () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}
const clone = (fx: Fixture): Fixture => JSON.parse(JSON.stringify(fx));
type Frame = Fixture["frames"][number] & Record<string, unknown>;

/** Remove detections in bursts, as a detector loses a ball, until `share` is gone. */
function dropBursts(fx: Fixture, share: number, seed: number): Fixture {
  const out = clone(fx), r = rng(seed);
  const frames = out.frames as Frame[];
  const have = frames.filter((f) => f.ball_img_pos).length;
  let removed = 0, guard = 0;
  while (removed < have * share && guard++ < 10000) {
    const start = Math.floor(r() * frames.length);
    for (let i = start; i < Math.min(frames.length, start + 10); i++) {
      if (frames[i].ball_img_pos) removed++;
      for (const k of BALL_FIELDS) delete frames[i][k];
    }
  }
  return out;
}

/** Gaussian position noise on every detection, in pixels at 1920 wide. */
function jitter(fx: Fixture, px: number, seed: number): Fixture {
  const out = clone(fx), r = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r());
  for (const f of out.frames as Frame[]) {
    if (!f.ball_img_pos) continue;
    const dx = (gauss() * px) / 1920, dy = (gauss() * px) / 1080;
    for (const k of ["ball_img_pos", "ball_ref_pos"]) {
      const p = f[k] as [number, number] | undefined;
      if (p) f[k] = [p[0] + dx, p[1] + dy];
    }
  }
  return out;
}

/** Swap the ball on a share of frames for a confident detection somewhere else. */
function falseBalls(fx: Fixture, share: number, seed: number): Fixture {
  const out = clone(fx), r = rng(seed);
  for (const f of out.frames as Frame[]) {
    if (!f.ball_img_pos || r() > share) continue;
    const p: [number, number] = [r(), r()];
    f.ball_img_pos = p;
    if (f.ball_ref_pos) f.ball_ref_pos = p;
    delete f.ball_diameter_px;
    delete f.ball_ref_diameter_px;
  }
  return out;
}

function line(name: string, results: CaseResult[], base?: number) {
  const s = realSummary(results)!;
  const d = base == null ? "" : `  (${s.mean - base >= 0 ? "+" : ""}${(s.mean - base).toFixed(1)})`;
  console.log(`${name.padEnd(36)} ${s.mean.toFixed(1).padStart(5)}${d.padEnd(9)}  recall ${s.recall.toFixed(2)}  precision ${s.precision.toFixed(2)}`);
  return s.mean;
}

function evalWith(transform: (c: EvalCase, seed: number) => Fixture, seeds = 1): CaseResult[] {
  const out: CaseResult[] = [];
  for (const c of labelled) {
    for (let s = 0; s < seeds; s++) out.push(evaluateCase({ ...c, fixture: transform(c, 11 + s * 101) }));
  }
  return out;
}

/** A pipeline that knows nothing: the right number of touches, evenly spaced, mid-court. */
function nullModel(c: EvalCase, base: CaseResult): CaseResult {
  const run = base.run;
  if (!run || !c.truth) return base;
  const n = c.truth.touches.length, F = c.fixture.frames.length;
  const cycle = ["serve", "reception", "set", "attack"];
  const touches = Array.from({ length: n }, (_, i) => ({
    frame: Math.round(((i + 0.5) * F) / n), label: cycle[i % 4], position: [0, 1.5, 0],
  })) as unknown as BallTouch[];
  const fake = { ...run, result: { ...run.result, touches } };
  const { metrics, matches } = scoreClip(c.fixture, c.truth, fake);
  return { case: c, run: fake, metrics, matches };
}

console.log(`Labelled real clips: ${labelled.length}, touches: ${labelled.reduce((s, c) => s + c.truth!.touches.length, 0)}\n`);
const base = evalWith((c) => c.fixture);
const b = line("as exported", base);
const repeat = line("as exported, solved again", evalWith((c) => c.fixture), b);
console.log(repeat === b ? "  solver is deterministic\n" : "  solver is NOT deterministic: small differences are noise\n");

console.log("Chance level");
line("no information (right count, mid-court)", labelled.map((c, i) => nullModel(c, base[i])), b);

console.log("\nFewer detections (bursts of 10 frames)");
for (const p of [0.1, 0.25, 0.5, 0.75]) line(`${Math.round(p * 100)}% removed`, evalWith((c, s) => dropBursts(c.fixture, p, s), 2), b);

console.log("\nNoisier detections");
for (const px of [2, 5, 10, 20]) line(`${px} px noise`, evalWith((c, s) => jitter(c.fixture, px, s), 2), b);

console.log("\nFalse detections");
for (const p of [0.05, 0.15, 0.3]) line(`${Math.round(p * 100)}% replaced by random points`, evalWith((c, s) => falseBalls(c.fixture, p, s), 2), b);

console.log("\nPer clip: one touch is worth this many points of recall alone (touches differ a lot per clip)");
for (const r of base) {
  const n = r.case.truth!.touches.length;
  console.log(`  ${r.case.name.padEnd(12)} ${String(n).padStart(2)} touches  ${(25 / n).toFixed(1).padStart(5)} pts  score ${r.metrics!.score.toFixed(1)}`);
}
