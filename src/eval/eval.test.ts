import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { __test__ } from "../utils/ballReconstruction";
import { EVAL_DIR, HOLDOUT, realCases, syntheticCases, type EvalCase } from "./dataset";
import { aggregateScore, evaluateCase, groupSummaries, type CaseResult } from "./evaluate";
import { calibrate, cameraFor, projectNorm, runFixture } from "./pipeline";
import type { CourtLabel, Fixture } from "./types";

describe("helpers", () => {
  it("fits a parabola exactly", () => {
    const ts = [0, 0.1, 0.2, 0.3, 0.4];
    const ys = ts.map((t) => 1 + 2 * t - 4.9 * t * t);
    const [c0, c1, c2] = __test__.fitQuadratic(ts, ys);
    expect(c0).toBeCloseTo(1, 6);
    expect(c1).toBeCloseTo(2, 6);
    expect(c2).toBeCloseTo(-4.9, 6);
  });

  it("applies serve rules: reception on the other side, deep", () => {
    // Near side (z > 0) serves; reception candidates at the net and deep on the far side.
    const anchors: [number, number][][] = [[[-2.6, 12.5]], [[-2.5, 10]], [[-2, 0.3], [-2.6, -4.7]], [], []];
    const zones = __test__.applyRallyRules(anchors);
    expect(zones[0]?.label).toBe("toss");
    expect(zones[2]?.label).toBe("reception");
    expect(anchors[2]).toEqual([[-2.6, -4.7]]);
  });
});

describe("camera roll", () => {
  // 1 degree off level, as on real broadcast cameras.
  const truth = { position: [-24, 9, 1] as [number, number, number], lookAt: [0, 0.5, 0] as [number, number, number], fov: 30, roll: (1 * Math.PI) / 180 };
  const cam = cameraFor(truth, 16 / 9);
  const fx: Fixture = { clip: "t", video: { width: 1920, height: 1080, fps: 25 }, courtCorners: null, frames: [] };
  const world: [number, number][] = [[4.5, -9], [4.5, 0], [4.5, 9], [-4.5, -9], [-4.5, 0], [-4.5, 9]];
  const points = world.map(([x, z], i) => ({ id: i, name: "", world: [x, z] as [number, number], frame: 0, pt: projectNorm(cam, [x, 0, z]) }));

  it("is recovered, and the court then reprojects onto its points", () => {
    const c = calibrate(fx, { view: "side", points })!;
    expect(c.roll!).toBeCloseTo(truth.roll, 3);
    const solved = cameraFor(c, 16 / 9);
    for (const p of points) {
      const q = projectNorm(solved, [p.world[0], 0, p.world[1]]);
      expect(Math.hypot((q[0] - p.pt[0]) * 1920, (q[1] - p.pt[1]) * 1080)).toBeLessThan(0.5);
    }
  });

  it("is about zero for a level camera", () => {
    const level = cameraFor({ ...truth, roll: 0 }, 16 / 9);
    const pts = world.map(([x, z], i) => ({ id: i, name: "", world: [x, z] as [number, number], frame: 0, pt: projectNorm(level, [x, 0, z]) }));
    expect(Math.abs(calibrate(fx, { view: "side", points: pts })!.roll!)).toBeLessThan(1e-3);
  });
});

describe("hand-placed court", () => {
  // Side camera as label_touches.py assumes it: at -x, picture's left end z = -9.
  const truthCam = { position: [-24, 9, 1] as [number, number, number], lookAt: [0, 0.5, 0] as [number, number, number], fov: 30 };
  const cam = cameraFor(truthCam, 16 / 9);
  const fx: Fixture = { clip: "t", video: { width: 1920, height: 1080, fps: 25 }, courtCorners: null, frames: [] };
  const posErr = (c: ReturnType<typeof calibrate>) =>
    c ? Math.hypot(c.position[0] - truthCam.position[0], c.position[1] - truthCam.position[1], c.position[2] - truthCam.position[2]) : Infinity;

  it("calibrates from points without any corner (court not fully visible)", () => {
    const world: [number, number][] = [[4.5, -3], [4.5, 0], [4.5, 3], [-4.5, -3], [-4.5, 3]];
    const points = world.map(([x, z], i) => ({ id: i, name: "", world: [x, z] as [number, number], frame: 0, pt: projectNorm(cam, [x, 0, z]) }));
    expect(posErr(calibrate(fx, { view: "side", points }))).toBeLessThan(0.1);
  });

  it("reads old side-view corner labels without mirroring the court", () => {
    // Image order far-left, far-right, near-right, near-left.
    const corners = ([[4.5, -9], [4.5, 9], [-4.5, 9], [-4.5, -9]] as [number, number][]).map(([x, z]) => projectNorm(cam, [x, 0, z]));
    expect(posErr(calibrate(fx, { view: "side", frame: 0, corners }))).toBeLessThan(0.1);
  });
});

describe("France vs Japan fixture", () => {
  const fj = realCases().find((c) => c.name === "France_vs_Japan");
  it.skipIf(!fj)("keeps the reviewed rally reconstruction", () => {
    const r = evaluateCase(fj!);
    const t = r.run!.result.touches!;
    const byLabel = (l: string) => t.find((x) => x.label === l)!;
    // Values reviewed by eye against the video (see the explainer page).
    expect(byLabel("toss").position[2]).toBeGreaterThan(11);          // behind the near baseline
    expect(byLabel("reception").position[2]).toBeLessThan(-2);        // deep on the far side
    expect(Math.abs(byLabel("set").position[2])).toBeLessThan(1.5);   // at the net
    expect(Math.abs(byLabel("attack").position[2])).toBeLessThan(1.6);
    expect(Math.abs(byLabel("block").position[2])).toBeLessThan(1);
  });
});

describe("ball line", () => {
  // Between the first solved frame of play and the last touch, as the 3D view draws it.
  const holesInPlay = (fx: Fixture, c: { truth: { touches: { frame: number }[]; court?: CourtLabel } | null }) => {
    const pos = runFixture(fx, undefined, false, c.truth!.court)!.result.positions;
    const labelled = c.truth!.touches.map((t) => t.frame);
    const first = Math.max(Math.min(...labelled), Math.min(...pos.keys())), last = Math.max(...labelled);
    const holes: number[] = [];
    for (let f = first; f <= last; f++) if (!pos.has(f)) holes.push(f);
    return holes;
  };
  const labelled = realCases().filter((c) => c.truth?.touches.length);

  for (const c of labelled) {
    it(`${c.name} has no holes through the rally`, async () => {
      await new Promise((r) => setImmediate(r));
      expect(holesInPlay(c.fixture, c)).toEqual([]);
    });
  }

  it("runs through a serve the detector never saw, marked as estimated", () => {
    const c = labelled.find((x) => x.name === "pol_ita_r4");
    if (!c) return;
    const serve = c.truth!.touches.findIndex((t) => t.type === "serve");
    const [a, b] = [c.truth!.touches[serve].frame, c.truth!.touches[serve + 1].frame];
    const fx: Fixture = JSON.parse(JSON.stringify(c.fixture));
    for (const f of fx.frames) {
      if (f.frame <= a || f.frame >= b) continue;
      f.ball_img_pos = undefined; f.ball_ref_pos = undefined; f.ball_interpolated = undefined;
      f.ball_conf = undefined; f.ball_bbox = undefined; f.ball_diameter_px = undefined;
      f.ball_ref_diameter_px = undefined; f.players_near_ball = undefined;
    }
    expect(holesInPlay(fx, c)).toEqual([]);
    const { methods } = runFixture(fx, undefined, false, c.truth!.court)!.result;
    expect(methods.get(Math.round((a + b) / 2))).toBe("estimated");
  });
});

/** Each case is solved once and shared by the gates below. */
const solvedCases = new Map<string, CaseResult>();
const resultOf = (c: EvalCase): CaseResult => {
  if (!solvedCases.has(c.name)) solvedCases.set(c.name, evaluateCase(c));
  return solvedCases.get(c.name)!;
};

describe("regression gate", () => {
  const baselineFile = path.join(EVAL_DIR, "baseline.json");
  const baseline: Record<string, number> = fs.existsSync(baselineFile)
    ? JSON.parse(fs.readFileSync(baselineFile, "utf8"))
    : {};
  const cases = [...syntheticCases(), ...realCases()].filter((c) => c.truth && !c.knownLimit);

  for (const c of cases) {
    it.skipIf(baseline[c.name] == null)(`${c.name} does not regress`, async () => {
      // Real clips take up to ~15 s of synchronous solving; let the test
      // worker answer its runner first, or vitest reports an RPC timeout.
      await new Promise((r) => setImmediate(r));
      const score = resultOf(c).metrics!.score;
      // A few points of slack: solver seeds and float order can move a score slightly.
      expect(score).toBeGreaterThanOrEqual(baseline[c.name] - 3);
    });
  }

  it("catches a broken parameter", async () => {
    // Three scenarios are enough to show it, and the broken settings make big,
    // slow solves; yield between cases so the test worker stays responsive.
    const pick = new Set(["near-serve", "far-serve", "no-size"]);
    const synth = syntheticCases().filter((c) => pick.has(c.name));
    const good = [], bad = [];
    for (const c of synth) {
      good.push(evaluateCase(c));
      await new Promise((r) => setImmediate(r));
      // Accept every kink as a touch: the rally structure falls apart.
      bad.push(evaluateCase(c, { touchMinDvMps: 0, touchSplitM: 0.01 }));
      await new Promise((r) => setImmediate(r));
    }
    expect(aggregateScore(bad)).toBeLessThan(aggregateScore(good) - 5);
  });
});

describe("acceptance", () => {
  // The criteria of EVAL.md that are met, held there. Those not yet met
  // (worst synthetic path, synthetic touch position, coverage back at its
  // baseline) are listed in EVAL.md and not asserted.
  const cases = [...syntheticCases(), ...realCases()].filter((c) => c.truth && !c.knownLimit);
  const groups = async () => {
    const results: CaseResult[] = [];
    for (const c of cases) {
      results.push(resultOf(c));
      await new Promise((r) => setImmediate(r));
    }
    return { results, ...groupSummaries(results, HOLDOUT) };
  };

  it("synthetic scenarios, tripod included", async () => {
    const { results, synthetic } = await groups();
    expect(synthetic.pathRmsMedianM!).toBeLessThanOrEqual(1.0);
    expect(synthetic.typeCorrect!).toBeGreaterThanOrEqual(0.9);
    // A trajectory on every frame between the first and the last contact.
    for (const r of results) if (r.case.synthetic) expect(r.metrics!.trajCoverage, r.case.name).toBe(1);
  });

  it("real clips, tuning set", async () => {
    const { tuning } = await groups();
    expect(tuning.touchFrameErrMean!).toBeLessThanOrEqual(1.0);
    expect(tuning.touchFrameErrWorst!).toBeLessThanOrEqual(2.0);
    expect(tuning.reprojMedianPx!).toBeLessThanOrEqual(3.0);
    expect(tuning.reprojWorstPx!).toBeLessThanOrEqual(6.0);
    expect(tuning.typeCorrect!).toBeGreaterThanOrEqual(0.85);
    expect(tuning.precision!).toBeGreaterThanOrEqual(0.85);
    expect(tuning.touchPosMedianM!).toBeLessThanOrEqual(1.2);
  });

  it("reprojection coverage stays above its floor", async () => {
    // A low reprojection error is easy to get by fitting fewer detections.
    // The floor is the lower of the 2026-11-12 baseline and the value when
    // the gate was added (eval/baseline-coverage.json); it only goes up.
    const file = path.join(EVAL_DIR, "baseline-coverage.json");
    const floors: Record<string, { baseline: number; floor: number }> = JSON.parse(fs.readFileSync(file, "utf8"));
    const { results } = await groups();
    for (const r of results) {
      const f = floors[r.case.name];
      if (f) expect(r.metrics!.reprojCoverage!, r.case.name).toBeGreaterThanOrEqual(f.floor - 0.01);
    }
  });
});
