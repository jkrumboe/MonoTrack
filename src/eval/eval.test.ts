import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { __test__ } from "../utils/ballReconstruction";
import { EVAL_DIR, realCases, syntheticCases } from "./dataset";
import { aggregateScore, evaluateCase } from "./evaluate";
import { calibrate, cameraFor, projectNorm } from "./pipeline";
import type { Fixture } from "./types";

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
      const score = evaluateCase(c).metrics!.score;
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
