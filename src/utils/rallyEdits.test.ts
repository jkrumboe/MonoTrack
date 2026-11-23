import { describe, expect, it } from "vitest";
import { syntheticCases } from "../eval/dataset";
import { cameraFor, projectNorm, runFixture } from "../eval/pipeline";
import type { FrameData } from "../hooks/useAnalysis";
import { solveBallTrajectories } from "./ballReconstruction";
import { applyBallEdits, ballRay, pinsAfterBallMark, touchEditsForSolver } from "./rallyEdits";

type V3 = [number, number, number];

const cal = { position: [0, 8, 22] as V3, lookAt: [0, 0, 0] as V3, fov: 45, roll: 0 };
const aspect = 16 / 9;
const frames = Array.from({ length: 10 }, (_, i) => ({ frame: i, timestamp: i / 30 }) as FrameData);
const project = (p: V3) => projectNorm(cameraFor({ ...cal } as never, aspect), p);
const distToRay = (p: V3, r: { origin: V3; dir: V3 }) => {
  const d = [0, 1, 2].map((c) => p[c] - r.origin[c]);
  const t = d[0] * r.dir[0] + d[1] * r.dir[1] + d[2] * r.dir[2];
  return Math.hypot(...[0, 1, 2].map((c) => d[c] - r.dir[c] * t));
};

describe("ball placed on a touch held in 3D", () => {
  it("moves the touch onto the new line of sight, keeping its depth", () => {
    const pin: V3 = [1, 2.5, 3];
    const mark = project([1.4, 2.8, 3]);
    const pins = pinsAfterBallMark({ touches: { 4: "set" }, pins: { 4: pin } }, 4, mark, frames, new Map(), cal, aspect);
    const q = pins[4];
    const ray = ballRay(applyBallEdits(frames, { 4: mark }), 4, cal, aspect)!;
    expect(distToRay(q, ray)).toBeLessThan(1e-6);
    // Same distance from the camera, near enough: only the image position changed.
    const dist = (p: V3) => Math.hypot(p[0] - cal.position[0], p[1] - cal.position[1], p[2] - cal.position[2]);
    expect(Math.abs(dist(q) - dist(pin))).toBeLessThan(0.2);
  });

  it("snaps a net touch onto the net again", () => {
    const mark = project([2, 2.6, 0]);
    const pins = pinsAfterBallMark({ touches: { 4: "net" }, pins: { 4: [0, 2.4, 0] } }, 4, mark, frames, new Map(), cal, aspect);
    expect(pins[4][2]).toBeCloseTo(0, 6);
    expect(pins[4][0]).toBeCloseTo(2, 3);
    expect(pins[4][1]).toBeCloseTo(2.6, 3);
  });

  it("leaves pins on other frames, and touches not held in 3D, alone", () => {
    const e = { touches: { 4: "set" as const }, pins: { 6: [0, 2, 1] as V3 } };
    expect(pinsAfterBallMark(e, 4, [0.5, 0.5], frames, new Map(), cal, aspect)).toBe(e.pins);
  });

  it("tells the solver which marked touches have the ball on their frame", () => {
    const te = touchEditsForSolver({ 4: "set", 9: "attack", 12: "none" }, {}, { 4: [0.5, 0.5], 9: null, 12: [0.2, 0.2] });
    expect(te?.ballAt).toEqual([4]);
  });
});

describe("solver: a touch marked with the ball on its frame", () => {
  it("is timed on that frame and sits on that ball in the image", () => {
    const c = syntheticCases().find((x) => !!x.truth && x.truth.touches.filter((t) => t.position).length >= 3)!;
    const fx = c.fixture, truth = c.truth!;
    const run = runFixture(fx, undefined, false, truth.court)!;
    const cam = cameraFor(run.calibration, fx.video.width / fx.video.height);
    // Every touch after the first, marked by hand, the ball clicked where it truly was.
    const marked = truth.touches.slice(1).filter((t) => t.position);
    const ball: Record<number, [number, number]> = {};
    for (const t of marked) ball[t.frame] = projectNorm(cam, t.position!);
    const touches = Object.fromEntries(marked.map((t) => [t.frame, t.type] as const)) as Record<number, never>;

    const solve = (withBall: boolean) => {
      const log = console.log, warn = console.warn;
      console.log = () => {}; console.warn = () => {};
      try {
        return solveBallTrajectories(applyBallEdits(fx.frames, ball), run.calibration, fx.video.width / fx.video.height, {
          useBallisticSolver: false, debugMode: true, diameterMode: "min-side",
          videoWidth: fx.video.width, videoHeight: fx.video.height, trajectoryMode: "physics-reprojection",
          useCalibrationFov: true, manualFovYDeg: 30, minBallConfidence: 0.6,
          touchEdits: touchEditsForSolver(touches, {}, withBall ? ball : {}),
        });
      } finally { console.log = log; console.warn = warn; }
    };
    // Image error (px) of the solved touch nearest each marked one, and its frame.
    const errs = (res: ReturnType<typeof solve>) => marked.map((m) => {
      const t = (res.touches ?? []).reduce((b, x) => (Math.abs(x.frame - m.frame) < Math.abs(b.frame - m.frame) ? x : b));
      const [u, v] = projectNorm(cam, t.position);
      return { frameOff: t.frame - m.frame, px: Math.hypot((u - ball[m.frame][0]) * fx.video.width, (v - ball[m.frame][1]) * fx.video.height) };
    });
    const without = errs(solve(false)), withBall = errs(solve(true));
    for (const e of withBall) {
      expect(e.frameOff).toBe(0);
      expect(e.px).toBeLessThan(10);
    }
    const mean = (a: { px: number }[]) => a.reduce((s, x) => s + x.px, 0) / a.length;
    // near-serve: 19 px mean without the ball, 3 px with it.
    expect(mean(withBall)).toBeLessThan(0.5 * mean(without));
  }, 120_000);
});
