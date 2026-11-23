import { describe, expect, it } from "vitest";
import { realCases, syntheticCases } from "../eval/dataset";
import { cameraFor, projectNorm, runFixture } from "../eval/pipeline";
import type { Fixture, Truth } from "../eval/types";
import {
  ARC_MAX_G, ARC_MIN_G, gravityRange, historyOf, historyPush, historyRedo, historyUndo, refitEditedPath, resetSegmentEdits,
  type RefitInput, type RefitResult, type RefitTouch, type V3,
} from "./rallyRefit";

const G = 9.81;

/** A solved rally with what the editor passes to the re-fit. */
function solved(fx: Fixture, truth: Truth | null) {
  const run = runFixture(fx, undefined, false, truth?.court)!;
  const cam = cameraFor(run.calibration, fx.video.width / fx.video.height);
  const observations = new Map<number, [number, number]>();
  for (const f of fx.frames) {
    const p = fx.cameraMoving ? f.ball_ref_pos : f.ball_img_pos;
    if (p && !f.ball_interpolated) observations.set(f.frame, [p[0] * 2 - 1, 1 - p[1] * 2]);
  }
  const project = (p: V3): [number, number] => { const [u, v] = projectNorm(cam, p); return [u * 2 - 1, 1 - v * 2]; };
  const base: RefitInput = {
    touches: (run.result.touches ?? []).map((t) => ({ ...t })),
    positions: run.result.positions, methods: run.result.methods, fps: fx.video.fps, observations, project,
  };
  return { base, fps: fx.video.fps };
}

describe("touches too far apart for one flight", () => {
  it("are not joined by an arc", () => {
    const fps = 25;
    const at = (frame: number, z: number): RefitTouch => ({ frame, position: [0, 2, z], label: "attack" });
    // One rally ends at frame 100; the next starts 20 s later, with no ball solved between.
    const r = refitEditedPath({ touches: [at(100, -3), at(600, 8)], positions: new Map(), fps });
    expect(r.segments).toHaveLength(0);
    expect(r.positions.size).toBe(0);
  });
});

const synth = syntheticCases().find((c) => c.name === "tripod-long-30")!;
const long = realCases().find((c) => c.name === "usa_jap_r3");

/** A physically valid edited path: see each expectation. */
function expectValid(r: RefitResult, touches: RefitTouch[], fps: number) {
  const first = touches[0].frame, last = touches[touches.length - 1].frame;
  // Every frame between the first and the last touch has a position.
  for (let f = first; f <= last; f++) expect(r.positions.has(f), `frame ${f} has a position`).toBe(true);
  for (const s of r.segments) {
    if (!s.rebuilt) continue;
    const g = s.g!;
    // Ballistic: falls at one rate, within what a ball can do.
    expect(g).toBeGreaterThanOrEqual(ARC_MIN_G * G - 1e-9);
    expect(g).toBeLessThanOrEqual(ARC_MAX_G * G + 1e-9);
    for (let f = s.from + 1; f < s.to; f++) {
      const a = r.positions.get(f - 1)!, b = r.positions.get(f)!, c = r.positions.get(f + 1)!;
      // Straight and steady over the floor.
      expect(Math.abs(a[0] - 2 * b[0] + c[0])).toBeLessThan(1e-6);
      expect(Math.abs(a[2] - 2 * b[2] + c[2])).toBeLessThan(1e-6);
      // Constant fall (unless the floor clamps it).
      if (Math.min(a[1], b[1], c[1]) > 0.12) expect(a[1] - 2 * b[1] + c[1]).toBeCloseTo(-g / (fps * fps), 6);
      expect(b[1]).toBeGreaterThanOrEqual(0.1);
    }
  }
  // Continuous at the touches: a rebuilt flight starts and ends exactly on them.
  for (const s of r.segments) {
    if (!s.rebuilt) continue;
    const a = touches.find((t) => t.frame === s.from)!, b = touches.find((t) => t.frame === s.to)!;
    expect(r.positions.get(s.from)).toEqual(a.position);
    expect(r.positions.get(s.to)).toEqual(b.position);
  }
}

describe("editor re-fit", () => {
  const { base, fps } = solved(synth.fixture, synth.truth);
  const touches = base.touches;

  it("leaves the automatic result alone when nothing is edited", () => {
    const r = refitEditedPath(base);
    expect(r.rebuiltFrames).toBe(0);
    for (const [f, p] of base.positions) expect(r.positions.get(f)).toEqual(p);
    expect(r.segments.length).toBe(touches.length - 1);
  });

  it("moving a touch rebuilds the two flights at it and nothing else", () => {
    const k = 3;
    const moved = touches.map((t, i) => (i === k ? { ...t, position: [t.position[0] + 1.2, t.position[1] + 0.4, t.position[2] - 0.8] as V3, pinned: true, edited: true } : t));
    const r = refitEditedPath({ ...base, touches: moved });
    expect(r.segments.map((s) => s.rebuilt)).toEqual(touches.slice(1).map((_, i) => i === k - 1 || i === k));
    expectValid(r, moved, fps);
    // The path goes through where the touch was put.
    expect(r.positions.get(moved[k].frame)).toEqual(moved[k].position);
    // Frames outside the two flights are the solver's.
    for (const [f, p] of base.positions) {
      if (f < moved[k - 1].frame || f > moved[k + 1].frame) expect(r.positions.get(f)).toEqual(p);
    }
  });

  it("a rebuilt flight follows the detections", () => {
    // Moved by a hair: the flight is rebuilt, and should land where the solver had it.
    const k = 2;
    const moved = touches.map((t, i) => (i === k ? { ...t, position: [t.position[0], t.position[1] + 0.06, t.position[2]] as V3, pinned: true } : t));
    const r = refitEditedPath({ ...base, touches: moved });
    const s = r.segments[k];
    expect(s.rebuilt).toBe(true);
    const mf = Math.round((s.from + s.to) / 2);
    const was = base.positions.get(mf)!, now = r.positions.get(mf)!;
    expect(Math.hypot(was[0] - now[0], was[1] - now[1], was[2] - now[2])).toBeLessThan(0.3);
  });

  it("an added touch on the path changes nothing until it is moved", () => {
    const a = touches[2], b = touches[3];
    const f = Math.round((a.frame + b.frame) / 2);
    const added: RefitTouch = { frame: f, position: base.positions.get(f)!, label: "touch", edited: true };
    const withAdded = [...touches.slice(0, 3), added, ...touches.slice(3)];
    expect(refitEditedPath({ ...base, touches: withAdded }).rebuiltFrames).toBe(0);
    const lifted = withAdded.map((t) => (t === added ? { ...t, position: [t.position[0], t.position[1] + 1, t.position[2]] as V3, pinned: true } : t));
    const r = refitEditedPath({ ...base, touches: lifted });
    expect(r.segments.filter((s) => s.rebuilt).map((s) => s.from)).toEqual([a.frame, f]);
    expectValid(r, lifted, fps);
  });

  it("removing a touch makes one arc of the two flights at it", () => {
    const k = 3;
    const without = touches.filter((_, i) => i !== k);
    const r = refitEditedPath({ ...base, touches: without, removedAt: [touches[k].frame] });
    const s = r.segments.find((x) => x.from === touches[k - 1].frame)!;
    expect(s.to).toBe(touches[k + 1].frame);
    expect(s.rebuilt).toBe(true);
    expectValid(r, without, fps);
  });

  it("the touch type bounds how the rebuilt flight falls", () => {
    const k = 2;
    for (const label of ["set", "attack", "reception", "dig", "serve"]) {
      const typed = touches.map((t, i) => (i === k ? { ...t, label, position: [t.position[0], t.position[1] + 0.2, t.position[2]] as V3, pinned: true } : t));
      const s = refitEditedPath({ ...base, touches: typed }).segments[k];
      const [lo, hi] = gravityRange(label);
      expect(s.g! / G).toBeGreaterThanOrEqual(lo - 1e-6);
      expect(s.g! / G).toBeLessThanOrEqual(hi + 1e-6);
    }
    // A set is a soft arc; an attack may dive.
    expect(gravityRange("set")[1]).toBeLessThan(gravityRange("attack")[1]);
  });

  it("the arc handle sets the height of a flight, within what a ball can do", () => {
    const k = 2, from = touches[k].frame;
    const auto = refitEditedPath(base).segments[k].mid[1];
    const r = refitEditedPath({ ...base, arcs: { [from]: auto + 1.5 } });
    expect(r.segments[k].manual).toBe(true);
    expect(r.segments[k].mid[1]).toBeCloseTo(auto + 1.5, 1);
    expect(r.segments.filter((s) => s.rebuilt).length).toBe(1);
    expectValid(r, touches, fps);
    // Dragged absurdly high or below the chord: clamped, still a flight.
    for (const midY of [40, -5]) expectValid(refitEditedPath({ ...base, arcs: { [from]: midY } }), touches, fps);
  });

  it("resetting a flight gives back the automatic one", () => {
    const k = 3, f = touches[k].frame;
    const edits = { pins: { [f]: [1, 2, 3] as V3, [touches[0].frame]: [0, 2, 12] as V3 }, arcs: { [f]: 6, [touches[1].frame]: 5 } };
    const reset = resetSegmentEdits(edits, f, touches[k + 1].frame, 4);
    expect(reset.pins).toEqual({ [touches[0].frame]: [0, 2, 12] });
    expect(reset.arcs).toEqual({ [touches[1].frame]: 5 });
  });

  it("reports a flight no ball could make", () => {
    const k = 3;
    const far = touches.map((t, i) => (i === k ? { ...t, position: [30, 2, 45] as V3, pinned: true } : t));
    const r = refitEditedPath({ ...base, touches: far });
    expect(r.segments.some((s) => s.warning)).toBe(true);
    expectValid(r, far, fps);
  });
});

describe("editor re-fit latency", () => {
  it.skipIf(!long)("re-fits a long rally in well under 200 ms per edit", async () => {
    await new Promise((r) => setImmediate(r));
    const { base, fps } = solved(long!.fixture, long!.truth);
    const touches = base.touches;
    expect(touches.length).toBeGreaterThan(10);
    let seed = 7;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
    const times: number[] = [];
    for (let i = 0; i < 60; i++) {
      const k = 1 + Math.floor(rnd() * (touches.length - 2));
      const edited = touches.map((t, j) => (j === k
        ? { ...t, position: [t.position[0] + rnd() * 2 - 1, Math.max(0.3, t.position[1] + rnd() - 0.5), t.position[2] + rnd() * 2 - 1] as V3, pinned: true }
        : t));
      const arcs = i % 3 === 0 ? { [touches[Math.floor(rnd() * (touches.length - 1))].frame]: 2 + rnd() * 4 } : undefined;
      const t0 = performance.now();
      const r = refitEditedPath({ ...base, touches: edited, arcs });
      times.push(performance.now() - t0);
      expectValid(r, edited, fps);
    }
    times.sort((a, b) => a - b);
    console.log(`re-fit of a ${base.positions.size}-frame, ${touches.length}-touch rally: median ${times[30].toFixed(2)} ms, worst ${times[59].toFixed(2)} ms`);
    expect(times[59]).toBeLessThan(200);
  });
});

describe("undo and redo", () => {
  it("steps back and forward through edits", () => {
    let h = historyOf(0);
    h = historyPush(historyPush(historyPush(h, 1), 2), 3);
    expect(h.present).toBe(3);
    h = historyUndo(historyUndo(h));
    expect(h.present).toBe(1);
    h = historyRedo(h);
    expect(h.present).toBe(2);
    // A new edit after an undo drops what was undone.
    h = historyPush(h, 9);
    expect(historyRedo(h).present).toBe(9);
    expect(historyUndo(historyUndo(historyUndo(historyUndo(h)))).present).toBe(0);
  });
});
