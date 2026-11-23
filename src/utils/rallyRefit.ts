/**
 * The path as edited by hand, at once.
 *
 * The solver's result and the edits (touches moved, added, removed or retyped,
 * arcs made higher or flatter) go in; the path that the 3D view shows comes
 * out. Only the flights an edit touches are rebuilt, each as one ballistic
 * arc between its two touches: straight across the floor, falling at a
 * constant rate that is fitted to the flight's own detections within what its
 * touch type allows (or set by the arc handle). Flights share their touch
 * points, so the path stays continuous. Everything else is the solver's, frame
 * for frame.
 *
 * It is a pure function and takes a millisecond or so, which is what lets a
 * drag show its result immediately. The full solve still runs in the
 * background with the same edits; when it returns a path that already passes
 * through the edited touches, nothing here needs rebuilding and its result is
 * shown as it is.
 */
import type { BallReconstructionMethod } from "../hooks/useAnalysis";
import { DEFAULT_RALLY_PARAMS, type BallTouch } from "./ballReconstruction";

export type V3 = [number, number, number];

const G = 9.81;
/** Ball centre never below this (m): its radius. */
const MIN_Y = 0.105;
/** Bounds on a flight's downward acceleration (multiples of g) whatever its type or the handle say. */
export const ARC_MIN_G = 0.3;
export const ARC_MAX_G = 2.5;
/** A touch this far (m) from the solver's path at its frame has been moved. */
const MOVED_M = 0.05;
/** Fastest a ball flies (m/s); faster flights are reported, not refused. */
export const MAX_SPEED_MPS = 40;
/**
 * Longest a ball is in the air between two touches (s). Touches further apart
 * are not one flight (the ball went dead, or they are in different rallies):
 * nothing is drawn between them, which an arc that long would put tens of
 * metres up.
 */
export const MAX_FLIGHT_S = 4;

/** What the editor needs of a touch. */
export interface RefitTouch extends BallTouch {
  /** Set or placed by hand (see rallyEdits). */
  edited?: boolean;
  pinned?: boolean;
}

/** A detection of the ball on a frame, for fitting a rebuilt flight: image position in NDC. */
export type RefitObservations = Map<number, [number, number]>;

export interface RefitInput {
  touches: RefitTouch[];
  /** The solver's path and how each frame was found. */
  positions: Map<number, V3>;
  methods?: Map<number, BallReconstructionMethod>;
  fps: number;
  /** Arc handle: frame of the touch a flight starts at -> height (m) of the ball midway through it. */
  arcs?: Record<number, number>;
  /** Frames where a touch was removed by hand: a flight across one is rebuilt as a single arc. */
  removedAt?: number[];
  /** Detections and the camera, to fit rebuilt flights to what the video shows. Without them a flight falls at g. */
  observations?: RefitObservations;
  project?: (p: V3) => [number, number];
}

/** One flight between two touches, as shown. */
export interface RefitSegment {
  /** Frames of the touches it runs between. */
  from: number;
  to: number;
  /** Downward acceleration of the arc (m/s²), when it is one. */
  g: number | null;
  /** Where the arc handle sits: the ball midway through the flight. */
  mid: V3;
  /** Rebuilt here rather than taken from the solver. */
  rebuilt: boolean;
  /** Its arc was set with the handle. */
  manual: boolean;
  /** Shaped by the user: its arc set with the handle, or a touch at either end placed by hand. */
  byHand: boolean;
  /** Launch speed (m/s) of a rebuilt arc. */
  speedMps: number | null;
  /** Why a rebuilt arc is not a plausible flight, if it is not. */
  warning: string | null;
}

export interface RefitResult {
  positions: Map<number, V3>;
  methods: Map<number, BallReconstructionMethod>;
  segments: RefitSegment[];
  /** Frames whose position was rebuilt. */
  rebuiltFrames: number;
}

/** Range of downward acceleration (multiples of g) of the flight a touch of this type starts. */
export function gravityRange(label: string): [number, number] {
  const rules = DEFAULT_RALLY_PARAMS.rules as Record<string, { minG: number; maxG: number } | undefined>;
  // A "block/dig" may be either: the wider range.
  const kinds = label.split("/").map((k) => rules[k]).filter((r): r is { minG: number; maxG: number } => !!r);
  if (!kinds.length) return [DEFAULT_RALLY_PARAMS.accelMinG, DEFAULT_RALLY_PARAMS.accelMaxG];
  return [Math.min(...kinds.map((r) => r.minG)), Math.max(...kinds.map((r) => r.maxG))];
}

/** Position on the arc from a (at frame fa) to b (at frame fb) falling at g, at frame f. */
export function arcAt(a: V3, b: V3, fa: number, fb: number, g: number, fps: number, f: number): V3 {
  const T = (fb - fa) / fps, s = (f - fa) / fps, w = T > 0 ? s / T : 0;
  return [a[0] + (b[0] - a[0]) * w, Math.max(MIN_Y, a[1] + (b[1] - a[1]) * w + 0.5 * g * s * (T - s)), a[2] + (b[2] - a[2]) * w];
}

/** The acceleration that puts the middle of the arc at height midY. */
export function gravityForMidHeight(a: V3, b: V3, fa: number, fb: number, fps: number, midY: number): number {
  const T = (fb - fa) / fps;
  if (T <= 0) return G;
  return (8 * (midY - (a[1] + b[1]) / 2)) / (T * T);
}

const clampG = (g: number, lo = ARC_MIN_G, hi = ARC_MAX_G) => Math.min(hi * G, Math.max(lo * G, g));

/** Acceleration of the arc between two touches that best follows the detections between them. */
function fitGravity(
  a: V3, b: V3, fa: number, fb: number, fps: number, range: [number, number],
  observations: RefitObservations | undefined, project: ((p: V3) => [number, number]) | undefined,
): number {
  const lo = Math.max(ARC_MIN_G, range[0]) * G, hi = Math.min(ARC_MAX_G, range[1]) * G;
  if (!observations || !project) return Math.min(hi, Math.max(lo, G));
  const obs: [number, [number, number]][] = [];
  for (let f = fa + 1; f < fb; f++) {
    const o = observations.get(f);
    if (o) obs.push([f, o]);
  }
  if (obs.length < 2) return Math.min(hi, Math.max(lo, G));
  // Median of absolute errors: one stray detection must not bend the arc.
  const cost = (g: number) => {
    const e = obs.map(([f, o]) => {
      const [u, v] = project(arcAt(a, b, fa, fb, g, fps, f));
      return Math.hypot(u - o[0], v - o[1]);
    }).sort((x, y) => x - y);
    return e[e.length >> 1] + 0.1 * (e.reduce((s, x) => s + x, 0) / e.length);
  };
  // Golden-section search: the error has one minimum in g.
  const phi = (Math.sqrt(5) - 1) / 2;
  let x0 = lo, x1 = hi;
  let c = x1 - phi * (x1 - x0), d = x0 + phi * (x1 - x0);
  let fc = cost(c), fd = cost(d);
  for (let i = 0; i < 24; i++) {
    if (fc < fd) { x1 = d; d = c; fd = fc; c = x1 - phi * (x1 - x0); fc = cost(c); }
    else { x0 = c; c = d; fc = fd; d = x0 + phi * (x1 - x0); fd = cost(d); }
  }
  return (x0 + x1) / 2;
}

/**
 * The edited path. Touches must be in frame order; a flight is the stretch
 * between two consecutive touches.
 */
export function refitEditedPath(input: RefitInput): RefitResult {
  const { touches, fps } = input;
  const positions = new Map(input.positions);
  const methods = new Map(input.methods ?? []);
  const segments: RefitSegment[] = [];
  let rebuiltFrames = 0;
  // Only a touch placed by hand can be off the solver's path. Its frame is
  // the nearest to the contact, which falls between two frames, so it is
  // compared with the path around that frame, not with one sample of it.
  const moved = (t: RefitTouch) => {
    if (!t.pinned) return false;
    let best = Infinity;
    for (const f of [t.frame - 1, t.frame]) {
      const a = input.positions.get(f), b = input.positions.get(f + 1);
      if (a && b) best = Math.min(best, distToSegment(t.position, a, b));
    }
    const at = input.positions.get(t.frame);
    if (at) best = Math.min(best, Math.hypot(at[0] - t.position[0], at[1] - t.position[1], at[2] - t.position[2]));
    return best > MOVED_M;
  };

  for (let k = 0; k + 1 < touches.length; k++) {
    const A = touches[k], B = touches[k + 1];
    const fa = A.frame, fb = B.frame;
    if (fb <= fa || (fb - fa) / fps > MAX_FLIGHT_S) continue;
    const a = A.position, b = B.position;
    const arcMid = input.arcs?.[fa];
    const manual = arcMid != null;
    const spansRemoved = (input.removedAt ?? []).some((f) => f > fa && f < fb);
    let holes = false;
    for (let f = fa; f <= fb && !holes; f++) holes = !input.positions.has(f);
    // A flight across a removed touch still has the solver's kink in it
    // until the background solve returns: only then is it one arc.
    const kinked = spansRemoved && !isOneArc(input.positions, fa, fb, fps);
    const byHand = manual || moved(A) || moved(B);
    const rebuild = byHand || holes || kinked;

    if (!rebuild) {
      const mf = Math.round((fa + fb) / 2);
      segments.push({ from: fa, to: fb, g: null, mid: input.positions.get(mf) ?? arcAt(a, b, fa, fb, G, fps, mf), rebuilt: false, manual: false, byHand: false, speedMps: null, warning: null });
      continue;
    }
    const g = manual
      ? clampG(gravityForMidHeight(a, b, fa, fb, fps, arcMid))
      : fitGravity(a, b, fa, fb, fps, gravityRange(A.label), input.observations, input.project);
    for (let f = fa; f <= fb; f++) {
      positions.set(f, f === fa ? a : f === fb ? b : arcAt(a, b, fa, fb, g, fps, f));
      // Placed by hand or by rule, not measured: drawn as an estimate.
      methods.set(f, "estimated");
      rebuiltFrames++;
    }
    const T = (fb - fa) / fps;
    const v0: V3 = [(b[0] - a[0]) / T, (b[1] - a[1]) / T + 0.5 * g * T, (b[2] - a[2]) / T];
    const v1: V3 = [v0[0], v0[1] - g * T, v0[2]];
    const speed = Math.max(Math.hypot(...v0), Math.hypot(...v1));
    let warning: string | null = null;
    if (speed > MAX_SPEED_MPS) warning = `too fast for a ball (${speed.toFixed(0)} m/s): the touches are too far apart for the time between them`;
    else if (b[2] * a[2] < 0) {
      // Crossing the net: over the tape, between the antennae.
      const w = a[2] / (a[2] - b[2]);
      const f = fa + w * (fb - fa);
      const at = arcAt(a, b, fa, fb, g, fps, f);
      if (Math.abs(at[0]) <= 4.5 && at[1] < 2.43) warning = `passes through the net (${at[1].toFixed(1)} m high at it)`;
    }
    segments.push({
      from: fa, to: fb, g, mid: arcAt(a, b, fa, fb, g, fps, (fa + fb) / 2), rebuilt: true, manual, byHand, speedMps: Math.hypot(...v0), warning,
    });
  }
  return { positions, methods, segments, rebuiltFrames };
}

/** Distance from p to the stretch a-b. */
function distToSegment(p: V3, a: V3, b: V3): number {
  const d: V3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const w = dd > 0 ? Math.min(1, Math.max(0, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / dd)) : 0;
  return Math.hypot(p[0] - a[0] - d[0] * w, p[1] - a[1] - d[1] * w, p[2] - a[2] - d[2] * w);
}

/**
 * The hand edits that shape one flight, cleared: the 3D placements of its
 * two touches and its arc. What is left is the automatic result for it.
 */
export function resetSegmentEdits<E extends { pins: Record<number, V3>; arcs?: Record<number, number> }>(
  edits: E, from: number, to: number, tolFrames: number,
): E {
  const pins = { ...edits.pins };
  for (const f of Object.keys(pins).map(Number)) {
    if (Math.abs(f - from) <= tolFrames || Math.abs(f - to) <= tolFrames) delete pins[f];
  }
  const arcs = { ...(edits.arcs ?? {}) };
  for (const f of Object.keys(arcs).map(Number)) if (Math.abs(f - from) <= tolFrames) delete arcs[f];
  return { ...edits, pins, arcs };
}

/** Does the path between two frames lie on a single arc (to 0.25 m)? */
function isOneArc(positions: Map<number, V3>, fa: number, fb: number, fps: number): boolean {
  const a = positions.get(fa), b = positions.get(fb);
  if (!a || !b) return false;
  // Least-squares g for fixed ends, from the heights.
  let num = 0, den = 0;
  const T = (fb - fa) / fps;
  for (let f = fa + 1; f < fb; f++) {
    const p = positions.get(f);
    if (!p) return false;
    const s = (f - fa) / fps, w = s / T, k = 0.5 * s * (T - s);
    num += k * (p[1] - (a[1] + (b[1] - a[1]) * w));
    den += k * k;
  }
  const g = den > 0 ? num / den : G;
  for (let f = fa + 1; f < fb; f++) {
    const p = positions.get(f)!, q = arcAt(a, b, fa, fb, g, fps, f);
    if (Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) > 0.25) return false;
  }
  return true;
}

// ─── Undo / redo ─────────────────────────────────────────────────────────────

/** A value with the states before and after it, for undo and redo. */
export interface History<T> {
  past: T[];
  present: T;
  future: T[];
}

const HISTORY_MAX = 200;

export const historyOf = <T>(present: T): History<T> => ({ past: [], present, future: [] });

/** A new state after the present one; anything undone before is dropped. */
export function historyPush<T>(h: History<T>, next: T): History<T> {
  if (next === h.present) return h;
  return { past: [...h.past, h.present].slice(-HISTORY_MAX), present: next, future: [] };
}

export function historyUndo<T>(h: History<T>): History<T> {
  if (!h.past.length) return h;
  return { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] };
}

export function historyRedo<T>(h: History<T>): History<T> {
  if (!h.future.length) return h;
  return { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) };
}
