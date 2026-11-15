/**
 * The solver's account of a path, in words and as coloured stretches, for the
 * developer view in Inspect & Edit ("How this path was made").
 *
 * The solver cuts the ball's track into flights at touches, fits all flights
 * of a rally together as arcs that meet at the touches, and then judges each
 * flight: does the arc pass through its detections? A flight that does is
 * drawn as fitted. One that does not is "refused": its arc is still used for
 * depth, but each frame is moved onto the camera's line of sight through the
 * detection, which is why such a stretch wobbles with the detections. Holes
 * with no track at all are bridged by a plain arc or followed on under
 * gravity. Everything but a fitted flight is an estimate and drawn dashed in
 * the 3D view.
 */
import type { SolverTrace, SolverTraceFlight, SolverTraceKnot, TouchSplitDecision } from "./ballReconstruction";

export type SpanStatus = "fitted" | "refused" | "bridge" | "follow" | "rebuilt";

/** A stretch of frames drawn one way, and why. */
export interface PathSpan {
  from: number;
  to: number;
  status: SpanStatus;
  /** One line: what this stretch is. */
  title: string;
  /** Why it is drawn the way it is. */
  why: string;
  /** For a flight: the knots it runs between and its numbers. */
  flight?: SolverTraceFlight;
  fromKnot?: SolverTraceKnot;
  toKnot?: SolverTraceKnot;
}

export const SPAN_COLORS: Record<SpanStatus, string> = {
  fitted: "#69f0ae",
  refused: "#ffa726",
  bridge: "#9aa4b2",
  follow: "#9aa4b2",
  rebuilt: "#00e5ff",
};

export const SPAN_NAMES: Record<SpanStatus, string> = {
  fitted: "fitted",
  refused: "refused, drawn as an estimate",
  bridge: "bridged hole",
  follow: "followed on",
  rebuilt: "rebuilt from your edit",
};

/** The solver's rejection code in plain words. */
export function explainReject(reason: string | null, limits: SolverTrace["limits"]): string {
  if (!reason) return "";
  const num = /\(([^)]*)\)/.exec(reason)?.[1] ?? "";
  if (reason.startsWith("median-err-too-high")) {
    return `the arc misses its detections by ${num.replace("px", " px")} at the median (limit ${limits.segMaxMedianErrPx} px). ` +
      "Usually one of its two touches is in the wrong place or at the wrong time, so no arc between them passes through the ball.";
  }
  if (reason.startsWith("too-few-inliers")) {
    return `only ${num} of its detections lie within ${limits.rallyOutlierPx} px of the arc. The arc and the detections disagree almost everywhere: ` +
      "a touch at one end is misplaced, or the detections are not the ball.";
  }
  if (reason.startsWith("too-many-outliers")) {
    return `only ${num} lie within ${limits.rallyOutlierPx} px of the arc (at least ${Math.round(limits.segMinInlierFrac * 100)}% needed). Part of the flight fits, part does not: likely a touch inside it was missed.`;
  }
  if (reason.startsWith("speed-too-high")) return `the arc would need ${num}: faster than a ball flies. Its two touches are too far apart for the time between them.`;
  if (reason.startsWith("y-below-ground")) return `the arc dips below the floor (${num}).`;
  if (reason.startsWith("y-too-high")) return `the arc rises above 10 m (${num}).`;
  if (reason.startsWith("x-out-of-bounds") || reason.startsWith("z-out-of-bounds")) return `the arc leaves the hall (${num} from the court's centre).`;
  if (reason.startsWith("too-few-observations")) return "too few detections to fit anything.";
  return reason;
}

const knotName = (k: SolverTraceKnot | undefined) => (k ? (k.reported ? k.label : k.label === "end" ? "end of track" : `${k.label} (not a touch)`) : "?");

/** The path as stretches: every flight of every rally, then the bridged and followed holes. */
export function traceSpans(trace: SolverTrace, rebuilt: { from: number; to: number }[] = []): PathSpan[] {
  const spans: PathSpan[] = [];
  for (const rally of trace.rallies) {
    rally.flights.forEach((fl, i) => {
      const a = rally.knots[i], b = rally.knots[i + 1];
      const numbers = `${fl.detections} detections, ${fl.inliers} kept` +
        (fl.medianErrPx != null ? `, median error ${fl.medianErrPx.toFixed(1)} px` : "") +
        `, launch ${fl.speedMps.toFixed(0)} m/s, falls at ${fl.gOverG.toFixed(2)} g`;
      spans.push({
        from: fl.fromFrame, to: fl.toFrame,
        status: fl.accepted ? "fitted" : "refused",
        title: `${knotName(a)} → ${knotName(b)}`,
        why: fl.accepted
          ? `Fitted: one arc through its detections (${numbers}).`
          : `Refused: ${explainReject(fl.rejectReason, trace.limits)} (${numbers}.) ` +
            "Drawn as an estimate: the refused arc gives the depth, and each frame is moved onto the camera's line of sight through its detection, so the line follows the detections and wobbles with them.",
        flight: fl, fromKnot: a, toKnot: b,
      });
    });
  }
  for (const f of trace.fills) {
    if (f.kind === "refused") continue; // already listed as its flight
    spans.push({
      from: f.fromFrame, to: f.toFrame, status: f.kind,
      title: f.kind === "bridge" ? "hole between two solved stretches" : "after the last solved flight",
      why: f.kind === "bridge"
        ? `No track was solved here (${f.detections} detections in ${f.toFrame - f.fromFrame + 1} frames). Joined by one arc under gravity between the last solved point before and the first after${f.detections ? ", moved onto the detections there are" : ""}.`
        : "The ball was lost. The last flight is carried on under gravity until it reaches the floor or 1.5 s pass.",
    });
  }
  for (const r of rebuilt) {
    spans.push({
      from: r.from, to: r.to, status: "rebuilt", title: "flight at one of your edits",
      why: "Rebuilt at once as one arc between its two touches (see the editor); the background solve will replace it.",
    });
  }
  return spans.sort((x, y) => x.from - y.from || x.to - y.to);
}

/** The stretch a frame is drawn by: a rebuilt one first, then the shortest that holds it. */
export function spanAt(spans: PathSpan[], frame: number): PathSpan | null {
  let best: PathSpan | null = null;
  const rank = (s: PathSpan) => (s.status === "rebuilt" ? 0 : 1);
  for (const s of spans) {
    if (frame < s.from || frame > s.to) continue;
    if (!best || rank(s) < rank(best) || (rank(s) === rank(best) && s.to - s.from < best.to - best.from)) best = s;
  }
  return best;
}

/** Why a candidate split was kept as a touch or dropped. */
export function explainSplit(s: TouchSplitDecision, limits: SolverTrace["limits"]): string {
  const how = s.kink != null
    ? `a kink in the track (score ${s.kink.toFixed(0)}, needs ${limits.touchKinkScore})`
    : `a gap in the detections (${s.frameAfter - s.frameBefore - 1} frames)`;
  const need = s.kink != null ? limits.touchMinDvKinkMps : limits.touchMinDvMps;
  const dv = Number.isFinite(s.dvMps) ? `velocity change ${s.dvMps.toFixed(1)} m/s (needs ${need})` : "too few samples to measure a velocity change";
  const players = s.candidates ? `${s.candidates} player box${s.candidates === 1 ? "" : "es"} in reach` : "no player in reach";
  let verdict: string;
  if (s.kept) verdict = Number.isFinite(s.dvMps) && s.dvMps < need ? "kept (next to another touch, or too clear to doubt)" : "kept as a possible touch";
  else if (!s.candidates && Number.isFinite(s.dvMps) && s.dvMps >= need) verdict = "dropped: nobody there to touch it";
  else verdict = "dropped: the ball did not change course enough";
  return `${how}; ${dv}; ${players} → ${verdict}`;
}
