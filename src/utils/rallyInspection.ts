/**
 * Likely problems in a solved rally, and a 0-100 score, for the Inspect &
 * Edit view. The app has no ground truth, so both come from what the solver
 * itself can tell: where the ball was not seen, which flights did not fit,
 * which detections are off the path, which touches the rally rules could not
 * identify. Each problem says which hand correction fixes it. Same stages
 * and thresholds as the eval's rally inspector (src/eval/inspectRun.ts).
 */
import type { FrameData } from "../hooks/useAnalysis";
import type { PhysicsReprojectionDebug, TouchSplitDecision } from "./ballReconstruction";
import { DEFAULT_RALLY_PARAMS } from "./ballReconstruction";
import { touchMatchFrames, type EditedTouch, type RallyEdits } from "./rallyEdits";

export type Severity = "high" | "medium" | "low";

/** What the user can do about a problem. */
export type Fix =
  | { kind: "mark-ball"; label: string }
  | { kind: "remove-ball"; label: string; frames: number[] }
  | { kind: "add-touch"; label: string; frame: number }
  | { kind: "set-touch"; label: string; frame: number }
  | { kind: "remove-touch"; label: string; frame: number }
  | { kind: "place-3d"; label: string; frame: number }
  | { kind: "align-court"; label: string };

export interface Problem {
  /** Stable across re-solves, so "not a problem" sticks. */
  key: string;
  severity: Severity;
  frame: number;
  frameEnd?: number;
  title: string;
  detail: string;
  fixes: Fix[];
}

export interface InspectionInput {
  /** With the ball edits applied. */
  frames: FrameData[];
  fps: number;
  positions: Map<number, [number, number, number]>;
  physicsDebug?: PhysicsReprojectionDebug[];
  touches: EditedTouch[];
  touchSplits?: TouchSplitDecision[];
  edits: RallyEdits;
  /** Worst court keypoint reprojection error (px), when known. */
  courtErrPx: number | null;
  hasCalibration: boolean;
}

export interface ScorePart {
  name: string;
  /** 0-1, or null when there is nothing to judge. */
  value: number | null;
  weight: number;
}

export interface Inspection {
  problems: Problem[];
  /** Problems not dismissed. */
  open: Problem[];
  score: number;
  parts: ScorePart[];
  /** First and last frame the ball is seen on. */
  rally: [number, number];
  stats: { seen: number; rallyFrames: number; solved: number; reprojMedianPx: number | null; touches: number };
}

const SEV_RANK: Record<Severity, number> = { high: 0, medium: 1, low: 2 };
/** What an open problem costs the score. */
const SEV_COST: Record<Severity, number> = { high: 1, medium: 0.4, low: 0.1 };
/** A detection gap longer than this (s) is the ball out of play, not lost. */
const MAX_GAP_REPORTED_S = 1.5;
/** A flight faster than this (m/s) is implausible: a hard jump serve is ~30. */
const MAX_PLAUSIBLE_SPEED = 34;
/** A touch this far (m) outside the court lines is implausible. */
const OUT_OF_PLAY_M = 4;
/** Slack (m) on a touch type's usual contact height before it is flagged. */
const HEIGHT_SLACK_M = 0.4;
/** Touches closer than this (s) are one touch counted twice, unless a block or dig. */
const MIN_TOUCH_GAP_S = 0.3;

const median = (a: number[]) => {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Runs of consecutive frames where `pred` holds, as [first, last]. */
function runs(frames: number[], pred: (f: number) => boolean): [number, number][] {
  const out: [number, number][] = [];
  let start = -1, prev = -1;
  for (const f of frames) {
    if (pred(f) && (start < 0 || f === prev + 1)) { if (start < 0) start = f; }
    else if (start >= 0) { out.push([start, prev]); start = pred(f) ? f : -1; }
    prev = f;
  }
  if (start >= 0) out.push([start, prev]);
  return out;
}

const secs = (n: number, fps: number) => `${(n / fps).toFixed(2)} s`;
const range = (a: number, b: number) => (a === b ? `frame ${a}` : `frames ${a}–${b}`);

export function inspectRally(inp: InspectionInput): Inspection {
  const { frames, fps, positions, edits } = inp;
  const pd = new Map((inp.physicsDebug ?? []).map((d) => [d.frame, d]));
  const all = frames.map((f) => f.frame);
  const seenAt = (f: number) => { const x = frames[f]; return !!x?.ball_img_pos && !x.ball_interpolated; };
  const seen = all.filter(seenAt);
  const rally: [number, number] = [seen[0] ?? 0, seen[seen.length - 1] ?? 0];
  const inRally = all.filter((f) => f >= rally[0] && f <= rally[1]);
  const edited = (f: number) => f in edits.ball;
  const tol = touchMatchFrames(fps);
  const touchNear = (f: number, within = tol) => inp.touches.find((t) => Math.abs(t.frame - f) <= within);
  const problems: Problem[] = [];

  // Ball lost: gaps in detection inside the rally.
  const minGap = Math.max(3, Math.round(0.2 * fps));
  for (const [a, b] of runs(inRally, (f) => !seenAt(f))) {
    const n = b - a + 1;
    if (n < minGap || n > MAX_GAP_REPORTED_S * fps) continue;
    const hides = touchNear((a + b) / 2, n / 2 + 1);
    problems.push({
      key: `gap:${a}`, severity: hides ? "high" : n >= 0.5 * fps ? "medium" : "low", frame: a, frameEnd: b,
      title: `Ball lost for ${n} frames (${secs(n, fps)})`,
      detail: hides
        ? `The ${hides.label} at frame ${hides.frame} sits in this gap, so its timing and place are guesses. Click the ball on a few frames here.`
        : "Click the ball on a few of these frames (every 2nd or 3rd is enough) to fill the path in.",
      fixes: [{ kind: "mark-ball", label: "Mark the ball" }],
    });
  }

  // Moving camera: detections dropped because the camera motion was not trusted.
  if (frames.some((f) => f.ref_from_frame)) {
    for (const [a, b] of runs(inRally, (f) => seenAt(f) && !frames[f].ball_ref_pos)) {
      if (b - a + 1 < 3) continue;
      problems.push({
        key: `camera:${a}`, severity: b - a + 1 >= 8 ? "medium" : "low", frame: a, frameEnd: b,
        title: `Ball seen but not used on ${b - a + 1} frames`,
        detail: "The camera's pan/zoom could not be measured here, so these detections cannot be placed on the court.",
        fixes: [],
      });
    }
  }

  // Court calibration.
  if (!inp.hasCalibration) {
    problems.push({ key: "court:none", severity: "high", frame: 0, title: "No court calibration",
      detail: "The court was not found, so nothing can be placed in 3D. Align the court by hand.",
      fixes: [{ kind: "align-court", label: "Align court" }] });
  } else if (inp.courtErrPx != null && inp.courtErrPx > 6) {
    problems.push({ key: "court:error", severity: inp.courtErrPx > 15 ? "high" : "medium", frame: 0,
      title: `Court lines off by up to ${inp.courtErrPx.toFixed(0)} px`,
      detail: "A misplaced court skews every 3D position. Check the court lines on the video and re-align them.",
      fixes: [{ kind: "align-court", label: "Align court" }] });
  }

  // Flights that could not be fitted.
  const arcs = new Map<number, { first: number; last: number; n: number; ok: boolean | null; why: string | null; speed: number | null }>();
  for (const d of inp.physicsDebug ?? []) {
    if (d.arcId == null) continue;
    const a = arcs.get(d.arcId) ?? { first: d.frame, last: d.frame, n: 0, ok: d.arcAccepted, why: d.arcRejectReason, speed: d.speedMps };
    a.first = Math.min(a.first, d.frame); a.last = Math.max(a.last, d.frame); a.n++;
    arcs.set(d.arcId, a);
  }
  for (const a of arcs.values()) {
    if (a.ok === false && a.n >= 4) {
      problems.push({
        key: `arc:${a.first}`, severity: a.n >= 8 ? "medium" : "low", frame: a.first, frameEnd: a.last,
        title: `Flight at ${range(a.first, a.last)} could not be fitted`,
        detail: a.why === "too-few-observations"
          ? "Too few detections to fit a flight. Mark the ball on more frames."
          : "No ball flight matches these detections. Usually a wrong detection (remove it), or a touch the solver missed (add it).",
        fixes: [{ kind: "mark-ball", label: "Fix the ball" }, { kind: "add-touch", label: "Add a touch", frame: Math.round((a.first + a.last) / 2) }],
      });
    } else if (a.ok && a.speed != null && a.speed > MAX_PLAUSIBLE_SPEED) {
      problems.push({
        key: `fast:${a.first}`, severity: "medium", frame: a.first, frameEnd: a.last,
        title: `Ball flies at ${a.speed.toFixed(0)} m/s at ${range(a.first, a.last)}`,
        detail: "Faster than any real hit, so the depth is wrong. Often a touch is missing in this flight.",
        fixes: [{ kind: "add-touch", label: "Add a touch", frame: Math.round((a.first + a.last) / 2) }],
      });
    }
  }

  // Detections off the fitted path.
  for (const [a, b] of runs(all, (f) => !!pd.get(f)?.isOutlier && !edited(f))) {
    const errs = all.filter((f) => f >= a && f <= b).map((f) => pd.get(f)?.reprojectionErrorPx ?? 0);
    const outFrames = all.filter((f) => f >= a && f <= b && seenAt(f));
    problems.push({
      key: `outlier:${a}`, severity: b - a + 1 >= 3 ? "medium" : "low", frame: a, frameEnd: b,
      title: `Detection off the path at ${range(a, b)} (up to ${Math.max(...errs).toFixed(0)} px)`,
      detail: "Either the detector picked something else (another ball, a head), or a player touched the ball here.",
      fixes: [
        { kind: "remove-ball", label: outFrames.length > 1 ? `Not the ball (${outFrames.length})` : "Not the ball", frames: outFrames },
        { kind: "mark-ball", label: "Move it" },
        { kind: "add-touch", label: "It's a touch", frame: a },
      ],
    });
  }

  // Ball seen but given no 3D position.
  for (const [a, b] of runs(inRally, (f) => seenAt(f) && !positions.has(f))) {
    if (b - a + 1 < 3 || problems.some((p) => p.key.startsWith("arc:") && p.frame <= b && (p.frameEnd ?? p.frame) >= a)) continue;
    problems.push({
      key: `unsolved:${a}`, severity: "medium", frame: a, frameEnd: b,
      title: `No 3D path at ${range(a, b)}`,
      detail: "The ball is seen here but the flight was dropped. Check the detections and touches around it.",
      fixes: [{ kind: "mark-ball", label: "Check the ball" }],
    });
  }

  // Touches.
  inp.touches.forEach((t, i) => {
    if (t.edited) return;
    if (t.label === "touch") {
      problems.push({
        key: `touchtype:${t.frame}`, severity: "medium", frame: t.frame,
        title: `Touch at frame ${t.frame} has no type`,
        detail: "The rally rules could not tell what this touch is. Set its type, or remove it if nobody played the ball.",
        fixes: [{ kind: "set-touch", label: "Set type", frame: t.frame }, { kind: "remove-touch", label: "Not a touch", frame: t.frame }],
      });
    }
    // Where in 3D: the depth along the line of sight is the solver's weak
    // direction, and it shows as touches at heights or places nobody plays.
    const [x, y, z] = t.position;
    const rule = (DEFAULT_RALLY_PARAMS.rules as Record<string, { minY: number; maxY: number }>)[t.label.split("/")[0]];
    const place3d: Fix = { kind: "place-3d", label: "Place in 3D", frame: t.frame };
    if (Math.abs(z) > 9 + OUT_OF_PLAY_M || Math.abs(x) > 4.5 + OUT_OF_PLAY_M) {
      problems.push({
        key: `where:${t.frame}`, severity: "medium", frame: t.frame,
        title: `The ${t.label} at frame ${t.frame} is ${Math.abs(z) > 9 + OUT_OF_PLAY_M ? `${(Math.abs(z) - 9).toFixed(1)} m behind the end line` : `${(Math.abs(x) - 4.5).toFixed(1)} m outside the sideline`}`,
        detail: "Players rarely play the ball that far out, so its depth is probably wrong. Drag it to where it was played.",
        fixes: [place3d],
      });
    } else if (rule && t.label !== "toss" && (y < rule.minY - HEIGHT_SLACK_M || y > rule.maxY + HEIGHT_SLACK_M)) {
      // Not the toss: its point is wherever the clip first sees the ball, already on its way up.
      problems.push({
        key: `height:${t.frame}`, severity: "medium", frame: t.frame,
        title: y < 0 ? `The ${t.label} at frame ${t.frame} is below the floor` : `The ${t.label} at frame ${t.frame} is ${y.toFixed(1)} m high`,
        detail: `A ${t.label} is usually played at ${rule.minY}–${rule.maxY} m. Drag it to the right height on its line of sight.`,
        fixes: [place3d],
      });
    }
    const prev = inp.touches[i - 1];
    if (prev && (t.frame - prev.frame) / fps < MIN_TOUCH_GAP_S && !/block|dig/.test(t.label)) {
      problems.push({
        key: `double:${t.frame}`, severity: "low", frame: prev.frame, frameEnd: t.frame,
        title: `Two touches ${t.frame - prev.frame} frames apart`,
        detail: `A ${prev.label} and a ${t.label} this close are usually one touch counted twice.`,
        fixes: [{ kind: "remove-touch", label: `Remove the ${t.label}`, frame: t.frame }],
      });
    }
  });
  // Touches the user marked that have no 3D path to sit on.
  for (const [key, kind] of Object.entries(edits.touches)) {
    const f = Number(key);
    if (kind === "none" || inp.touches.some((t) => t.edited && t.frame === f)) continue;
    problems.push({
      key: `unplaced:${f}`, severity: "medium", frame: f,
      title: `Your ${kind} at frame ${f} has no 3D path around it`,
      detail: "The ball is not solved near this touch, so it cannot be shown on the court. Mark the ball on the frames around it.",
      fixes: [{ kind: "mark-ball", label: "Mark the ball" }],
    });
  }
  // Possible missed touches: a velocity jump near a player the solver let go.
  for (const s of inp.touchSplits ?? []) {
    const f = Math.round((s.frameBefore + s.frameAfter) / 2);
    if (s.kept || s.candidates === 0 || s.dvMps < 0.6 * DEFAULT_RALLY_PARAMS.touchMinDvMps) continue;
    if (touchNear(f) || f in edits.touches) continue;
    problems.push({
      key: `missed:${f}`, severity: "low", frame: f,
      title: `Possible touch at frame ${f}`,
      detail: `The ball changes direction near a player here (${s.dvMps.toFixed(1)} m/s), but not enough to count as a touch.`,
      fixes: [{ kind: "add-touch", label: "Add touch", frame: f }],
    });
  }

  // Off-path detections inside a flight that did not fit are why it did not:
  // one problem, offering to remove them first.
  const folded = new Set<Problem>();
  for (const arc of problems.filter((x) => x.key.startsWith("arc:"))) {
    const inside = problems.filter((x) => x.key.startsWith("outlier:") && x.frame >= arc.frame && (x.frameEnd ?? x.frame) <= (arc.frameEnd ?? arc.frame));
    const frames = inside.flatMap((x) => x.fixes.flatMap((fix) => (fix.kind === "remove-ball" ? fix.frames : [])));
    if (!frames.length) continue;
    inside.forEach((x) => folded.add(x));
    arc.detail = `${frames.length} detection${frames.length > 1 ? "s are" : " is"} far off the path here: probably not the ball. ${arc.detail}`;
    arc.fixes = [{ kind: "remove-ball", label: frames.length > 1 ? `Not the ball (${frames.length})` : "Not the ball", frames }, ...arc.fixes];
    if (arc.severity === "low") arc.severity = "medium";
  }
  const listed = problems.filter((x) => !folded.has(x))
    .sort((a, b) => SEV_RANK[a.severity] - SEV_RANK[b.severity] || a.frame - b.frame);
  const dismissed = new Set(edits.dismissed);
  const open = listed.filter((p) => !dismissed.has(p.key));

  // Score: how complete and consistent the rally is, and what is left to fix.
  const reproj = (inp.physicsDebug ?? []).flatMap((d) => (d.usedInFit && d.reprojectionErrorPx != null ? [d.reprojectionErrorPx] : []));
  const reprojMedianPx = median(reproj);
  const seenInRally = inRally.filter(seenAt);
  const solvedSeen = seenInRally.filter((f) => positions.has(f));
  const typed = inp.touches.filter((t) => t.label !== "touch");
  const cost = open.reduce((s, p) => s + SEV_COST[p.severity], 0);
  const parts: ScorePart[] = [
    { name: "Ball seen", weight: 0.2, value: inRally.length ? seenInRally.length / inRally.length : null },
    { name: "3D path", weight: 0.2, value: seenInRally.length ? solvedSeen.length / seenInRally.length : null },
    { name: "Path fit", weight: 0.15, value: reprojMedianPx == null ? null : Math.exp(-reprojMedianPx / 10) },
    { name: "Touches known", weight: 0.15, value: inp.touches.length ? typed.length / inp.touches.length : null },
    { name: "Open problems", weight: 0.3, value: Math.exp(-cost / 4) },
  ];
  const have = parts.filter((p) => p.value != null);
  const wsum = have.reduce((s, p) => s + p.weight, 0);
  const score = inp.hasCalibration && wsum > 0 ? (100 * have.reduce((s, p) => s + p.weight * p.value!, 0)) / wsum : 0;

  return {
    problems: listed, open, score, parts, rally,
    stats: { seen: seenInRally.length, rallyFrames: inRally.length, solved: positions.size, reprojMedianPx, touches: inp.touches.length },
  };
}
