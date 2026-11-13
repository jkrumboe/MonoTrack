/**
 * Hand corrections to a rally, made in the Inspect & Edit view and saved per
 * video, so a clip is corrected once rather than on every analysis.
 *
 * Ball edits change the solver's input (a detection moved, added or removed);
 * touch edits steer where the solver splits the track into flights, and are
 * applied again to its output, since the rally rules may still drop or
 * mislabel a touch the user marked. Pins place a touch in 3D: the depth a
 * single camera cannot see. The solver holds the touch there and refits the
 * flights either side, so moving one touch moves the path around it.
 */
import { getBackend } from "../platform/backend";
import type { Side } from "./matchTimeline";
import type { FrameData } from "../hooks/useAnalysis";
import type { BallReconstructionOptions, BallTouch } from "./ballReconstruction";

/**
 * Touch types that can be set by hand. The solver finds the first seven; a
 * free ball, a touch of the net and the ball landing are marked in the editor.
 */
export const TOUCH_KINDS = ["toss", "serve", "reception", "set", "attack", "block", "dig", "free ball", "net", "floor"] as const;
export type TouchKind = (typeof TOUCH_KINDS)[number];

/** Marker colour per touch type (a solver "block/dig" takes its first kind's). */
export const TOUCH_COLORS: Record<string, string> = {
  toss: "#7fb3e6", serve: "#3a88d8", reception: "#17a085", set: "#b38d0c", attack: "#e0443f",
  block: "#a860d2", dig: "#d67a14", "free ball": "#5fb0a6", net: "#e6d25a", floor: "#8a6b4f", touch: "#9a9a9a",
};
export const touchColor = (label: string) => TOUCH_COLORS[label] ?? TOUCH_COLORS[label.split("/")[0]] ?? TOUCH_COLORS.touch;

export interface RallyEdits {
  version: 1;
  /** Frame -> ball centre (normalised coords of that frame's own image), or null: no ball on it. */
  ball: Record<number, [number, number] | null>;
  /** Frame -> touch made there, or "none": the touch found there is not one. */
  touches: Record<number, TouchKind | "none">;
  /** Frame of a touch -> where the ball was at it (m, court coords: x across, y up, z along). */
  pins: Record<number, [number, number, number]>;
  /**
   * Frame of the touch a flight starts at -> height (m) of the ball midway
   * through that flight, set with the arc handle (see rallyRefit.ts).
   */
  arcs?: Record<number, number>;
  /** Keys of problems marked "not a problem". */
  dismissed: string[];
  /** Points set by hand: rally start frame -> the side that won it (see matchTimeline). */
  points?: Record<number, Side>;
  /** Team names by side of the court. */
  teams?: Partial<Record<Side, string>>;
}

export const EMPTY_EDITS: RallyEdits = { version: 1, ball: {}, touches: {}, pins: {}, dismissed: [] };

/** A touch after corrections; `edited` when the user set or placed it, `pinned` when placed in 3D. */
export interface EditedTouch extends BallTouch {
  edited?: boolean;
  pinned?: boolean;
}

/** Nearest frame within `reach` of `idx` that has a camera-motion homography. */
function nearestHomography(frames: FrameData[], idx: number, reach = 5): number[] | undefined {
  for (let d = 0; d <= reach; d++) {
    const H = frames[idx - d]?.ref_from_frame ?? frames[idx + d]?.ref_from_frame;
    if (H) return H;
  }
  return undefined;
}

function applyH(H: number[], [x, y]: [number, number]): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/**
 * Frames with the ball edits applied. A placed ball is a sure detection; its
 * measured size is dropped, as it belonged to whatever was detected before.
 * A moving camera's solver reads the reference-image fields, so those are
 * set too (borrowing a neighbour's camera motion when this frame has none).
 */
export function applyBallEdits(frames: FrameData[], ball: RallyEdits["ball"]): FrameData[] {
  const entries = Object.entries(ball);
  if (!entries.length) return frames;
  const moving = frames.some((f) => f.ref_from_frame);
  const out = frames.slice();
  for (const [key, pos] of entries) {
    const i = Number(key);
    const f = out[i];
    if (!f) continue;
    if (pos === null) {
      out[i] = { ...f, ball_img_pos: undefined, ball_ref_pos: undefined, ball_interpolated: false, ball_conf: 0,
        ball_bbox: undefined, ball_diameter_px: undefined, ball_ref_diameter_px: undefined };
      continue;
    }
    const H = moving ? nearestHomography(frames, i) : undefined;
    out[i] = {
      ...f, ball_img_pos: pos, ball_interpolated: false, ball_conf: 1, ball_bbox: undefined,
      ball_diameter_px: undefined, ball_ref_diameter_px: undefined,
      ball_ref_pos: H ? applyH(H, pos) : undefined,
    };
  }
  return out;
}

/** Solver hints for the touch edits. */
export function touchEditsForSolver(
  touches: RallyEdits["touches"],
  pins: RallyEdits["pins"],
): BallReconstructionOptions["touchEdits"] {
  const touchAt: number[] = [], noTouchAt: number[] = [];
  for (const [f, kind] of Object.entries(touches)) (kind === "none" ? noTouchAt : touchAt).push(Number(f));
  const pinList = Object.entries(pins).map(([f, pos]) => ({ frame: Number(f), pos }));
  return touchAt.length || noTouchAt.length || pinList.length ? { touchAt, noTouchAt, pins: pinList } : undefined;
}

/** How far (frames) a found touch may be from an edit and still be the touch it edits. */
export function touchMatchFrames(fps: number): number {
  return Math.max(3, Math.round(0.15 * fps));
}

/** Solved position at a frame, or the nearest solved one up to half a second away. */
function positionNear(positions: Map<number, V3>, f: number): V3 | null {
  for (let d = 0; d <= 15; d++) {
    const p = positions.get(f - d) ?? positions.get(f + d);
    if (p) return p;
  }
  return null;
}

/**
 * The solver's touches with the touch edits applied: removed ones dropped,
 * marked ones relabelled (and moved to the marked frame), pinned ones put at
 * their pin, and marked or pinned ones the solver did not produce added.
 */
export function applyTouchEdits(
  touches: BallTouch[],
  e: Pick<RallyEdits, "touches" | "pins">,
  positions: Map<number, V3>,
  fps: number,
): EditedTouch[] {
  const frames = [...new Set([...Object.keys(e.touches), ...Object.keys(e.pins)].map(Number))];
  if (!frames.length) return touches;
  const tol = touchMatchFrames(fps);
  const removed = frames.filter((f) => e.touches[f] === "none");
  const out: EditedTouch[] = touches.filter((t) => !removed.some((f) => Math.abs(f - t.frame) <= tol));
  for (const frame of frames) {
    const kind = e.touches[frame];
    if (kind === "none") continue;
    const pin = e.pins[frame];
    let best = -1;
    out.forEach((t, i) => {
      if (t.edited || Math.abs(t.frame - frame) > tol) return;
      if (best < 0 || Math.abs(t.frame - frame) < Math.abs(out[best].frame - frame)) best = i;
    });
    if (best >= 0) {
      const t = out[best];
      out[best] = { ...t, frame, label: kind ?? t.label, position: pin ?? t.position, edited: true, pinned: !!pin };
    } else {
      const position = pin ?? positionNear(positions, frame);
      if (position) out.push({ frame, position, label: kind ?? "touch", edited: true, pinned: !!pin });
    }
  }
  return out.sort((a, b) => a.frame - b.frame);
}

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a: V3): V3 => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/**
 * The camera's line of sight through the ball on a frame, in court coords:
 * the ball is somewhere along it. Null when the ball is not in the picture.
 * Same camera model as the solver (three.js PerspectiveCamera.lookAt).
 */
export function ballRay(
  frames: FrameData[],
  frame: number,
  calibration: { position: V3; lookAt: V3; fov: number; roll?: number },
  aspect: number,
): { origin: V3; dir: V3 } | null {
  const f = frames[frame];
  // A moving camera's calibration belongs to its reference image.
  const img = frames.some((x) => x.ref_from_frame) ? f?.ball_ref_pos : f?.ball_img_pos;
  if (!img) return null;
  const fwd = unit(sub(calibration.lookAt, calibration.position));
  const levelRight = unit(cross(fwd, [0, 1, 0]));
  const levelUp = cross(levelRight, fwd);
  const th = Math.tan((calibration.fov * Math.PI) / 360);
  const u0 = (img[0] * 2 - 1) * th * aspect, v0 = (1 - img[1] * 2) * th;
  const c = Math.cos(calibration.roll ?? 0), s = Math.sin(calibration.roll ?? 0);
  // The image axes, turned by the camera's roll, in terms of the level ones.
  const u = u0 * c - v0 * s, v = u0 * s + v0 * c;
  const right = levelRight, up = levelUp;
  return {
    origin: calibration.position,
    dir: unit([0, 1, 2].map((c) => fwd[c] + right[c] * u + up[c] * v) as V3),
  };
}

/** Saved edits for a video, or empty ones. Desktop keeps them in ~/.tennis-3d, the server with the match. */
export async function loadEdits(videoPath: string): Promise<RallyEdits> {
  const text = await (await getBackend()).loadEdits(videoPath);
  if (!text) return EMPTY_EDITS;
  const saved = JSON.parse(text);
  const e = saved.edits ?? saved;
  return {
    version: 1, ball: e.ball ?? {}, touches: e.touches ?? {}, pins: e.pins ?? {}, arcs: e.arcs ?? {}, dismissed: e.dismissed ?? [],
    points: e.points ?? {}, teams: e.teams ?? {},
  };
}

export async function saveEdits(videoPath: string, e: RallyEdits): Promise<void> {
  const content = JSON.stringify({ video: videoPath, savedAt: new Date().toISOString(), edits: e }, null, 1);
  await (await getBackend()).saveEdits(videoPath, content);
}
