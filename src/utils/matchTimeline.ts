/**
 * The match as a sequence of rallies and points, built from the rally
 * solver's touches: what a coach scrolls through instead of the raw video.
 *
 * A rally starts at a serve (or the toss before it) and holds every touch up
 * to the next one. Volleyball uses rally-point scoring - the team that wins a
 * rally serves the next - so the side that serves next is the most reliable
 * evidence of who won the point. Where there is no next serve (the last rally,
 * or one the camera missed) the point goes by where the ball came down: in
 * the court, against the side it landed on; out, against the side that
 * touched it last. A point the user set by hand overrides both.
 *
 * Sides are named by the half of the court seen from the camera: "near"
 * (z > 0, the half closest to the camera) and "far". Teams change ends
 * between sets, so a team name belongs to a side only for one set.
 */
import type { BallTouch } from "./ballReconstruction";

export type Side = "near" | "far";
export const SIDES: readonly Side[] = ["near", "far"];
export const otherSide = (s: Side): Side => (s === "near" ? "far" : "near");
/** The half of the court a point lies in (the net is at z = 0). */
export const sideOf = (z: number): Side => (z >= 0 ? "near" : "far");

/** The touch type of a solver label: "block/dig" counts as its first kind. */
export const touchKind = (label: string) => label.split("/")[0];

const COURT_HALF_W = 4.5;
const COURT_HALF_L = 9;
/** Line width (5 cm) plus a margin for the reconstruction's error. */
const LINE_MARGIN_M = 0.15;
/** Below this height (m) the last tracked position counts as the ball coming down. */
const LANDING_MAX_Y = 1.0;
/** Nearer the net than this (m) a landing cannot be given to a side. */
const NET_DEAD_ZONE_M = 0.3;

export interface TimelineTouch {
  frame: number;
  time: number;
  /** Touch type, e.g. "serve", "attack" ("block/dig" stays as the solver gave it). */
  label: string;
  /** Side of the net the touch was made on; null when the solver had no position. */
  side: Side | null;
  position: [number, number, number];
}

export type WinnerSource = "manual" | "next-serve" | "landing";

/** How a rally ended, from the point winner's view. */
export type RallyEnding = "ace" | "service-error" | "kill" | "block" | "error" | null;

export interface Rally {
  /** 1-based number in the match. */
  index: number;
  /** First and last frame of play: the first touch to where the ball was last tracked. */
  startFrame: number;
  endFrame: number;
  /** Frames to play for this rally, with a little before and after. */
  clipStart: number;
  clipEnd: number;
  server: Side | null;
  touches: TimelineTouch[];
  winner: Side | null;
  winnerSource: WinnerSource | null;
  ending: RallyEnding;
  /** Where the ball was last tracked (m), when it was. */
  lastBall: [number, number, number] | null;
  /** Score after this rally. */
  score: Record<Side, number>;
}

export type SideStats = Record<string, number>;

export interface MatchTimeline {
  rallies: Rally[];
  score: Record<Side, number>;
  /** Per side: touch counts by type, plus points by ending ("points", "aces", "kills", ...). */
  stats: Record<Side, SideStats>;
}

export interface TimelineInput {
  touches: readonly BallTouch[];
  /** Solved 3D ball positions per frame. */
  positions: ReadonlyMap<number, readonly [number, number, number]>;
  fps: number;
  totalFrames: number;
  /** Points set by hand: rally start frame -> winner. */
  pointOverrides?: Readonly<Record<number, Side>>;
  /** Seconds of video kept before a rally's first touch and after its last tracked ball. */
  leadInS?: number;
  tailS?: number;
  /**
   * Touches further apart than this (s) belong to different rallies even
   * without a serve between them (the serve was missed).
   */
  maxTouchGapS?: number;
  /** How long (s) after the last touch the ball is still followed to where it lands. */
  maxFlightS?: number;
}

/** True when a touch opens a rally: a toss, or a serve not right after its toss. */
function opensRally(t: BallTouch, prev: BallTouch | undefined): boolean {
  const k = touchKind(t.label);
  if (k === "toss") return true;
  return k === "serve" && (!prev || touchKind(prev.label) !== "toss");
}

/** Where the ball ended up: the last solved position in [from, to]. */
function lastPosition(
  positions: TimelineInput["positions"],
  from: number,
  to: number,
): { frame: number; pos: [number, number, number] } | null {
  let best: { frame: number; pos: [number, number, number] } | null = null;
  for (const [f, p] of positions) {
    if (f < from || f > to) continue;
    if (!best || f > best.frame) best = { frame: f, pos: [p[0], p[1], p[2]] };
  }
  return best;
}

/** Winner by where the ball came down, or null when that is not clear. */
function winnerByLanding(lastBall: [number, number, number] | null, lastToucher: Side | null): Side | null {
  if (!lastBall) return null;
  const [x, y, z] = lastBall;
  if (y > LANDING_MAX_Y) return null;
  const inside = Math.abs(x) <= COURT_HALF_W + LINE_MARGIN_M && Math.abs(z) <= COURT_HALF_L + LINE_MARGIN_M;
  if (inside) return Math.abs(z) < NET_DEAD_ZONE_M ? null : otherSide(sideOf(z));
  return lastToucher ? otherSide(lastToucher) : null;
}

function rallyEnding(r: Pick<Rally, "server" | "winner" | "touches">): RallyEnding {
  if (!r.winner || !r.server) return null;
  const receiverTouched = r.touches.some((t) => t.side && t.side !== r.server);
  if (!receiverTouched) return r.winner === r.server ? "ace" : "service-error";
  const last = [...r.touches].reverse().find((t) => t.side);
  if (!last) return null;
  if (last.side !== r.winner) return "error";
  const k = touchKind(last.label);
  if (k === "attack") return "kill";
  if (k === "block") return "block";
  return null;
}

const ENDING_STAT: Record<Exclude<RallyEnding, null>, string> = {
  "ace": "aces",
  "service-error": "service errors",
  "kill": "kills",
  "block": "block points",
  "error": "errors",
};

export function buildMatchTimeline(inp: TimelineInput): MatchTimeline {
  const { fps, totalFrames, positions } = inp;
  const leadIn = Math.round((inp.leadInS ?? 1.5) * fps);
  const tail = Math.round((inp.tailS ?? 1.5) * fps);
  const maxGap = (inp.maxTouchGapS ?? 8) * fps;
  const maxFlight = Math.round((inp.maxFlightS ?? 4) * fps);
  const lastFrame = Math.max(0, totalFrames - 1);

  // Group touches into rallies.
  const sorted = [...inp.touches].sort((a, b) => a.frame - b.frame);
  const groups: BallTouch[][] = [];
  sorted.forEach((t, i) => {
    const prev = sorted[i - 1];
    const cur = groups[groups.length - 1];
    if (!cur || opensRally(t, prev) || t.frame - prev.frame > maxGap) groups.push([t]);
    else cur.push(t);
  });

  const rallies: Rally[] = groups.map((g, i) => {
    const touches: TimelineTouch[] = g.map((t) => ({
      frame: t.frame,
      time: t.frame / fps,
      label: t.label,
      side: Number.isFinite(t.position[2]) ? sideOf(t.position[2]) : null,
      position: t.position,
    }));
    const startFrame = touches[0].frame;
    const lastTouch = touches[touches.length - 1].frame;
    const nextStart = groups[i + 1]?.[0].frame ?? lastFrame + 1;
    const ball = lastPosition(positions, lastTouch, Math.min(nextStart - 1, lastTouch + maxFlight));
    const endFrame = Math.max(lastTouch, ball?.frame ?? Math.min(lastTouch + fps, nextStart - 1, lastFrame));
    const opener = touches.find((t) => ["toss", "serve"].includes(touchKind(t.label)));
    return {
      index: i + 1,
      startFrame,
      endFrame,
      clipStart: Math.max(0, startFrame - leadIn),
      clipEnd: Math.min(lastFrame, endFrame + tail),
      server: opener?.side ?? null,
      touches,
      winner: null,
      winnerSource: null,
      ending: null,
      lastBall: ball?.pos ?? null,
      score: { near: 0, far: 0 },
    };
  });

  // Point winners, then the running score.
  rallies.forEach((r, i) => {
    const manual = inp.pointOverrides?.[r.startFrame];
    const next = rallies[i + 1];
    const lastToucher = [...r.touches].reverse().find((t) => t.side)?.side ?? null;
    if (manual) {
      r.winner = manual;
      r.winnerSource = "manual";
    } else if (next?.server) {
      r.winner = next.server;
      r.winnerSource = "next-serve";
    } else {
      r.winner = winnerByLanding(r.lastBall, lastToucher);
      r.winnerSource = r.winner ? "landing" : null;
    }
    r.ending = rallyEnding(r);
  });

  const score: Record<Side, number> = { near: 0, far: 0 };
  const stats: Record<Side, SideStats> = { near: {}, far: {} };
  const bump = (s: Side, key: string) => { stats[s][key] = (stats[s][key] ?? 0) + 1; };
  for (const r of rallies) {
    for (const t of r.touches) if (t.side) bump(t.side, touchKind(t.label));
    if (r.winner) {
      score[r.winner]++;
      bump(r.winner, "points");
      if (r.ending) {
        // Errors and service errors are counted against the side that made them.
        const by = r.ending === "error" || r.ending === "service-error" ? otherSide(r.winner) : r.winner;
        bump(by, ENDING_STAT[r.ending]);
      }
    }
    r.score = { ...score };
  }
  return { rallies, score, stats };
}

/** The rally whose clip contains `frame`, or null. */
export function rallyAt(rallies: readonly Rally[], frame: number): Rally | null {
  return rallies.find((r) => frame >= r.clipStart && frame <= r.clipEnd) ?? null;
}

/** The first rally whose clip starts after `frame`, or null. */
export function nextRallyAfter(rallies: readonly Rally[], frame: number): Rally | null {
  return rallies.find((r) => r.clipStart > frame) ?? null;
}
