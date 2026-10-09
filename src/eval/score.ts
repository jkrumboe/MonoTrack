import type { BallTouch } from "../utils/ballReconstruction";
import { cameraFor, floorPoint, type PipelineRun } from "./pipeline";
import type { ClipMetrics, Fixture, Truth } from "./types";

/** A found touch counts as the true one if it is within this many seconds. */
export const MATCH_TOLERANCE_S = 0.15;

const median = (a: number[]) => {
  if (a.length === 0) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);

export interface TouchMatch {
  trueFrame: number;
  trueType: string;
  found?: BallTouch;
  /** Court (x, z) of the truth: synthetic position, or labelled feet on the floor. */
  truePos?: [number, number];
  posErrM?: number;
  heightErrM?: number;
}

/** Greedy one-to-one matching of found touches to true ones by time. */
export function matchTouches(fx: Fixture, truth: Truth, run: PipelineRun): TouchMatch[] {
  const fps = fx.video.fps;
  const found = [...(run.result.touches ?? [])];
  const used = new Set<number>();
  const cam = cameraFor(run.calibration, fx.video.width / fx.video.height);
  return truth.touches.map((tt) => {
    let best = -1, bestD = Infinity;
    found.forEach((f, i) => {
      if (used.has(i)) return;
      const d = Math.abs(f.frame - tt.frame) / fps;
      if (d <= MATCH_TOLERANCE_S && d < bestD) { bestD = d; best = i; }
    });
    let truePos: [number, number] | undefined;
    if (tt.position) truePos = [tt.position[0], tt.position[2]];
    else if (tt.feet) {
      // Feet are clicked on the raw frame; a moving camera's frame must be
      // mapped into the reference image the calibration belongs to.
      const H = fx.frames[tt.frame]?.ref_from_frame;
      const [x, y] = tt.feet;
      const w = H ? H[6] * x + H[7] * y + H[8] : 1;
      const ref: [number, number] = H ? [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w] : [x, y];
      truePos = floorPoint(cam, ref) ?? undefined;
    }
    const m: TouchMatch = { trueFrame: tt.frame, trueType: tt.type, truePos };
    if (best >= 0) {
      used.add(best);
      const f = found[best];
      m.found = f;
      if (truePos) m.posErrM = Math.hypot(f.position[0] - truePos[0], f.position[2] - truePos[1]);
      if (tt.position) m.heightErrM = Math.abs(f.position[1] - tt.position[1]);
    }
    return m;
  });
}

/**
 * Combined score, 0-100. Weighted towards what the 3D view is for: finding
 * every touch, putting it in the right place, and a path close to the truth.
 * Terms without data (no path on real clips) are left out and the rest
 * renormalised.
 */
function combine(parts: { w: number; v: number | null }[]): number {
  const have = parts.filter((p) => p.v != null && Number.isFinite(p.v));
  const wsum = have.reduce((s, p) => s + p.w, 0);
  return wsum > 0 ? (100 * have.reduce((s, p) => s + p.w * (p.v as number), 0)) / wsum : 0;
}

export function scoreClip(fx: Fixture, truth: Truth, run: PipelineRun | null): { metrics: ClipMetrics; matches: TouchMatch[] } {
  const synthetic = truth.path != null;
  if (!run) {
    return {
      matches: [],
      metrics: {
        clip: fx.clip, synthetic, touchesTrue: truth.touches.length, touchesFound: 0, touchesMatched: 0,
        recall: 0, precision: 0, typeAccuracy: 0, touchPosErrM: null, touchHeightErrM: null,
        pathRmsM: null, pathCoverage: synthetic ? 0 : null, reprojMedianPx: null, score: 0,
      },
    };
  }
  const matches = matchTouches(fx, truth, run);
  const matched = matches.filter((m) => m.found);
  const nFound = run.result.touches?.length ?? 0;
  const recall = truth.touches.length ? matched.length / truth.touches.length : 1;
  const precision = nFound ? matched.length / nFound : 0;
  const typeAccuracy = matched.length ? matched.filter((m) => m.found!.label === m.trueType).length / matched.length : 0;
  const posErrs = matched.flatMap((m) => (m.posErrM != null ? [m.posErrM] : []));
  const hErrs = matched.flatMap((m) => (m.heightErrM != null ? [m.heightErrM] : []));

  let pathRmsM: number | null = null, pathCoverage: number | null = null;
  if (truth.path) {
    const frames = Object.keys(truth.path).map(Number);
    let ss = 0, n = 0;
    for (const f of frames) {
      const p = run.result.positions.get(f);
      if (!p) continue;
      const q = truth.path[f];
      ss += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
      n++;
    }
    pathRmsM = n ? Math.sqrt(ss / n) : null;
    pathCoverage = frames.length ? n / frames.length : 0;
  }
  const reproj = (run.result.physicsDebug ?? [])
    .filter((d) => d.usedInFit && d.reprojectionErrorPx != null)
    .map((d) => d.reprojectionErrorPx as number);

  const touchPosErrM = posErrs.length ? mean(posErrs) : null;
  const score = combine([
    { w: 0.25, v: recall },
    { w: 0.1, v: precision },
    { w: 0.15, v: matched.length ? typeAccuracy : 0 },
    // 1.5 m off scores ~0.37; a perfect touch 1.
    { w: 0.25, v: touchPosErrM == null ? (truth.touches.some((t) => t.position || t.feet) ? 0 : null) : Math.exp(-touchPosErrM / 1.5) },
    { w: 0.15, v: pathRmsM == null ? (synthetic ? 0 : null) : Math.exp(-pathRmsM / 1.0) * (pathCoverage ?? 1) },
    { w: 0.1, v: reproj.length ? Math.exp(-median(reproj) / 10) : null },
  ]);

  return {
    matches,
    metrics: {
      clip: fx.clip, synthetic,
      touchesTrue: truth.touches.length, touchesFound: nFound, touchesMatched: matched.length,
      recall, precision, typeAccuracy,
      touchPosErrM, touchHeightErrM: hErrs.length ? mean(hErrs) : null,
      pathRmsM, pathCoverage, reprojMedianPx: reproj.length ? median(reproj) : null,
      score,
    },
  };
}
