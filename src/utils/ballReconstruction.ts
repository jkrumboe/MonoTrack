import * as THREE from "three";
import type { FrameData, BallReconstructionMethod } from "../hooks/useAnalysis";
import type { CalibrationResult } from "./cameraSolver";
import { applyHomography, imageToCourtHomography } from "./courtHomography";

// FIVB volleyball: 65–67 cm circumference → ~0.21 m diameter.
// Override via BallReconstructionOptions.ballDiameterM if needed.
export const DEFAULT_BALL_DIAMETER_M = 0.21;
const HALF_BALL = DEFAULT_BALL_DIAMETER_M / 2;
const COURT_HALF_W = 4.5;
const COURT_HALF_L = 9;
const MAX_BALL_HEIGHT_M = 8;
const COURT_WIDTH = 9;
const COURT_LENGTH = 18;

/** Which bbox dimension to use when computing depth from apparent ball size. */
export type DiameterMode = "width" | "height" | "min-side" | "max-side" | "average";

/** Controls how 3D ball positions are computed from the per-frame observations. */
export type TrajectoryMode =
  | "raw-bbox-depth"        // per-frame, raw apparent-size depth
  | "smoothed-bbox-depth"   // per-frame, median-smoothed bbox diameter
  | "physics-reprojection"; // ballistic arc fit minimising 2D reprojection error

export interface BallReconstructionOptions {
  /** Real-world ball diameter in metres. Default: 0.21 (FIVB volleyball). */
  ballDiameterM?: number;
  /**
   * Force a specific reconstruction method, or "auto" (default) for normal priority.
   * "bbox-depth" isolates the known-size depth estimate and skips all court-plane methods.
   */
  forceMethod?: "bbox-depth" | "bbox-bottom-ray" | "homography-ray" | "ground-intersection" | "auto";
  /**
   * When false (default for debug), use per-frame bbox-depth estimation directly.
   * Set to true to re-enable ballistic arc trajectory fitting.
   */
  useBallisticSolver?: boolean;
  /**
   * Which bbox dimension to use for the final depth estimate.
   * "min-side": min(zFromWidth, zFromHeight) — uses the larger apparent side, less conservative.
   * Default: "min-side".
   */
  diameterMode?: DiameterMode;
  /**
   * When true, bypass the isPlausibleInCourtVolume check for bbox-depth estimates.
   * The position is still recorded with rejectedReason set when it would have failed.
   * Useful for diagnosing false rejections. Default: false.
   */
  debugMode?: boolean;
  /** Actual video width in pixels — used for pixel-space depth calculation. Default: derived from aspect × 1080. */
  videoWidth?: number;
  /** Actual video height in pixels — used for pixel-space depth calculation. Default: 1080. */
  videoHeight?: number;
  /**
   * How to compute the final 3D trajectory. Default: "raw-bbox-depth".
   *   "raw-bbox-depth"       — per-frame apparent-size depth (noisy but no fitting latency)
   *   "smoothed-bbox-depth"  — same but diameter median-smoothed ±3 detected-frames
   *   "physics-reprojection" — ballistic arc fit minimising 2D reprojection error
   */
  trajectoryMode?: TrajectoryMode;
  /**
   * When true (default), use the FoV from camera calibration.
   * Set to false to use manualFovYDeg instead.
   */
  useCalibrationFov?: boolean;
  /** Manual vertical FoV override in degrees. Used only when useCalibrationFov = false. */
  manualFovYDeg?: number;
  /**
   * Minimum YOLO confidence to use a detection as a 3D observation.
   * Below this threshold the detection is kept for 2D visualisation but excluded from
   * depth / arc fitting. Default: 0.6.
   */
  minBallConfidence?: number;
}

export interface ReconstructionStats {
  /** Frames with a YOLO detection (not interpolated). */
  detectedFrames: number;
  /** Frames that received a 3D position. */
  solved3dFrames: number;
  /** Frames that had a detection but were rejected (failed plausibility checks). */
  rejectedFrames: number;
  /** Count of positions produced per method. */
  methodCounts: Partial<Record<BallReconstructionMethod, number>>;
  /** Mean YOLO confidence over detected (non-interpolated) frames. 0 if none. */
  avgConf: number;
}

/** Per-frame debug record for ball 3D reconstruction attempts. */
export interface BallFrameDebug {
  frame: number;
  /** Normalised image-space X remapped to NDC [-1, 1]. */
  u: number;
  /** Normalised image-space Y remapped to NDC [-1, 1]. */
  v: number;
  /** Video width used for pixel-space calculations (px). */
  videoWidth: number;
  /** Video height used for pixel-space calculations (px). */
  videoHeight: number;
  /** Raw normalised bbox from YOLO: [x1/imgW, y1/imgH, x2/imgW, y2/imgH]. */
  bbox: [number, number, number, number] | null;
  /** Coordinate format assumed for the bbox field. Always "x1y1x2y2-norm". */
  bboxFormatAssumed: "x1y1x2y2-norm";
  bboxWidthNorm: number | null;
  bboxHeightNorm: number | null;
  /** Apparent bbox width in pixels (bboxWidthNorm × videoWidth). */
  bboxWidthPx: number | null;
  /** Apparent bbox height in pixels (bboxHeightNorm × videoHeight). */
  bboxHeightPx: number | null;
  confidence: number | null;
  fovY: number;
  fovX: number;
  /** Horizontal pixel focal length: videoWidth / (2·tan(fovX/2)). */
  fx: number;
  /** Vertical pixel focal length: videoHeight / (2·tan(fovY/2)). */
  fy: number;
  ballDiameterM: number;
  zFromWidth: number | null;
  zFromHeight: number | null;
  /** min(zFromWidth, zFromHeight) — depth using the larger apparent bbox side. */
  zFromMinSide: number | null;
  zUsed: number | null;
  /** Raw min-side bbox diameter in pixels (before smoothing). */
  diameterPxRaw: number | null;
  /** Median-smoothed bbox diameter in pixels (±3 detected-frame window). */
  diameterPxSmoothed: number | null;
  /** Depth estimate computed from smoothedDiameterPx. */
  zUsedSmoothed: number | null;
  method: BallReconstructionMethod | null;
  /** Raw 3D position before plausibility check (set whenever bbox + ray compute succeeds). */
  worldPositionRaw: [number, number, number] | null;
  /** 3D position after plausibility check. Null if rejected (unless debugMode bypasses check). */
  worldPosition: [number, number, number] | null;
  /** True if the position passed the isPlausibleInCourtVolume check. */
  accepted: boolean;
  rejectedReason: string | null;
  /** Camera world position from calibration. */
  cameraPosition: [number, number, number] | null;
  /** True if this frame was eligible to be included in trajectory fitting (conf ≥ minBallConfidence). */
  usedInFit: boolean;
  /** NDC X of the projected fitted trajectory position (physics-reprojection mode). */
  projectedU: number | null;
  /** NDC Y of the projected fitted trajectory position (physics-reprojection mode). */
  projectedV: number | null;
  /** Pixel reprojection error between observed and projected positions. */
  reprojectionErrorPx: number | null;
  /** Fitted ballistic trajectory position at this frame. */
  fittedPosition: [number, number, number] | null;
}

/**
 * Per-frame output from physics-reprojection mode.
 * Exported separately so callers can save/inspect the full reprojection diagnostic.
 */
export interface PhysicsReprojectionDebug {
  frame: number;
  observedU: number;
  observedV: number;
  projectedU: number | null;
  projectedV: number | null;
  reprojectionErrorPx: number | null;
  fittedX: number | null;
  fittedY: number | null;
  fittedZ: number | null;
  rawBboxDepthZ: number | null;
  smoothedBboxDepthZ: number | null;
  confidence: number | null;
  usedInFit: boolean;
  /** Which arc produced this observation (null for unmatched detections). */
  arcId: number | null;
  /** Whether the arc was accepted after physical validation. */
  arcAccepted: boolean | null;
  /** Reason the arc was rejected, or null if accepted. */
  arcRejectReason: string | null;
  /** Ball speed (m/s) at this frame relative to the previous fitted point. */
  speedMps: number | null;
  /** Camera-space forward depth of the fitted world point (m). */
  cameraDepthFitted: number | null;
  /** (cameraDepthFitted − smoothedBboxDepthZ) / smoothedBboxDepthZ, or null. */
  depthPriorError: number | null;
  /** True when this observation was removed during iterative outlier rejection. */
  isOutlier: boolean;
}

export interface BallTrajectoryResult {
  positions: Map<number, [number, number, number]>;
  methods: Map<number, BallReconstructionMethod>;
  stats: ReconstructionStats;
  /** Per-frame debug records for all detected (non-interpolated) ball frames. */
  debugRecords: BallFrameDebug[];
  /** Physics-reprojection mode diagnostic (only present when trajectoryMode = "physics-reprojection"). */
  physicsDebug?: PhysicsReprojectionDebug[];
}

export interface BallWorldEstimate {
  position: [number, number, number];
  ground: [number, number, number];
  height: number;
  method: BallReconstructionMethod;
}

function isPlausibleInCourtVolume(pos: THREE.Vector3): boolean {
  return (
    Number.isFinite(pos.x) &&
    Number.isFinite(pos.y) &&
    Number.isFinite(pos.z) &&
    Math.abs(pos.x) <= COURT_HALF_W + 2 &&
    Math.abs(pos.z) <= COURT_HALF_L + 4 &&
    pos.y >= HALF_BALL &&
    pos.y <= MAX_BALL_HEIGHT_M
  );
}

function createCamera(
  calibration: CalibrationResult,
  aspect: number,
  fovYDeg?: number,
): THREE.PerspectiveCamera {
  const fov = fovYDeg ?? calibration.fov;
  const cam = new THREE.PerspectiveCamera(fov, aspect, 0.1, 500);
  cam.position.set(...calibration.position);
  cam.lookAt(new THREE.Vector3(...calibration.lookAt));
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  return cam;
}

function rayFromImagePos(
  cam: THREE.PerspectiveCamera,
  imgPos: [number, number],
): { origin: THREE.Vector3; dir: THREE.Vector3 } {
  const ndcX = imgPos[0] * 2 - 1;
  const ndcY = 1 - imgPos[1] * 2;
  const origin = cam.position.clone();
  const worldPoint = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(cam);
  const dir = worldPoint.sub(origin).normalize();
  return { origin, dir };
}

function normCourtToWorld(nx: number, ny: number): [number, number] {
  const x = (nx - 0.5) * COURT_WIDTH;
  const z = (ny - 0.5) * COURT_LENGTH;
  return [x, z];
}

function estimateHeightFromRayAndGroundXZ(
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  x: number,
  z: number,
): number {
  const useX = Math.abs(dir.x) >= Math.abs(dir.z);
  let t = 0;
  if (useX && Math.abs(dir.x) > 1e-5) {
    t = (x - origin.x) / dir.x;
  } else if (Math.abs(dir.z) > 1e-5) {
    t = (z - origin.z) / dir.z;
  }
  if (t <= 0 || !Number.isFinite(t)) return HALF_BALL;
  const y = origin.y + t * dir.y;
  return Math.max(HALF_BALL, Math.min(MAX_BALL_HEIGHT_M, y));
}

/**
 * Reconstruct a volleyball 3D point from monocular data.
 *
 * Priority:
 * 1) bbox-bottom-ray: Use the bbox bottom-centre as the ground shadow point (depth-map
 *    ray cast to y=0 ground plane) for accurate X/Z, then derive Y from apparent bbox size.
 * 2) homography-ray: Fall back to court-corner homography when bbox is absent.
 * 3) bbox-depth: Apparent-size depth estimate without ground constraint.
 * 4) ground-intersection: Plain ray-ground intersection.
 */

function rayHeightFromDepth(
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  cam: THREE.PerspectiveCamera,
  depthCam: number,
): number {
  const invQ = cam.quaternion.clone().invert();
  const forward = -dir.clone().applyQuaternion(invQ).z;
  if (forward <= 1e-5) return HALF_BALL;
  const t = depthCam / forward;
  return Math.max(HALF_BALL, origin.clone().addScaledVector(dir, t).y);
}

interface BboxDepthDetail {
  zFromHeight: number;
  zFromWidth: number;
  /** min(zFromWidth, zFromHeight) — uses the larger apparent side, gives the smaller depth. */
  zFromMinSide: number;
  zUsed: number;
  fovX: number;
  fovY: number;
  /** Horizontal pixel focal length: videoWidth / (2·tan(fovX/2)). */
  fx: number;
  /** Vertical pixel focal length: videoHeight / (2·tan(fovY/2)). */
  fy: number;
  bboxWidthPx: number;
  bboxHeightPx: number;
}

/**
 * Compute monocular depth from known ball diameter and apparent bbox size.
 *
 * Uses separate vertical (fovY) and horizontal (fovX) fields of view so that
 * non-square image sensors / aspect ratios are handled correctly.
 *
 *   fy = videoHeight / (2·tan(fovY/2))         (pixel focal length)
 *   fx = videoWidth  / (2·tan(fovX/2))         where fovX = 2·atan(tan(fovY/2)·aspect)
 *   zFromHeight = fy·D / bboxHeightPx           ≡ D / (h_norm·2·tan(fovY/2))
 *   zFromWidth  = fx·D / bboxWidthPx            ≡ D / (w_norm·2·tan(fovX/2))
 *
 * The pixel and normalised forms are algebraically identical; pixel form is
 * exposed for debugging (bboxWidthPx, bboxHeightPx, fx, fy visible in JSON).
 */
function bboxDepthCamDetail(
  bbox: [number, number, number, number],
  fovYDeg: number,
  ballDiameterM: number,
  aspect: number,
  videoWidth = 1920,
  videoHeight = 1080,
  diameterMode: DiameterMode = "min-side",
): BboxDepthDetail {
  const [x1, y1, x2, y2] = bbox;
  const w = Math.max(1e-4, Math.abs(x2 - x1)); // normalised image width fraction
  const h = Math.max(1e-4, Math.abs(y2 - y1)); // normalised image height fraction
  const halfFovY = (fovYDeg * Math.PI) / 360;  // fovYDeg/2 in radians
  const halfFovX = Math.atan(Math.tan(halfFovY) * aspect);

  // Pixel-space focal lengths (exposed for debug output; identical to normalised formula)
  const fy = videoHeight / (2 * Math.tan(halfFovY));
  const fx = videoWidth  / (2 * Math.tan(halfFovX));

  const bboxWidthPx  = w * videoWidth;
  const bboxHeightPx = h * videoHeight;

  const zFromHeight = fy * ballDiameterM / Math.max(bboxHeightPx, 1e-3);
  const zFromWidth  = fx * ballDiameterM / Math.max(bboxWidthPx,  1e-3);
  const zFromMinSide = Math.min(zFromWidth, zFromHeight);

  let zUsed: number;
  if (diameterMode === "width")          zUsed = zFromWidth;
  else if (diameterMode === "height")    zUsed = zFromHeight;
  else if (diameterMode === "min-side")  zUsed = zFromMinSide;
  else if (diameterMode === "max-side")  zUsed = Math.max(zFromWidth, zFromHeight);
  else                                   zUsed = (zFromWidth + zFromHeight) / 2; // "average"

  return {
    zFromHeight, zFromWidth, zFromMinSide, zUsed,
    fovX: (halfFovX * 360) / Math.PI, fovY: fovYDeg,
    fx, fy, bboxWidthPx, bboxHeightPx,
  };
}

/** Groups ball-size / video-space parameters used by the depth-estimation helpers. */
interface DepthContext {
  ballDiameterM: number;
  aspect: number;
  videoWidth: number;
  videoHeight: number;
  diameterMode: DiameterMode;
  /** Effective vertical FoV in degrees (may be manually overridden). */
  fovYDeg: number;
}

function tryBboxBottomRay(
  frame: Pick<FrameData, "ball_img_pos" | "ball_bbox">,
  cam: THREE.PerspectiveCamera,
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  ctx: DepthContext,
): BallWorldEstimate | null {
  if (!frame.ball_bbox || !frame.ball_img_pos) return null;
  const [x1, , x2, y2] = frame.ball_bbox;
  const { origin: bOrig, dir: bDir } = rayFromImagePos(cam, [(x1 + x2) * 0.5, y2]);
  if (Math.abs(bDir.y) <= 1e-6) return null;
  const tGnd = -bOrig.y / bDir.y;
  if (tGnd <= 0) return null;
  const groundPt = bOrig.clone().addScaledVector(bDir, tGnd);
  const { zUsed: depth } = bboxDepthCamDetail(
    frame.ball_bbox, ctx.fovYDeg, ctx.ballDiameterM, ctx.aspect,
    ctx.videoWidth, ctx.videoHeight, ctx.diameterMode,
  );
  const ballY = rayHeightFromDepth(origin, dir, cam, depth);
  const pos = new THREE.Vector3(groundPt.x, ballY, groundPt.z);
  if (!isPlausibleInCourtVolume(pos)) return null;
  return {
    position: [pos.x, pos.y, pos.z],
    ground: [groundPt.x, 0, groundPt.z],
    height: ballY,
    method: "bbox-bottom-ray",
  };
}

function tryHomographyRay(
  frame: Pick<FrameData, "ball_img_pos">,
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  courtCorners: number[][],
): BallWorldEstimate | null {
  const H = imageToCourtHomography(courtCorners);
  if (!H || !frame.ball_img_pos) return null;
  const [nx, ny] = applyHomography(H, frame.ball_img_pos[0], frame.ball_img_pos[1]);
  if (nx < -0.2 || nx > 1.2 || ny < -0.2 || ny > 1.2) return null;
  const [hx, hz] = normCourtToWorld(nx, ny);
  const h = estimateHeightFromRayAndGroundXZ(origin, dir, hx, hz);
  const pos = new THREE.Vector3(hx, h, hz);
  if (!isPlausibleInCourtVolume(pos)) return null;
  return {
    position: [pos.x, pos.y, pos.z],
    ground: [pos.x, 0, pos.z],
    height: pos.y,
    method: "homography-ray",
  };
}

function tryBboxDepth(
  frame: Pick<FrameData, "ball_img_pos" | "ball_bbox">,
  cam: THREE.PerspectiveCamera,
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  ctx: DepthContext,
  skipPlausibility = false,
): BallWorldEstimate | null {
  if (!frame.ball_bbox) return null;
  const { zUsed: depth } = bboxDepthCamDetail(
    frame.ball_bbox, ctx.fovYDeg, ctx.ballDiameterM, ctx.aspect,
    ctx.videoWidth, ctx.videoHeight, ctx.diameterMode,
  );
  const invQ = cam.quaternion.clone().invert();
  const forward = -dir.clone().applyQuaternion(invQ).z;
  if (forward <= 1e-5) return null;
  const pos = origin.clone().addScaledVector(dir, depth / forward);
  pos.y = Math.max(HALF_BALL, pos.y);
  if (!skipPlausibility && !isPlausibleInCourtVolume(pos)) return null;
  return {
    position: [pos.x, pos.y, pos.z],
    ground: [pos.x, 0, pos.z],
    height: pos.y,
    method: "bbox-depth",
  };
}

export function estimateBallWorldPosition(
  frame: Pick<FrameData, "ball_img_pos" | "ball_bbox">,
  calibration: CalibrationResult,
  aspect: number,
  courtCorners?: number[][],
  options?: BallReconstructionOptions,
): BallWorldEstimate | null {
  if (!frame.ball_img_pos) return null;
  if (aspect <= 0) return null;

  const ballDiameterM = options?.ballDiameterM ?? DEFAULT_BALL_DIAMETER_M;
  const forceMethod   = options?.forceMethod   ?? "auto";
  const diameterMode  = options?.diameterMode  ?? "min-side";
  const videoHeight   = options?.videoHeight   ?? 1080;
  const videoWidth    = options?.videoWidth    ?? Math.round(videoHeight * aspect);
  const effectiveFovY = (options?.useCalibrationFov ?? true)
    ? calibration.fov
    : (options?.manualFovYDeg ?? calibration.fov);
  const ctx: DepthContext = { ballDiameterM, aspect, videoWidth, videoHeight, diameterMode, fovYDeg: effectiveFovY };
  const cam = createCamera(calibration, aspect, effectiveFovY);
  const { origin, dir } = rayFromImagePos(cam, frame.ball_img_pos);

  // ── 1. bbox-depth: primary for airborne balls (known-size depth) ────────────
  // Placed first because the ball is typically airborne, not touching the ground.
  if (forceMethod === "auto" || forceMethod === "bbox-depth") {
    const r = tryBboxDepth(frame, cam, origin, dir, ctx);
    if (r) return r;
  }
  if (forceMethod === "bbox-depth") return null; // hard stop in forced mode

  // ── 2. bbox-bottom-ray: ground-plane constraint (ball near ground) ────────
  if (forceMethod === "auto" || forceMethod === "bbox-bottom-ray") {
    const r = tryBboxBottomRay(frame, cam, origin, dir, ctx);
    if (r) return r;
  }
  if (forceMethod === "bbox-bottom-ray") return null;

  // ── 3. homography-ray: court-keypoint homography ──────────────────────
  if ((forceMethod === "auto" || forceMethod === "homography-ray") && courtCorners?.length === 4) {
    const r = tryHomographyRay(frame, origin, dir, courtCorners);
    if (r) return r;
  }
  if (forceMethod === "homography-ray") return null;

  // ── 4. ground-intersection: last resort ─────────────────────────────
  if (forceMethod === "auto" || forceMethod === "ground-intersection") {
    if (Math.abs(dir.y) < 1e-6) return null;
    const tGround = -origin.y / dir.y;
    if (tGround <= 0) return null;
    const onGround = origin.clone().addScaledVector(dir, tGround);
    return {
      position: [onGround.x, HALF_BALL, onGround.z],
      ground: [onGround.x, 0, onGround.z],
      height: HALF_BALL,
      method: "ground-intersection",
    };
  }

  return null;
}

// ─── Physics-based ballistic trajectory solver ────────────────────────────────

const GRAVITY = 9.81; // m/s²
const MIN_ARC_FRAMES = 4; // minimum observations needed to solve an arc
/** Split a new arc when the time gap between consecutive observations exceeds this (s). */
const MAX_ARC_GAP_S = 0.5;
/** Hard cap on a single arc's total duration (s). */
const MAX_ARC_DURATION_S = 2.5;
/** Split when the frame-index gap between consecutive observations exceeds this. */
const MAX_FRAME_GAP = 3;
/** Split when the NDC 2D jump between consecutive observations exceeds this. */
const MAX_2D_JUMP_NDC = 0.3;
/** Split when bbox camera-ray depth changes by more than this fraction. */
const MAX_DEPTH_CHANGE_FRAC = 0.3;
/** Weight of the soft camera-depth prior in the arc solver objective. */
const DEPTH_PRIOR_WEIGHT = 2e-3;
/** Pixel error threshold for iterative outlier rejection. */
const OUTLIER_ERR_PX = 30;
/** Minimum inlier observations after outlier rejection to attempt fitting. */
const MIN_ARC_INLIERS = 5;
/** Maximum average reprojection error (px) for an accepted arc. */
const MAX_ARC_AVG_ERR_PX = 25;
/** Maximum single-frame reprojection error (px) for an accepted arc. */
const MAX_ARC_MAX_ERR_PX = 50;
/** Maximum ball speed (m/s) between consecutive observations. */
const MAX_BALL_SPEED_MPS = 40;
/** Minimum fitted y (m) — below this signals a bounce or calibration error. */
const MIN_FIT_Y_M = -0.1;
/** Maximum fitted y (m). */
const MAX_FIT_Y_M = 6.0;
/** Z-axis soft bound: >20% of points outside this triggers rejection. */
const COURT_Z_SOFT_M = 11;
/** Z-axis hard bound: any single point outside this triggers rejection. */
const COURT_Z_HARD_M = 14;

/**
 * Nelder-Mead simplex optimizer for unconstrained minimisation.
 * params:  initial guess (n-dim array)
 * fn:      objective function, returns scalar
 * Returns: optimised parameter array
 */
interface NmParams { ALPHA: number; GAMMA: number; RHO: number; SIGMA: number }
interface NmState { simplex: number[][]; scores: number[]; sorted: number[]; n: number }

function nelderMead(
  params: number[],
  fn: (p: number[]) => number,
  maxIter = 2000,
  tol = 1e-7,
): number[] {
  const n = params.length;
  const nmP: NmParams = { ALPHA: 1, GAMMA: 2, RHO: 0.5, SIGMA: 0.5 };

  // Build initial simplex: one vertex per dimension + initial guess
  const simplex: number[][] = [params.slice()];
  for (let i = 0; i < n; i++) {
    const v = params.slice();
    v[i] += 0.5;
    simplex.push(v);
  }

  let scores = simplex.map(fn);

  for (let iter = 0; iter < maxIter; iter++) {
    const order = scores.map((s, i) => [s, i] as [number, number]).sort((a, b) => a[0] - b[0]);
    const sorted = order.map(([, i]) => i);

    if (scores[sorted[n]] - scores[sorted[0]] < tol) break;

    const centroid = new Array<number>(n).fill(0);
    for (let k = 0; k < n; k++) {
      centroid[k] = sorted.slice(0, n).reduce((s, idx) => s + simplex[idx][k], 0) / n;
    }

    const st: NmState = { simplex, scores, sorted, n };
    scores = _nmStep(st, centroid, fn, nmP);
  }

  const bestIdx = scores.indexOf(Math.min(...scores));
  return simplex[bestIdx];
}

function _nmStep(
  st: NmState,
  centroid: number[],
  fn: (p: number[]) => number,
  p: NmParams,
): number[] {
  const { simplex, scores, sorted, n } = st;
  const worst = simplex[sorted[n]].slice();
  const best = simplex[sorted[0]].slice();

  const reflected = centroid.map((c, k) => c + p.ALPHA * (c - worst[k]));
  const reflScore = fn(reflected);

  if (reflScore < scores[sorted[0]]) {
    const expanded = centroid.map((c, k) => c + p.GAMMA * (reflected[k] - c));
    const expScore = fn(expanded);
    simplex[sorted[n]] = expScore < reflScore ? expanded : reflected;
    scores[sorted[n]] = Math.min(expScore, reflScore);
    return scores;
  }

  if (reflScore < scores[sorted[n - 1]]) {
    simplex[sorted[n]] = reflected;
    scores[sorted[n]] = reflScore;
    return scores;
  }

  const contracted = centroid.map((c, k) => c + p.RHO * (worst[k] - c));
  if (fn(contracted) < scores[sorted[n]]) {
    simplex[sorted[n]] = contracted;
    scores[sorted[n]] = fn(contracted);
    return scores;
  }

  // Shrink
  for (let i = 1; i <= n; i++) {
    simplex[sorted[i]] = best.map((b, k) => b + p.SIGMA * (simplex[sorted[i]][k] - b));
    scores[sorted[i]] = fn(simplex[sorted[i]]);
  }
  return scores;
}

/** Project a 3D world point through the calibrated camera to NDC [-1,1]. */
function projectToNDC(
  cam: THREE.PerspectiveCamera,
  wx: number,
  wy: number,
  wz: number,
): [number, number] {
  const v = new THREE.Vector3(wx, wy, wz).project(cam);
  return [v.x, v.y];
}

/**
 * Solve for the 3D ballistic trajectory that best explains a sequence of
 * 2D ball observations through a calibrated camera.
 *
 * obs:          array of {t: seconds, u: NDC_x [-1,1], v: NDC_y [-1,1]}
 * cam:          Three.js PerspectiveCamera matching the calibration
 * initPos:      initial 3D guess for the ball at t=0 (e.g. from ground-intersection)
 *
 * Returns array of [x, y, z] world positions at the same times as obs,
 * or null if the solve fails to converge.
 */
function solveArc(
  obs: { t: number; u: number; v: number; fi: number }[],
  cam: THREE.PerspectiveCamera,
): { fi: number; pos: [number, number, number] }[] | null {
  if (obs.length < MIN_ARC_FRAMES) return null;

  const t0 = obs[0].t;
  const ts = obs.map((o) => o.t - t0); // relative times

  // Initial guess: put ball at a plausible court position with zero velocity
  // Use ray-ground intersection of first observation as starting X/Z
  const dir0 = new THREE.Vector3(obs[0].u, obs[0].v, 0.5).unproject(cam).sub(cam.position).normalize();
  const tGnd = Math.abs(dir0.y) > 1e-6 ? -cam.position.y / dir0.y : 0;
  const initX = tGnd > 0 ? cam.position.x + dir0.x * tGnd : 0;
  const initZ = tGnd > 0 ? cam.position.z + dir0.z * tGnd : 0;

  // params: [x0, y0, z0, vx, vy, vz]
  const init = [initX, 0.5, initZ, 0, 3, 0];

  const objective = (p: number[]) => {
    const [x0, y0, z0, vx, vy, vz] = p;
    let err = 0;
    for (let i = 0; i < obs.length; i++) {
      const t = ts[i];
      const wx = x0 + vx * t;
      const wy = y0 + vy * t - 0.5 * GRAVITY * t * t;
      const wz = z0 + vz * t;
      // Penalise positions below ground or above max height
      if (wy < -0.5 || wy > MAX_BALL_HEIGHT_M + 1) err += 10;
      const [pu, pv] = projectToNDC(cam, wx, wy, wz);
      const du = pu - obs[i].u;
      const dv = pv - obs[i].v;
      err += du * du + dv * dv;
    }
    return err;
  };

  const solution = nelderMead(init, objective, 3000, 1e-8);
  const finalErr = objective(solution);

  // Reject if reprojection error per frame is unreasonably large (>5% of NDC)
  if (finalErr / obs.length > 0.05) return null;

  const [x0, y0, z0, vx, vy, vz] = solution;
  return obs.map((o, i) => {
    const t = ts[i];
    const wx = x0 + vx * t;
    const wy = Math.max(0, y0 + vy * t - 0.5 * GRAVITY * t * t);
    const wz = z0 + vz * t;
    return { fi: o.fi, pos: [wx, wy, wz] as [number, number, number] };
  });
}

type ArcObs = { t: number; u: number; v: number; fi: number };

function buildObservations(
  frames: Pick<FrameData, "frame" | "timestamp" | "ball_img_pos">[],
): ArcObs[] {
  const obs: ArcObs[] = [];
  for (const f of frames) {
    if (!f.ball_img_pos) continue;
    obs.push({
      t: f.timestamp,
      u: f.ball_img_pos[0] * 2 - 1,
      v: 1 - f.ball_img_pos[1] * 2,
      fi: f.frame,
    });
  }
  return obs;
}

function splitIntoArcs(obs: ArcObs[]): ArcObs[][] {
  if (obs.length === 0) return [];
  const arcs: ArcObs[][] = [];
  let arc: ArcObs[] = [obs[0]];
  for (let i = 1; i < obs.length; i++) {
    const gapS  = obs[i].t - obs[i - 1].t;
    const spanS = obs[i].t - arc[0].t;
    if (gapS > MAX_ARC_GAP_S || spanS > MAX_ARC_DURATION_S) {
      arcs.push(arc);
      arc = [];
    }
    arc.push(obs[i]);
  }
  arcs.push(arc);
  return arcs;
}

/**
 * Stricter arc splitter for the physics-reprojection path.
 * Splits on time gap, arc duration, frame-index gap, 2D NDC jump, or depth discontinuity.
 */
function splitArcsEx(obs: ArcObsEx[]): ArcObsEx[][] {
  if (obs.length === 0) return [];
  const arcs: ArcObsEx[][] = [];
  let arc: ArcObsEx[] = [obs[0]];
  for (let i = 1; i < obs.length; i++) {
    const prev = obs[i - 1];
    const curr = obs[i];
    const gapS     = curr.t - prev.t;
    const spanS    = curr.t - arc[0].t;
    const frameGap = curr.fi - prev.fi;
    const jump2d   = Math.hypot(curr.u - prev.u, curr.v - prev.v);
    let depthChangeFrac = 0;
    if (prev.depthZ != null && curr.depthZ != null && prev.depthZ > 0.5) {
      depthChangeFrac = Math.abs(curr.depthZ - prev.depthZ) / prev.depthZ;
    }
    const shouldSplit =
      gapS     > MAX_ARC_GAP_S          ||
      spanS    > MAX_ARC_DURATION_S     ||
      frameGap > MAX_FRAME_GAP          ||
      jump2d   > MAX_2D_JUMP_NDC        ||
      depthChangeFrac > MAX_DEPTH_CHANGE_FRAC;
    if (shouldSplit) {
      arcs.push(arc);
      arc = [curr];
    } else {
      arc.push(curr);
    }
  }
  arcs.push(arc);
  return arcs;
}

// ─── Physics-reprojection arc solver ─────────────────────────────────────────

/** Extended arc observation: includes confidence and optional smoothed depth prior. */
interface ArcObsEx extends ArcObs {
  conf: number;
  depthZ: number | null;
}

/** Per-observation output from the arc solver (includes outliers). */
interface ArcSolveObservation {
  fi: number;
  /** Fitted 3D position — y is NOT clamped to ground. */
  pos: [number, number, number];
  projU: number;
  projV: number;
  errPx: number;
  /** Ball speed (m/s) relative to the previous observation (0 for the first). */
  speedMps: number;
  /** Camera-space forward depth of this fitted world point (m). */
  cameraDepthFitted: number;
  /** (cameraDepthFitted − depthZ) / depthZ, or null when no depth prior is available. */
  depthPriorError: number | null;
  /** True when removed by iterative outlier rejection. */
  isOutlier: boolean;
  obsRef: ArcObsEx;
}

interface ArcSolveResult {
  arcParams: [number, number, number, number, number, number];
  t0: number;
  accepted: boolean;
  rejectReason: string | null;
  /** All original observations including outliers. */
  observations: ArcSolveObservation[];
}

/**
 * Fit a ballistic trajectory to a sequence of 2D observations by minimising
 * confidence-weighted reprojection error with a soft camera-depth prior.
 * Includes iterative outlier rejection and physical arc validation.
 *
 * Returns null only when the arc has fewer than MIN_ARC_FRAMES observations.
 * Otherwise always returns an ArcSolveResult with `accepted` indicating validity.
 */
function solveArcReprojection(
  obs: ArcObsEx[],
  cam: THREE.PerspectiveCamera,
  videoWidth: number,
  videoHeight: number,
  arcId = 0,
): ArcSolveResult | null {
  if (obs.length < MIN_ARC_FRAMES) return null;

  const t0 = obs[0].t;

  // Camera-space forward direction for the depth prior.
  const camFwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
  const [cfx, cfy, cfz] = [camFwd.x, camFwd.y, camFwd.z];
  const [cpx, cpy, cpz] = [cam.position.x, cam.position.y, cam.position.z];

  // Initial position & velocity guess using bbox-depth ray projection.
  let initX = 0, initY = 1.0, initZ = 0;
  let initVx = 0, initVy = 4, initVz = -5;
  const invQ = cam.quaternion.clone().invert();
  {
    const d0 = new THREE.Vector3(obs[0].u, obs[0].v, 0.5)
      .unproject(cam).sub(cam.position).normalize();
    const depth0 = obs[0].depthZ;
    if (depth0 != null && depth0 > 0.1) {
      const fwd0 = -d0.clone().applyQuaternion(invQ).z;
      if (fwd0 > 1e-5) {
        const p = cam.position.clone().addScaledVector(d0, depth0 / fwd0);
        initX = p.x; initY = Math.max(0.1, p.y); initZ = p.z;
      }
    } else {
      const tGnd = Math.abs(d0.y) > 1e-4 ? -cam.position.y / d0.y : 0;
      if (tGnd > 0 && tGnd < 200) {
        initX = cam.position.x + d0.x * tGnd;
        initZ = cam.position.z + d0.z * tGnd;
      }
    }
    const worldZs: number[] = [];
    for (const o of obs) {
      if (o.depthZ == null || o.depthZ <= 0.1) continue;
      const dv = new THREE.Vector3(o.u, o.v, 0.5).unproject(cam).sub(cam.position).normalize();
      const fwdV = -dv.clone().applyQuaternion(invQ).z;
      if (fwdV < 1e-5) continue;
      worldZs.push(cam.position.clone().addScaledVector(dv, o.depthZ / fwdV).z);
    }
    if (worldZs.length >= 2) {
      const split = Math.max(1, Math.floor(worldZs.length / 3));
      const zFirst = _median(worldZs.slice(0, split));
      const zLast  = _median(worldZs.slice(worldZs.length - split));
      const arcDuration = obs[obs.length - 1].t - obs[0].t;
      if (arcDuration > 1e-3) initVz = (zLast - zFirst) / arcDuration;
      initZ = _median(worldZs);
    } else if (worldZs.length === 1) {
      initZ = worldZs[0];
    }
  }

  const VSCALE = 10;
  const hw = videoWidth  / 2;
  const hh = videoHeight / 2;

  /** Build an objective function for a given active observation set. */
  const makeObjective = (activeObs: ArcObsEx[]) => {
    const ts = activeObs.map((o) => o.t - t0);
    return (p: number[]) => {
      const [x0, y0, z0, svx, svy, svz] = p;
      const vx = svx * VSCALE, vy = svy * VSCALE, vz = svz * VSCALE;
      let err = 0;
      for (let i = 0; i < activeObs.length; i++) {
        const t  = ts[i];
        const wx = x0 + vx * t;
        const wy = y0 + vy * t - 0.5 * GRAVITY * t * t;
        const wz = z0 + vz * t;
        // Soft physical bounds
        if (wy < -0.5 || wy > MAX_BALL_HEIGHT_M + 2) err += 20;
        if (Math.abs(wx) > COURT_HALF_W + 8 || Math.abs(wz) > COURT_HALF_L + 12) err += 10;
        // Confidence-weighted reprojection error
        const [pu, pv] = projectToNDC(cam, wx, wy, wz);
        const du = pu - activeObs[i].u;
        const dv = pv - activeObs[i].v;
        err += Math.max(0.1, activeObs[i].conf) * (du * du + dv * dv);
        // Soft camera-depth prior (ray depth, not world-z)
        if (activeObs[i].depthZ != null && activeObs[i].depthZ! > 0.5) {
          const camDepth = (wx - cpx) * cfx + (wy - cpy) * cfy + (wz - cpz) * cfz;
          if (camDepth > 0.1) {
            const dRel = (camDepth - activeObs[i].depthZ!) / activeObs[i].depthZ!;
            err += DEPTH_PRIOR_WEIGHT * dRel * dRel;
          }
        }
      }
      return err;
    };
  };

  /** Multi-seed fit; if prevSol is given, performs a single warm-start refinement. */
  const fitArc = (activeObs: ArcObsEx[], prevSol?: number[]): number[] => {
    const obj = makeObjective(activeObs);
    let best: number[] | null = null;
    let bestErr = Infinity;
    const tryP = (p0: number[]) => {
      const sol = nelderMead(p0, obj, 1500, 1e-10);
      const e = obj(sol);
      if (e < bestErr) { bestErr = e; best = sol; }
    };
    if (prevSol) {
      tryP(prevSol);
    } else {
      for (const [sivx, sivy, sivz] of [
        [0,  0.4, -0.5],
        [0,  0.4,  0.5],
        [0, -0.5, -0.5],
        [initVx / VSCALE, initVy / VSCALE, initVz / VSCALE],
      ] as [number, number, number][]) {
        tryP([initX, initY, initZ, sivx, sivy, sivz]);
      }
    }
    return best ?? [initX, initY, initZ, 0, 0.4, -0.5];
  };

  /** Compute per-observation pixel reprojection errors for a given solution. */
  const computeErrsPx = (sol: number[], activeObs: ArcObsEx[]): number[] => {
    const vx = sol[3] * VSCALE, vy = sol[4] * VSCALE, vz = sol[5] * VSCALE;
    return activeObs.map((o) => {
      const t = o.t - t0;
      const wx = sol[0] + vx * t;
      const wy = sol[1] + vy * t - 0.5 * GRAVITY * t * t;
      const wz = sol[2] + vz * t;
      const [pu, pv] = projectToNDC(cam, wx, wy, wz);
      return Math.hypot((pu - o.u) * hw, (pv - o.v) * hh);
    });
  };

  // ── Iterative outlier rejection ───────────────────────────────────────────
  let activeObs = obs.slice();
  let currentSol = fitArc(activeObs);
  const outlierFrames = new Set<number>();

  for (let round = 0; round < 3; round++) {
    if (activeObs.length < MIN_ARC_FRAMES) break;
    const errs = computeErrsPx(currentSol, activeObs);
    const newInliers = activeObs.filter((_, i) => errs[i] <= OUTLIER_ERR_PX);
    const newOutliers = activeObs.filter((_, i) => errs[i] > OUTLIER_ERR_PX);
    if (newOutliers.length === 0) break;
    if (newInliers.length < MIN_ARC_INLIERS) break; // don't shrink below minimum
    for (const o of newOutliers) outlierFrames.add(o.fi);
    activeObs = newInliers;
    currentSol = fitArc(activeObs, currentSol); // warm-start refit on inliers
  }

  // ── Build per-observation results for ALL original observations ───────────
  const [x0, y0, z0] = currentSol;
  const vx = currentSol[3] * VSCALE;
  const vy = currentSol[4] * VSCALE;
  const vz = currentSol[5] * VSCALE;

  const observations: ArcSolveObservation[] = obs.map((o, i) => {
    const t  = o.t - t0;
    const wx = x0 + vx * t;
    const wy = y0 + vy * t - 0.5 * GRAVITY * t * t; // NOT clamped
    const wz = z0 + vz * t;
    const [projU, projV] = projectToNDC(cam, wx, wy, wz);
    const errPx = Math.hypot((projU - o.u) * hw, (projV - o.v) * hh);
    const cameraDepthFitted = (wx - cpx) * cfx + (wy - cpy) * cfy + (wz - cpz) * cfz;
    const depthPriorError = (o.depthZ != null && o.depthZ > 0.5 && cameraDepthFitted > 0.1)
      ? (cameraDepthFitted - o.depthZ) / o.depthZ
      : null;
    let speedMps = 0;
    if (i > 0) {
      const pt = obs[i - 1].t - t0;
      const px = x0 + vx * pt, py = y0 + vy * pt - 0.5 * GRAVITY * pt * pt, pz = z0 + vz * pt;
      const dt = t - pt;
      if (dt > 1e-6) speedMps = Math.hypot(wx - px, wy - py, wz - pz) / dt;
    }
    return { fi: o.fi, pos: [wx, wy, wz], projU, projV, errPx, speedMps,
             cameraDepthFitted, depthPriorError, isOutlier: outlierFrames.has(o.fi), obsRef: o };
  });

  // ── Physical arc validation (inliers only) ────────────────────────────────
  const inlierObs = observations.filter((o) => !o.isOutlier);

  if (inlierObs.length < MIN_ARC_INLIERS) {
    return {
      arcParams: [x0, y0, z0, vx, vy, vz], t0,
      accepted: false, rejectReason: `too-few-inliers (${inlierObs.length})`,
      observations,
    };
  }

  const inlierErrs  = inlierObs.map((o) => o.errPx);
  const avgErrPx    = inlierErrs.reduce((s, e) => s + e, 0) / inlierErrs.length;
  const maxErrPx    = Math.max(...inlierErrs);
  const maxSpeed    = Math.max(0, ...observations.map((o) => o.speedMps));
  const allY        = observations.map((o) => o.pos[1]);
  const allZ        = observations.map((o) => o.pos[2]);
  const minY        = Math.min(...allY);
  const maxY        = Math.max(...allY);
  const hardZViol   = allZ.some((z) => Math.abs(z) > COURT_Z_HARD_M);
  const softZFrac   = allZ.filter((z) => Math.abs(z) > COURT_Z_SOFT_M).length / allZ.length;

  let rejectReason: string | null = null;
  if      (minY < MIN_FIT_Y_M)   rejectReason = `y-below-ground (min=${minY.toFixed(2)}m)`;
  else if (maxY > MAX_FIT_Y_M)   rejectReason = `y-too-high (max=${maxY.toFixed(2)}m)`;
  else if (hardZViol)             rejectReason = `z-out-of-court (max|z|=${Math.max(...allZ.map(Math.abs)).toFixed(1)}m > ${COURT_Z_HARD_M}m)`;
  else if (softZFrac > 0.2)       rejectReason = `z-soft-exceeded (${(softZFrac * 100).toFixed(0)}% > ±${COURT_Z_SOFT_M}m)`;
  else if (maxSpeed > MAX_BALL_SPEED_MPS) rejectReason = `speed-too-high (${maxSpeed.toFixed(1)} m/s)`;
  else if (avgErrPx > MAX_ARC_AVG_ERR_PX) rejectReason = `avg-err-too-high (${avgErrPx.toFixed(1)}px)`;
  else if (maxErrPx > MAX_ARC_MAX_ERR_PX) rejectReason = `max-err-too-high (${maxErrPx.toFixed(1)}px)`;

  const accepted = rejectReason === null;
  console.log(
    `[physicsReproj] arc[${arcId}] n=${obs.length}→inliers=${inlierObs.length}` +
    ` sol=[${x0.toFixed(2)},${y0.toFixed(2)},${z0.toFixed(2)}]` +
    ` v=[${vx.toFixed(1)},${vy.toFixed(1)},${vz.toFixed(1)}]` +
    ` avgErr=${avgErrPx.toFixed(1)}px maxErr=${maxErrPx.toFixed(1)}px` +
    ` yRange=[${minY.toFixed(2)},${maxY.toFixed(2)}]` +
    ` maxSpeed=${maxSpeed.toFixed(1)}m/s` +
    ` ${accepted ? "ACCEPTED" : `REJECTED: ${rejectReason}`}`,
  );

  return { arcParams: [x0, y0, z0, vx, vy, vz], t0, accepted, rejectReason, observations };
}

/**
 * Helpers for stats, summary logging, and the main trajectory solver.
 *
 * Priority in per-frame mode (useBallisticSolver = false):
 *   1. bbox-depth  — known-size monocular depth (primary for airborne ball)
 *   2. bbox-bottom-ray — court-plane fallback
 *   3. homography-ray  — court-corner homography (only when courtCorners provided)
 *   4. ground-intersection — last resort
 *
 * The Depth Anything V2 relative depth (ball_depth_estimate) is intentionally
 * NOT used here — it is scale-ambiguous and reserved for debug visualisation.
 */

function _median(arr: number[]): number {
  if (arr.length === 0) return Number.NaN;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Apply a median filter to an array with radius `r` (window = 2r+1). Null values are ignored. */
function _medianSmooth(values: (number | null)[], r: number): (number | null)[] {
  return values.map((_, i) => {
    const win = values
      .slice(Math.max(0, i - r), i + r + 1)
      .filter((v): v is number => v != null);
    return win.length > 0 ? _median(win) : null;
  });
}

function _printConsoleSummary(
  debugRecords: BallFrameDebug[],
  positions: Map<number, [number, number, number]>,
): void {
  const withBbox   = debugRecords.filter((d) => d.bbox != null).length;
  const bboxDepthN = debugRecords.filter((d) => d.method === "bbox-depth").length;
  const rejected   = debugRecords.filter((d) => !d.accepted);

  const posArr = [...positions.values()];
  const xs = posArr.map(([x]) => x);
  const ys = posArr.map(([, y]) => y);
  const zs = posArr.map(([, , z]) => z);

  const zFW    = debugRecords.flatMap((d) => d.zFromWidth    == null ? [] : [d.zFromWidth]);
  const zFH    = debugRecords.flatMap((d) => d.zFromHeight   == null ? [] : [d.zFromHeight]);
  const bboxWs = debugRecords.flatMap((d) => d.bboxWidthNorm  == null ? [] : [d.bboxWidthNorm]);
  const bboxHs = debugRecords.flatMap((d) => d.bboxHeightNorm == null ? [] : [d.bboxHeightNorm]);

  // Pixel-space stats
  const bboxWpx = debugRecords.flatMap((d) => d.bboxWidthPx  == null ? [] : [d.bboxWidthPx]);
  const bboxHpx = debugRecords.flatMap((d) => d.bboxHeightPx == null ? [] : [d.bboxHeightPx]);
  const fxArr   = debugRecords.flatMap((d) => d.fx > 0 ? [d.fx] : []);
  const fyArr   = debugRecords.flatMap((d) => d.fy > 0 ? [d.fy] : []);
  const confArr = debugRecords.flatMap((d) => d.confidence == null ? [] : [d.confidence]);

  // Bbox pixel aspect ratio outliers (W/H outside 0.6–1.6)
  const aspectOutliers = debugRecords.filter((d) => {
    if (d.bboxWidthPx == null || d.bboxHeightPx == null || d.bboxHeightPx < 1e-3) return false;
    const r = d.bboxWidthPx / d.bboxHeightPx;
    return r < 0.6 || r > 1.6;
  }).length;

  // Confidence histogram (<0.4, 0.4–0.6, 0.6–0.8, ≥0.8)
  const cb = [0, 0, 0, 0];
  for (const c of confArr) {
    if      (c < 0.4) cb[0]++;
    else if (c < 0.6) cb[1]++;
    else if (c < 0.8) cb[2]++;
    else              cb[3]++;
  }
  const confHist = `[<0.4]:${cb[0]}  [0.4–0.6]:${cb[1]}  [0.6–0.8]:${cb[2]}  [≥0.8]:${cb[3]}`;

  const rejCounts: Record<string, number> = {};
  for (const d of rejected) {
    const r = d.rejectedReason ?? "unknown";
    rejCounts[r] = (rejCounts[r] ?? 0) + 1;
  }
  const topRej = Object.entries(rejCounts).sort((a, b) => b[1] - a[1])[0];

  const fmt2 = (v: number) => v.toFixed(2);
  const fmtRange = (a: number[]) =>
    a.length ? `${fmt2(Math.min(...a))} / ${fmt2(_median(a))} / ${fmt2(Math.max(...a))}` : "n/a";

  console.log(
    "[ballRecon] ─── RECONSTRUCTION SUMMARY ───────────────────────────\n" +
    `  detected (non-interp)   : ${debugRecords.length}\n` +
    `  with YOLO bbox          : ${withBbox}\n` +
    `  using bbox-depth        : ${bboxDepthN}\n` +
    `  solved 3D               : ${positions.size}\n` +
    `  rejected                : ${rejected.length}` +
    (topRej ? `  (top: ${topRej[0]} ×${topRej[1]})` : "") + "\n" +
    `  x  min/med/max (m)      : ${fmtRange(xs)}\n` +
    `  y  min/med/max (m)      : ${fmtRange(ys)}\n` +
    `  z  min/med/max (m)      : ${fmtRange(zs)}\n` +
    `  zFromWidth  min/med/max : ${fmtRange(zFW)}\n` +
    `  zFromHeight min/med/max : ${fmtRange(zFH)}\n` +
    `  bboxWidthPx   med (px)  : ${bboxWpx.length ? _median(bboxWpx).toFixed(1) : "n/a"}\n` +
    `  bboxHeightPx  med (px)  : ${bboxHpx.length ? _median(bboxHpx).toFixed(1) : "n/a"}\n` +
    `  fx median               : ${fxArr.length ? _median(fxArr).toFixed(0) : "n/a"}\n` +
    `  fy median               : ${fyArr.length ? _median(fyArr).toFixed(0) : "n/a"}\n` +
    `  bbox aspect outliers    : ${aspectOutliers} / ${withBbox} (px W/H outside 0.6–1.6)\n` +
    `  confidence histogram    : ${confHist}\n` +
    `  avg bbox width  (norm)  : ${bboxWs.length ? (bboxWs.reduce((a, b) => a + b, 0) / bboxWs.length).toFixed(4) : "n/a"}\n` +
    `  avg bbox height (norm)  : ${bboxHs.length ? (bboxHs.reduce((a, b) => a + b, 0) / bboxHs.length).toFixed(4) : "n/a"}\n` +
    "[ballRecon] ────────────────────────────────────────────────────────"
  );
}

function _computeDetectionStats(
  frames: Pick<FrameData, "ball_img_pos" | "ball_interpolated" | "ball_conf">[],
): { detectedFrames: number; avgConf: number } {
  let confSum = 0;
  let confCount = 0;
  let detectedFrames = 0;
  for (const f of frames) {
    if (f.ball_img_pos && !f.ball_interpolated) {
      detectedFrames++;
      if (f.ball_conf != null && f.ball_conf > 0) {
        confSum += f.ball_conf;
        confCount++;
      }
    }
  }
  return { detectedFrames, avgConf: confCount > 0 ? confSum / confCount : 0 };
}

function _processArcs(
  arcs: ArcObs[][],
  cam: THREE.PerspectiveCamera,
): {
  positions: Map<number, [number, number, number]>;
  methods: Map<number, BallReconstructionMethod>;
  rejectedFrames: number;
} {
  const positions = new Map<number, [number, number, number]>();
  const methods = new Map<number, BallReconstructionMethod>();
  let rejectedFrames = 0;
  for (const arc of arcs) {
    if (arc.length < MIN_ARC_FRAMES) continue;
    const solved = solveArc(arc, cam);
    if (solved) {
      for (const { fi, pos } of solved) {
        positions.set(fi, pos);
        methods.set(fi, "ballistic-arc");
      }
    } else {
      rejectedFrames += arc.length;
    }
  }
  return { positions, methods, rejectedFrames };
}

export function solveBallTrajectories(
  frames: Pick<FrameData, "frame" | "timestamp" | "ball_img_pos" | "ball_bbox" | "ball_interpolated" | "ball_conf">[],
  calibration: CalibrationResult,
  aspect: number,
  options?: BallReconstructionOptions,
): BallTrajectoryResult {
  const { detectedFrames, avgConf } = _computeDetectionStats(frames);
  const positions = new Map<number, [number, number, number]>();
  const methods   = new Map<number, BallReconstructionMethod>();
  const debugRecords: BallFrameDebug[] = [];
  const ballDiameterM    = options?.ballDiameterM    ?? DEFAULT_BALL_DIAMETER_M;
  const useBallisticSolver = options?.useBallisticSolver ?? false;

  if (aspect <= 0) {
    return { positions, methods, debugRecords, stats: { detectedFrames, solved3dFrames: 0, rejectedFrames: 0, methodCounts: {}, avgConf } };
  }

  // ── Per-frame / physics-reprojection modes (useBallisticSolver = false) ─────
  if (!useBallisticSolver) {
    const vH             = options?.videoHeight       ?? 1080;
    const vW             = options?.videoWidth        ?? Math.round(vH * aspect);
    const diamMode       = options?.diameterMode      ?? "min-side";
    const debugMode      = options?.debugMode         ?? false;
    const trajectoryMode = options?.trajectoryMode    ?? "raw-bbox-depth";
    const minBallConf    = options?.minBallConfidence ?? 0.6;
    const effectiveFovY  = (options?.useCalibrationFov ?? true)
      ? calibration.fov
      : (options?.manualFovYDeg ?? calibration.fov);

    const halfFovYRad = (effectiveFovY * Math.PI) / 360;
    const halfFovXRad = Math.atan(Math.tan(halfFovYRad) * aspect);
    const fyPx        = vH / (2 * Math.tan(halfFovYRad));
    const fxPx        = vW / (2 * Math.tan(halfFovXRad));
    const fovXDeg     = (halfFovXRad * 360) / Math.PI;

    const depthCtx: DepthContext = {
      ballDiameterM, aspect, videoWidth: vW, videoHeight: vH,
      diameterMode: diamMode, fovYDeg: effectiveFovY,
    };
    const cam = createCamera(calibration, aspect, effectiveFovY);

    // ── Pre-compute per-detected-frame raw / smoothed diameter arrays ────────
    const detectedList = frames.filter(
      (f): f is typeof f & { ball_img_pos: [number, number] } =>
        f.ball_img_pos != null && !f.ball_interpolated,
    );

    const rawDiamPx: (number | null)[] = detectedList.map((f) => {
      if (!f.ball_bbox) return null;
      const bw = Math.abs(f.ball_bbox[2] - f.ball_bbox[0]) * vW;
      const bh = Math.abs(f.ball_bbox[3] - f.ball_bbox[1]) * vH;
      return Math.min(bw, bh);
    });

    const smoothedDiamPx   = _medianSmooth(rawDiamPx, 3);
    const rawZFromDiam     = rawDiamPx.map((d) =>
      d != null && d > 1e-3 ? fyPx * ballDiameterM / d : null);
    const smoothedZFromDiam = smoothedDiamPx.map((d) =>
      d != null && d > 1e-3 ? fyPx * ballDiameterM / d : null);

    let loggedCount = 0;

    // ── Per-frame reconstruction loop ────────────────────────────────────────
    for (let fi = 0; fi < detectedList.length; fi++) {
      const f = detectedList[fi];
      const { origin, dir } = rayFromImagePos(cam, f.ball_img_pos);

      const depthDetail = f.ball_bbox
        ? bboxDepthCamDetail(f.ball_bbox, effectiveFovY, ballDiameterM, aspect, vW, vH, diamMode)
        : null;

      // Raw bbox-depth estimate (plausibility skipped — recorded for debug)
      const rawEstimate = tryBboxDepth(f, cam, origin, dir, depthCtx, true);
      const plausible   = rawEstimate != null
        && isPlausibleInCourtVolume(new THREE.Vector3(...rawEstimate.position));

      const aboveConf = (f.ball_conf ?? 0) >= minBallConf;

      let estimate: BallWorldEstimate | null = null;
      let accepted = false;
      let rejectedReason: string | null = null;

      if (trajectoryMode === "raw-bbox-depth" || trajectoryMode === "smoothed-bbox-depth") {
        if (rawEstimate == null) {
          rejectedReason = f.ball_bbox ? "ray-angle-too-small" : "no-bbox";
        } else if (!aboveConf) {
          rejectedReason = `conf<${minBallConf} (${(f.ball_conf ?? 0).toFixed(2)})`;
        } else {
          const zOverride = trajectoryMode === "smoothed-bbox-depth"
            ? smoothedZFromDiam[fi]
            : null;

          if (zOverride != null) {
            // Re-place along ray at smoothed depth
            const invQ    = cam.quaternion.clone().invert();
            const forward = -dir.clone().applyQuaternion(invQ).z;
            if (forward > 1e-5) {
              const pos = origin.clone().addScaledVector(dir, zOverride / forward);
              pos.y = Math.max(HALF_BALL, pos.y);
              if (debugMode || isPlausibleInCourtVolume(pos)) {
                estimate = { position: [pos.x, pos.y, pos.z], ground: [pos.x, 0, pos.z], height: pos.y, method: "bbox-depth" };
                accepted = true;
              } else {
                rejectedReason = "plausibility-check-failed";
              }
            } else {
              rejectedReason = "ray-angle-too-small";
            }
          } else {
            if (debugMode || plausible) {
              estimate = rawEstimate;
              accepted = true;
            } else {
              rejectedReason = "plausibility-check-failed";
            }
          }
        }
      }
      // physics-reprojection: positions filled in post-loop arc solver

      const bw = f.ball_bbox ? Math.abs(f.ball_bbox[2] - f.ball_bbox[0]) : null;
      const bh = f.ball_bbox ? Math.abs(f.ball_bbox[3] - f.ball_bbox[1]) : null;

      const rec: BallFrameDebug = {
        frame:              f.frame,
        u:                  f.ball_img_pos[0] * 2 - 1,
        v:                  1 - f.ball_img_pos[1] * 2,
        videoWidth:         vW,
        videoHeight:        vH,
        bbox:               f.ball_bbox ?? null,
        bboxFormatAssumed:  "x1y1x2y2-norm",
        bboxWidthNorm:      bw,
        bboxHeightNorm:     bh,
        bboxWidthPx:        bw == null ? null : bw * vW,
        bboxHeightPx:       bh == null ? null : bh * vH,
        confidence:         f.ball_conf ?? null,
        fovY:               effectiveFovY,
        fovX:               fovXDeg,
        fx:                 depthDetail?.fx ?? fxPx,
        fy:                 depthDetail?.fy ?? fyPx,
        ballDiameterM,
        zFromWidth:         depthDetail?.zFromWidth   ?? null,
        zFromHeight:        depthDetail?.zFromHeight  ?? null,
        zFromMinSide:       depthDetail?.zFromMinSide ?? null,
        zUsed:              depthDetail?.zUsed        ?? null,
        diameterPxRaw:      rawDiamPx[fi],
        diameterPxSmoothed: smoothedDiamPx[fi],
        zUsedSmoothed:      smoothedZFromDiam[fi],
        method:             estimate?.method ?? null,
        worldPositionRaw:   rawEstimate?.position ?? null,
        worldPosition:      estimate?.position   ?? null,
        accepted,
        rejectedReason,
        cameraPosition:     calibration.position,
        usedInFit:          trajectoryMode !== "physics-reprojection"
          ? accepted
          : false, // updated below after arc solver
        projectedU:         null,
        projectedV:         null,
        reprojectionErrorPx: null,
        fittedPosition:     null,
      };
      debugRecords.push(rec);

      if (loggedCount < 50) {
        loggedCount++;
        const posStr = rec.worldPositionRaw
          ? `[${rec.worldPositionRaw.map((n) => n.toFixed(2)).join(",")}]`
          : "null";
        console.log(
          `[ballRecon] f=${f.frame}` +
          ` bboxPx=${rec.bboxWidthPx?.toFixed(1) ?? "n/a"}×${rec.bboxHeightPx?.toFixed(1) ?? "n/a"}` +
          ` fy=${rec.fy.toFixed(0)} fovY=${rec.fovY.toFixed(2)}°` +
          ` zH=${rec.zFromHeight?.toFixed(2) ?? "n/a"}` +
          ` zW=${rec.zFromWidth?.toFixed(2) ?? "n/a"}` +
          ` zSmoothed=${rec.zUsedSmoothed?.toFixed(2) ?? "n/a"}` +
          ` zUsed=${rec.zUsed?.toFixed(2) ?? "n/a"}` +
          ` conf=${(f.ball_conf ?? 0).toFixed(2)}` +
          ` acc=${rec.accepted ? "Y" : "N"}` +
          ` posRaw=${posStr}` +
          (rec.rejectedReason ? ` ⚠ ${rec.rejectedReason}` : ""),
        );
      }

      if (accepted && estimate) {
        positions.set(f.frame, estimate.position);
        methods.set(f.frame, estimate.method);
      }
    }

    // ── Physics-reprojection arc solver (post-loop) ──────────────────────────
    const physicsDebug: PhysicsReprojectionDebug[] = [];

    if (trajectoryMode === "physics-reprojection") {
      // Use a lower confidence threshold for 2D fitting observations.
      const minFitConf = 0.25;
      const obsEx: ArcObsEx[] = detectedList
        .map((f, i) => ({
          t:      f.timestamp,
          u:      f.ball_img_pos[0] * 2 - 1,
          v:      1 - f.ball_img_pos[1] * 2,
          fi:     f.frame,
          conf:   f.ball_conf ?? 0,
          depthZ: smoothedZFromDiam[i],
        }))
        .filter((o) => o.conf >= minFitConf);

      const arcs = splitArcsEx(obsEx);

      console.log(
        `[physicsReproj] mode=physics-reprojection` +
        ` obs=${obsEx.length} arcs=${arcs.length}` +
        ` (minFitConf=${minFitConf}, minBallConf=${minBallConf})`,
      );
      for (let ai = 0; ai < arcs.length; ai++) {
        const a = arcs[ai];
        const durS = a[a.length - 1].t - a[0].t;
        console.log(
          `[physicsReproj]   arc[${ai}] frames ${a[0].fi}–${a[a.length - 1].fi}` +
          ` n=${a.length} dur=${durS.toFixed(2)}s`,
        );
      }

      let totalFittedArcs = 0;
      const allErrPx: number[] = [];

      for (let ai = 0; ai < arcs.length; ai++) {
        const arc = arcs[ai];
        const solved = solveArcReprojection(arc, cam, vW, vH, ai);

        if (!solved) {
          // Arc has fewer than MIN_ARC_FRAMES — add minimal debug records
          for (const o of arc) {
            const origIdx = detectedList.findIndex((df) => df.frame === o.fi);
            physicsDebug.push({
              frame: o.fi,
              observedU: o.u, observedV: o.v,
              projectedU: null, projectedV: null, reprojectionErrorPx: null,
              fittedX: null, fittedY: null, fittedZ: null,
              rawBboxDepthZ:      origIdx >= 0 ? rawZFromDiam[origIdx]      : null,
              smoothedBboxDepthZ: origIdx >= 0 ? smoothedZFromDiam[origIdx] : null,
              confidence: o.conf, usedInFit: false,
              arcId: ai, arcAccepted: false, arcRejectReason: "too-few-observations",
              speedMps: null, cameraDepthFitted: null, depthPriorError: null, isOutlier: false,
            });
          }
          continue;
        }

        const { arcParams, t0: arcT0, accepted, rejectReason, observations: solvedObs } = solved;
        const [ax0, ay0, az0, avx, avy, avz] = arcParams;

        if (accepted) {
          totalFittedArcs++;
          const arcStartT = arc[0].t;
          const arcEndT   = arc[arc.length - 1].t;
          let filledCount = 0;
          for (const fr of frames) {
            if (fr.timestamp < arcStartT - 1e-6 || fr.timestamp > arcEndT + 1e-6) continue;
            if (positions.has(fr.frame)) continue;
            const t  = fr.timestamp - arcT0;
            const wx = ax0 + avx * t;
            const wy = ay0 + avy * t - 0.5 * GRAVITY * t * t; // NOT clamped
            const wz = az0 + avz * t;
            positions.set(fr.frame, [wx, wy, wz]);
            methods.set(fr.frame, "physics-reprojection");
            filledCount++;
          }
          console.log(
            `[physicsReproj] arc[${ai}] ACCEPTED filled=${filledCount} frames`,
          );
        }

        // Emit physicsDebug for ALL observations (accepted + rejected, inliers + outliers)
        for (const { fi, pos, projU, projV, errPx, speedMps, cameraDepthFitted,
                     depthPriorError, isOutlier, obsRef } of solvedObs) {
          if (!isOutlier) allErrPx.push(errPx);

          if (accepted && !isOutlier) {
            const rec = debugRecords.find((d) => d.frame === fi);
            if (rec) {
              rec.fittedPosition      = pos;
              rec.projectedU          = projU;
              rec.projectedV          = projV;
              rec.reprojectionErrorPx = errPx;
              rec.worldPosition       = pos;
              rec.accepted            = true;
              rec.rejectedReason      = null;
              rec.method              = "physics-reprojection";
              rec.usedInFit           = true;
            }
          }

          const origIdx = detectedList.findIndex((df) => df.frame === fi);
          physicsDebug.push({
            frame:               fi,
            observedU:           obsRef.u,
            observedV:           obsRef.v,
            projectedU:          projU,
            projectedV:          projV,
            reprojectionErrorPx: errPx,
            fittedX:             pos[0],
            fittedY:             pos[1],
            fittedZ:             pos[2],
            rawBboxDepthZ:       origIdx >= 0 ? rawZFromDiam[origIdx]      : null,
            smoothedBboxDepthZ:  origIdx >= 0 ? smoothedZFromDiam[origIdx] : null,
            confidence:          obsRef.conf,
            usedInFit:           accepted && !isOutlier,
            arcId:               ai,
            arcAccepted:         accepted,
            arcRejectReason:     accepted ? null : rejectReason,
            speedMps,
            cameraDepthFitted,
            depthPriorError,
            isOutlier,
          });
        }
      }

      // Emit physicsDebug records for detected frames NOT covered by any arc
      // (low-confidence detections filtered out before arc splitting)
      const processedFrames = new Set(physicsDebug.map((d) => d.frame));
      for (let i = 0; i < detectedList.length; i++) {
        const f = detectedList[i];
        if (processedFrames.has(f.frame)) continue;
        physicsDebug.push({
          frame:               f.frame,
          observedU:           f.ball_img_pos[0] * 2 - 1,
          observedV:           1 - f.ball_img_pos[1] * 2,
          projectedU:          null,
          projectedV:          null,
          reprojectionErrorPx: null,
          fittedX:             null,
          fittedY:             null,
          fittedZ:             null,
          rawBboxDepthZ:       rawZFromDiam[i],
          smoothedBboxDepthZ:  smoothedZFromDiam[i],
          confidence:          f.ball_conf ?? null,
          usedInFit:           false,
          arcId:               null,
          arcAccepted:         null,
          arcRejectReason:     "not-in-arc",
          speedMps:            null,
          cameraDepthFitted:   null,
          depthPriorError:     null,
          isOutlier:           false,
        });
      }

      const avgErrPxAll = allErrPx.length > 0
        ? allErrPx.reduce((s, e) => s + e, 0) / allErrPx.length
        : null;
      console.log(
        `[physicsReproj] summary: ${totalFittedArcs}/${arcs.length} arcs accepted,` +
        ` ${positions.size} total frames,` +
        ` avgErrPx=${avgErrPxAll != null ? avgErrPxAll.toFixed(1) + "px" : "n/a"}`,
      );

      if (totalFittedArcs === 0) {
        console.warn(
          `[physicsReproj] physics-reprojection failed for all ${arcs.length} arc(s);` +
          ` displaying raw bbox-depth debug points only.`,
        );
      }
    }

    const solved3dFrames = positions.size;
    const rejectedFrames = debugRecords.filter((d) => !d.accepted).length;
    const methodCounts: Partial<Record<BallReconstructionMethod, number>> = {};
    for (const m of methods.values()) methodCounts[m] = (methodCounts[m] ?? 0) + 1;

    _printConsoleSummary(debugRecords, positions);

    return {
      positions, methods, debugRecords,
      physicsDebug: physicsDebug.length > 0 ? physicsDebug : undefined,
      stats: { detectedFrames, solved3dFrames, rejectedFrames, methodCounts, avgConf },
    };
  }

  // ── Ballistic arc solver mode (useBallisticSolver = true) ───────────────────
  const cam = createCamera(calibration, aspect);
  const allObs = buildObservations(frames);
  if (allObs.length === 0) {
    return { positions, methods, debugRecords, stats: { detectedFrames, solved3dFrames: 0, rejectedFrames: 0, methodCounts: {}, avgConf } };
  }

  const { positions: arcPos, methods: arcMethods, rejectedFrames } = _processArcs(splitIntoArcs(allObs), cam);
  for (const [k, v] of arcPos)     positions.set(k, v);
  for (const [k, v] of arcMethods) methods.set(k, v);

  const solved3dFrames = positions.size;
  const methodCounts: Partial<Record<BallReconstructionMethod, number>> = {};
  if (solved3dFrames > 0) methodCounts["ballistic-arc"] = solved3dFrames;

  return {
    positions, methods, debugRecords,
    stats: { detectedFrames, solved3dFrames, rejectedFrames, methodCounts, avgConf },
  };
}
