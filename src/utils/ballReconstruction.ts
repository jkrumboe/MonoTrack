import * as THREE from "three";
import type { FrameData, BallReconstructionMethod } from "../hooks/useAnalysis";
import { aimCamera, type CalibrationResult } from "./cameraSolver";
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
  | "physics-reprojection"  // ballistic arc fit minimising 2D reprojection error
  | "short-arc-debug";      // single arc over a manually selected frame range

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
  /**
   * First frame index (inclusive) to use in short-arc-debug mode.
   * Only detections with frame >= arcStartFrame are used for arc fitting.
   */
  arcStartFrame?: number;
  /**
   * Last frame index (inclusive) to use in short-arc-debug mode.
   * Only detections with frame <= arcEndFrame are used for arc fitting.
   */
  arcEndFrame?: number;
  /**
   * When true, zeros out the bbox-depth soft prior in the arc solver so the
   * reprojection objective is driven purely by 2D observations.
   * Default: false.
   */
  disableBboxDepthPrior?: boolean;
  /**
   * Overrides for the rally solver's tuned numbers (physics-reprojection
   * mode). Unset fields keep DEFAULT_RALLY_PARAMS; `rules` merges per touch.
   */
  rallyParams?: Partial<Omit<RallyParams, "rules">> & { rules?: Partial<RallyParams["rules"]> };
  /**
   * Touch corrections made by hand (the app's Inspect & Edit view).
   * `touchAt`: frames where a player touched the ball; the track is split
   * there and the split kept whatever the velocity jump. `noTouchAt`: frames
   * where a found touch was removed; a split within a few frames of one is
   * merged back into a single flight. `pins`: where in 3D (m) the ball was
   * at a touch, placed by hand; the touch's knot is held there and the
   * flights either side fit to it. A pin is also a `touchAt`.
   */
  touchEdits?: { touchAt?: number[]; noTouchAt?: number[]; pins?: { frame: number; pos: [number, number, number] }[] };
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
  /** Physics-reprojection / short-arc-debug mode diagnostic. */
  physicsDebug?: PhysicsReprojectionDebug[];
  /** Whether the single arc fitted in short-arc-debug mode was accepted. */
  shortArcAccepted?: boolean;
  /** Rejection reason for the short-arc-debug arc, or null when accepted. */
  shortArcRejectReason?: string | null;
  /** Player touches found by the rally solver (physics-reprojection mode). */
  touches?: BallTouch[];
  /** Every candidate touch split and the decision on it (diagnostics). */
  touchSplits?: TouchSplitDecision[];
}

/** One touch of the ball in a rally, as placed by the rally solver. */
export interface BallTouch {
  frame: number;
  position: [number, number, number];
  /** "toss" | "serve" | "reception" | "set" | "attack" | "block" | "dig" | "touch". */
  label: string;
  /** Court (x, z) of the players who could have made the touch, after the rules. */
  candidates?: [number, number][];
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
  aimCamera(cam, calibration);
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

/** A player on the court for display: feet position and height (m). */
export interface CourtPlayer {
  x: number;
  z: number;
  height: number;
  /**
   * Arm joints in world metres when the pose model saw them: shoulder, elbow,
   * wrist for the player's left and right arm (null where not confident).
   * Lifted onto the vertical plane through the feet that faces the camera, so
   * depth along the view ray is a guess but the arm's shape on screen is kept.
   */
  arms?: { left: ArmJoints | null; right: ArmJoints | null };
}

/** Shoulder, elbow, wrist of one arm, world metres. */
export type ArmJoints = [THREE.Vector3, THREE.Vector3, THREE.Vector3];

const ARM_MIN_CONF = 0.3;

/**
 * Players of one frame on the court: feet from the bottom of each person box
 * through the floor, height from where the ray to the top of the box passes
 * over the feet. Uses the reference-frame boxes for a moving camera (the
 * calibration's image). People well off the court (bench, crowd, officials
 * at the scorer's table) are left out.
 */
export function playersOnCourt(
  frame: Pick<FrameData, "players" | "players_all_ref" | "players_arms" | "players_arms_ref">,
  calibration: CalibrationResult,
  aspect: number,
): CourtPlayer[] {
  const useRef = !!frame.players_all_ref;
  const boxes = useRef ? frame.players_all_ref : frame.players;
  const allArms = useRef ? frame.players_arms_ref : frame.players_arms;
  if (!boxes?.length || aspect <= 0) return [];
  const cam = createCamera(calibration, aspect);
  const out: CourtPlayer[] = [];
  for (const [bi, [x1, y1, x2, y2]] of boxes.entries()) {
    const cx = (x1 + x2) / 2;
    const feet = rayFromImagePos(cam, [cx, y2]);
    if (feet.dir.y > -1e-6) continue;
    const t = -feet.origin.y / feet.dir.y;
    const x = feet.origin.x + feet.dir.x * t, z = feet.origin.z + feet.dir.z * t;
    if (Math.abs(x) > COURT_HALF_W + 3 || Math.abs(z) > COURT_HALF_L + 6) continue;
    // Point of the head ray nearest the vertical line over the feet.
    const head = rayFromImagePos(cam, [cx, y1]);
    const hd = new THREE.Vector2(head.dir.x, head.dir.z);
    const along = hd.lengthSq() > 1e-9
      ? ((x - head.origin.x) * head.dir.x + (z - head.origin.z) * head.dir.z) / hd.lengthSq()
      : 0;
    const height = head.origin.y + head.dir.y * along;
    if (!(height > 0.5 && height < 3.2)) continue;
    const kps = allArms?.[bi];
    out.push({ x, z, height, arms: kps ? liftArms(cam, kps, x, z) : undefined });
  }
  return out;
}

/**
 * Arm keypoints ([x, y, conf] for L/R shoulder, L/R elbow, L/R wrist) onto
 * the vertical plane through the feet facing the camera.
 */
function liftArms(
  cam: THREE.PerspectiveCamera,
  kps: number[][],
  x: number,
  z: number,
): CourtPlayer["arms"] {
  const n = new THREE.Vector3(x - cam.position.x, 0, z - cam.position.z);
  if (n.lengthSq() < 1e-6 || kps.length < 6) return undefined;
  n.normalize();
  const feet = new THREE.Vector3(x, 0, z);
  const lift = (k: number[] | undefined): THREE.Vector3 | null => {
    if (!k || !(k[2] >= ARM_MIN_CONF)) return null;
    const ray = rayFromImagePos(cam, [k[0], k[1]]);
    const den = n.dot(ray.dir);
    if (Math.abs(den) < 1e-6) return null;
    const t = n.dot(feet.clone().sub(ray.origin)) / den;
    return t > 0 ? ray.origin.clone().addScaledVector(ray.dir, t) : null;
  };
  const arm = (s: number, e: number, w: number): ArmJoints | null => {
    const p = [lift(kps[s]), lift(kps[e]), lift(kps[w])];
    return p[0] && p[1] && p[2] ? [p[0], p[1], p[2]] : null;
  };
  const left = arm(0, 2, 4), right = arm(1, 3, 5);
  return left || right ? { left, right } : undefined;
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
/**
 * Weight of the soft camera-depth prior (from apparent ball size) in the arc
 * solver objective.
 *
 * Disabled. On broadcast footage the ball spans ~25 px under a ~9 deg FoV, so
 * depth-from-apparent-size carries noise of the same order as the signal and
 * drags the fit far off the true arc. Gravity already supplies the metric
 * scale that makes a ballistic arc observable from 2D reprojection alone.
 * Measured over 8 noise seeds against a known ground-truth arc, mean 3D error:
 *
 *   weight   dense    noisy    sparse
 *   2e-3     3.18m    7.19m    3.78m
 *   5e-4     2.17m    3.49m    2.18m
 *   1e-4     0.97m    1.39m    1.04m
 *   0        0.35m    1.59m    0.66m   <- selected
 *
 * The bbox depth is still used to seed the initial guess, where its noise is
 * harmless because the multi-seed fit refines from there.
 */
const DEPTH_PRIOR_WEIGHT = 0;
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

// ── Short-arc-debug mode: stricter validation bounds ─────────────────────────
/** Hard X bound for short-arc-debug mode (m). */
const SAD_MAX_X_M = 6;
/** Hard Z bound for short-arc-debug mode (m). */
const SAD_MAX_Z_M = 11;
/** Maximum average reprojection error for short-arc-debug acceptance (px). */
const SAD_MAX_AVG_ERR_PX = 20;
/** Maximum single-frame reprojection error for short-arc-debug acceptance (px). */
const SAD_MAX_MAX_ERR_PX = 40;
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
function splitArcsEx(obs: ArcObsEx[], videoWidth?: number, videoHeight?: number): ArcObsEx[][] {
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
  if (videoWidth == null || videoHeight == null) return arcs;
  return arcs.flatMap((a) => scanTouches(a, videoWidth / 2, videoHeight / 2));
}

/** Least-squares fit of y = c0 + c1·t + c2·t², returning [c0, c1, c2]. */
function _fitQuadratic(ts: number[], ys: number[]): [number, number, number] {
  // Normal equations on the 3×3 moment matrix, solved by Cramer's rule.
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, b0 = 0, b1 = 0, b2 = 0;
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i], t2 = t * t;
    s0 += 1; s1 += t; s2 += t2; s3 += t2 * t; s4 += t2 * t2;
    b0 += ys[i]; b1 += ys[i] * t; b2 += ys[i] * t2;
  }
  const det3 = (m: number[]) =>
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  const A = [s0, s1, s2, s1, s2, s3, s2, s3, s4];
  const D = det3(A);
  if (Math.abs(D) < 1e-12) return [ys[0], 0, 0];
  return [
    det3([b0, s1, s2, b1, s2, s3, b2, s3, s4]) / D,
    det3([s0, b0, s2, s1, b1, s3, s2, b2, s4]) / D,
    det3([s0, s1, b0, s1, s2, b1, s2, s3, b2]) / D,
  ];
}

// ─── Flight model for touch detection ────────────────────────────────────────
//
// A ball in free flight moves on a parabola in space. Under a long broadcast
// lens its picture is nearly a parabola in time too, but not under a wide
// one: from a phone behind the end line the ball is 6 m away at one baseline
// and 25 m at the other, and the picture of a serve slows to a crawl as it
// flies away. Fitted with an image parabola, such a flight was cut every few
// frames and its real touches were lost among the cuts.
//
// The picture of a parabola in space is a parabola divided by the distance
// of the ball, which along a flight is itself a parabola in time:
//   u(s) = (a0 + a1·s + a2·s²) / (1 + c1·s + c2·s²),  v(s) likewise with b.
// That is linear in its eight numbers once multiplied out, so it is fitted
// directly, without the camera. c1 and c2 are how fast the distance changes
// (per second); a weak pull to zero leaves the image parabola wherever the
// samples do not show the distance changing (a long lens, a short window).

/** Camera of the clip being solved, for distances at touches. Set by the physics block. */
let flightCam: {
  cam: THREE.PerspectiveCamera;
  /** Focal length (px) and distance to the middle of the court (m). */
  fPx: number;
  courtDist: number;
} | null = null;

/** Pull of the distance terms towards zero: px of residual per 1/s (c1) and 1/s² (c2). */
const FLIGHT_DEPTH_RIDGE_PX = 3;
const FLIGHT_DEPTH_RIDGE2_PX = 10;
/** Never divide by less than this share of the mid-window distance. */
const FLIGHT_MIN_REL_DEPTH = 0.25;

/** Distance in pixels between each sample and the flight fitted to them, or null if it cannot be fitted. */
function fitFlight(obs: ArcObs[], hw: number, hh: number): ((o: ArcObs) => number) | null {
  if (obs.length < 3) return null;
  const tm = 0.5 * (obs[0].t + obs[obs.length - 1].t);
  const N = 8;
  let x: number[] | null = null;
  const den = (sol: number[] | null, s: number) => (sol ? Math.max(FLIGHT_MIN_REL_DEPTH, 1 + sol[6] * s + sol[7] * s * s) : 1);
  // Reweighted least squares: dividing each row by its distance term turns
  // the multiplied-out equations back into pixel errors.
  for (let pass = 0; pass < 3; pass++) {
    const A = Array.from({ length: N }, () => new Array<number>(N).fill(0));
    const g = new Array<number>(N).fill(0);
    const add = (row: number[], rhs: number) => {
      for (let i = 0; i < N; i++) {
        if (row[i] === 0) continue;
        g[i] += row[i] * rhs;
        for (let j = i; j < N; j++) A[i][j] += row[i] * row[j];
      }
    };
    for (const o of obs) {
      const s = o.t - tm, s2 = s * s;
      const wu = hw / den(x, s), wv = hh / den(x, s);
      add([wu, wu * s, wu * s2, 0, 0, 0, -wu * o.u * s, -wu * o.u * s2], wu * o.u);
      add([0, 0, 0, wv, wv * s, wv * s2, -wv * o.v * s, -wv * o.v * s2], wv * o.v);
    }
    add([0, 0, 0, 0, 0, 0, FLIGHT_DEPTH_RIDGE_PX, 0], 0);
    add([0, 0, 0, 0, 0, 0, 0, FLIGHT_DEPTH_RIDGE2_PX], 0);
    for (let i = 0; i < N; i++) { A[i][i] += 1e-12; for (let j = 0; j < i; j++) A[i][j] = A[j][i]; }
    const sol = _solveLinear(A, g);
    if (!sol) break;
    x = sol;
  }
  const devOf = (sol: number[]) => (o: ArcObs) => {
    const s = o.t - tm, s2 = s * s, d = den(sol, s);
    return Math.hypot(((sol[0] + sol[1] * s + sol[2] * s2) / d - o.u) * hw, ((sol[3] + sol[4] * s + sol[5] * s2) / d - o.v) * hh);
  };
  // The image parabola is the same model with a constant distance. The
  // reweighting above does not always settle (a window with a touch in it has
  // no flight to find), so the parabola stands unless the fuller fit beats it.
  const ts = obs.map((o) => o.t - tm);
  const plain = [..._fitQuadratic(ts, obs.map((o) => o.u)), ..._fitQuadratic(ts, obs.map((o) => o.v)), 0, 0];
  const sse = (sol: number[]) => {
    const dev = devOf(sol);
    return obs.reduce((sum, o) => sum + dev(o) ** 2, 0) + (FLIGHT_DEPTH_RIDGE_PX * sol[6]) ** 2 + (FLIGHT_DEPTH_RIDGE2_PX * sol[7]) ** 2;
  };
  return devOf(x && sse(x) < sse(plain) ? x : plain);
}

/**
 * Detection noise of a track (px, per axis), from third differences of
 * consecutive samples: a flight contributes nothing to those, and the median
 * ignores the few that straddle a touch.
 */
function _trackNoisePx(obs: ArcObs[], hw: number, hh: number): number {
  const d3: number[] = [];
  for (let i = 0; i + 3 < obs.length; i++) {
    if (obs[i + 3].fi - obs[i].fi !== 3) continue;
    d3.push(Math.abs(obs[i + 3].u - 3 * obs[i + 2].u + 3 * obs[i + 1].u - obs[i].u) * hw);
    d3.push(Math.abs(obs[i + 3].v - 3 * obs[i + 2].v + 3 * obs[i + 1].v - obs[i].v) * hh);
  }
  // Var(d3) = 20 sigma²; median |x| = 0.6745 sigma for a normal x.
  return d3.length >= 8 ? _median(d3) / 0.6745 / Math.sqrt(20) : SCAN_MIN_SIGMA_PX;
}

/** Track either side of a possible touch that is compared (s), and the fewest samples a side. */
const SCAN_WINDOW_S = 0.5;
const SCAN_MIN_SIDE = 4;
/**
 * One side may be this short when the other is a full one: a touch two
 * samples before the track ends, or the two samples between an attack and
 * its block. Too few to fit a flight to, so such a cut needs
 * SCAN_SHORT_SIDE_FACTOR times the usual evidence.
 */
const SCAN_MIN_SHORT_SIDE = 2;
const SCAN_SHORT_SIDE_FACTOR = 2;
/**
 * What a sample in a too-short piece is charged, in noise variances: twice
 * what it would leave behind inside a fitted flight (2, one per axis), so
 * between equal cuts the one that keeps the short piece shortest wins.
 */
const SCAN_FREE_SAMPLE_COST = 2;
const _sidesOk = (nL: number, nR: number) => Math.min(nL, nR) >= SCAN_MIN_SHORT_SIDE && Math.max(nL, nR) >= SCAN_MIN_SIDE;
/** Detection noise is at least this (px), however well a short window happens to fit. */
const SCAN_MIN_SIGMA_PX = 1.5;

/** Samples either side of a cut it may still move by, and the samples beyond them each parabola is fitted to. */
const SCAN_FINE_FRAMES = 3;
const SCAN_FINE_WINDOW = 6;

/** As _flightSse, with a plain image parabola: for a few samples, where it is the same thing. */
function _parabolaSse(o: ArcObsEx[], hw: number, hh: number, sigma: number): number {
  if (o.length < SCAN_MIN_SIDE) return SCAN_FREE_SAMPLE_COST * sigma * sigma * o.length;
  const ts = o.map((q) => q.t - o[0].t);
  const cu = _fitQuadratic(ts, o.map((q) => q.u)), cv = _fitQuadratic(ts, o.map((q) => q.v));
  return o.reduce((sum, q, i) => sum + ((cu[0] + cu[1] * ts[i] + cu[2] * ts[i] ** 2 - q.u) * hw) ** 2 + ((cv[0] + cv[1] * ts[i] + cv[2] * ts[i] ** 2 - q.v) * hh) ** 2, 0);
}

/** Sum of squared distances (px²) between samples and the flight fitted to them. */
function _flightSse(obs: ArcObsEx[], hw: number, hh: number, sigma: number): number {
  // Fewer than four samples lie on some flight whatever they are. They are
  // charged their expected noise, or a cut would always gain by handing one
  // more sample to the side that fits anything.
  if (obs.length < SCAN_MIN_SIDE) return SCAN_FREE_SAMPLE_COST * sigma * sigma * obs.length;
  const dev = fitFlight(obs, hw, hh);
  if (!dev) return Infinity;
  let sse = 0;
  for (const o of obs) sse += dev(o) ** 2;
  return sse;
}

/**
 * How clearly the track breaks between samples b-1 and b, in units of the
 * detection noise: how much worse one flight fits the samples around b than
 * two flights meeting there. Windows stop at `lo` and `hi` (other splits).
 */
function _kinkScore(obs: ArcObsEx[], b: number, lo: number, hi: number, hw: number, hh: number, sigma: number): number {
  let i0 = b, i1 = b;
  while (i0 > lo && obs[b - 1].t - obs[i0 - 1].t <= SCAN_WINDOW_S) i0--;
  while (i1 < hi && obs[i1].t - obs[b].t <= SCAN_WINDOW_S) i1++;
  if (!_sidesOk(b - i0, i1 - b)) return 0;
  const sL = _flightSse(obs.slice(i0, b), hw, hh, sigma), sR = _flightSse(obs.slice(b, i1), hw, hh, sigma), sJ = _flightSse(obs.slice(i0, i1), hw, hh, sigma);
  if (!Number.isFinite(sL + sR + sJ)) return 0;
  return (sJ - sL - sR) / (sigma * sigma) / (Math.min(b - i0, i1 - b) < SCAN_MIN_SIDE ? SCAN_SHORT_SIDE_FACTOR : 1);
}

/**
 * The boundary where two flights leave the least error, over the window
 * around `b`. The score is high all around a touch (any window holding the
 * kink fits badly as one flight) and its peak can sit a few samples off, but
 * within one window the best place to cut is the touch itself.
 */
function _refineCut(obs: ArcObsEx[], b: number, lo: number, hi: number, hw: number, hh: number, sigma: number): number {
  let i0 = b, i1 = b;
  while (i0 > lo && obs[b - 1].t - obs[i0 - 1].t <= SCAN_WINDOW_S) i0--;
  while (i1 < hi && obs[i1].t - obs[b].t <= SCAN_WINDOW_S) i1++;
  const twoFlights = (c: number, from: number, to: number, sse: (o: ArcObsEx[]) => number) => sse(obs.slice(from, c)) + sse(obs.slice(c, to));
  let best = b, bestErr = Infinity;
  for (let c = i0 + SCAN_MIN_SHORT_SIDE; c <= i1 - SCAN_MIN_SHORT_SIDE; c++) {
    if (!_sidesOk(c - i0, i1 - c)) continue;
    const e = twoFlights(c, i0, i1, (o) => _flightSse(o, hw, hh, sigma));
    // The scanned position wins ties.
    if (e < bestErr - 1e-9 || (c === b && e <= bestErr + 1e-9)) { best = c; bestErr = e; }
  }
  // To the sample: over a few samples either side a flight is a plain image
  // parabola under any lens, and a parabola cannot bend round the touch the
  // way the fuller model sometimes does with a sample or two of the other
  // flight.
  const parabolaSse = (o: ArcObsEx[]) => _parabolaSse(o, hw, hh, sigma);
  const j0 = Math.max(i0, best - SCAN_FINE_FRAMES - SCAN_FINE_WINDOW), j1 = Math.min(i1, best + SCAN_FINE_FRAMES + SCAN_FINE_WINDOW);
  let fine = best, fineErr = Infinity;
  for (let c = best - SCAN_FINE_FRAMES; c <= best + SCAN_FINE_FRAMES; c++) {
    if (!_sidesOk(c - j0, j1 - c) || c - i0 < SCAN_MIN_SHORT_SIDE || i1 - c < SCAN_MIN_SHORT_SIDE) continue;
    const e = twoFlights(c, j0, j1, parabolaSse);
    if (e < fineErr - 1e-9 || (c === best && e <= fineErr + 1e-9)) { fine = c; fineErr = e; }
  }
  return fine;
}

/** Cuts closer than this (s) are placed together (see _refinePair), each moved by up to SCAN_PAIR_FRAMES. */
const SCAN_PAIR_MAX_S = 0.3;
const SCAN_PAIR_FRAMES = 2;

/**
 * Place two close cuts together. An attack and its block are a few frames
 * apart: each cut on its own is placed against a neighbouring "flight" that
 * holds the other touch, and lands a sample or two off. Fitting the three
 * flights at once (before, between, after) puts both where they are.
 */
function _refinePair(obs: ArcObsEx[], c1: number, c2: number, lo: number, hi: number, hw: number, hh: number, sigma: number): [number, number] {
  // A fixed stretch of samples either side, so every candidate is judged on the same ones.
  const i0 = Math.max(lo, c1 - SCAN_PAIR_FRAMES - SCAN_FINE_WINDOW), i1 = Math.min(hi, c2 + SCAN_PAIR_FRAMES + SCAN_FINE_WINDOW);
  const sse = (a: number, b: number) => _parabolaSse(obs.slice(a, b), hw, hh, sigma);
  let best: [number, number] = [c1, c2], bestErr = Infinity;
  for (let a = c1 - SCAN_PAIR_FRAMES; a <= c1 + SCAN_PAIR_FRAMES; a++) {
    for (let b = c2 - SCAN_PAIR_FRAMES; b <= c2 + SCAN_PAIR_FRAMES; b++) {
      if (a - i0 < SCAN_MIN_SHORT_SIDE || i1 - b < SCAN_MIN_SHORT_SIDE || b - a < SCAN_MIN_SHORT_SIDE) continue;
      const e = sse(i0, a) + sse(a, b) + sse(b, i1);
      if (e < bestErr - 1e-9 || (a === c1 && b === c2 && e <= bestErr + 1e-9)) { best = [a, b]; bestErr = e; }
    }
  }
  return best;
}

/**
 * Split a continuous ball track at player touches (see the flight model).
 *
 * A touch is a kink: the samples before and after it each lie on a flight,
 * but no single flight passes through both. Every position in the track is
 * scored for that, using all samples within SCAN_WINDOW_S on both sides, and
 * the clearest kinks are cut first; each cut then bounds its neighbours'
 * windows, so a touch close to a stronger one is still judged on its own
 * flights. Looking both ways with every sample is what finds a touch at the
 * far baseline of a wide shot, where it moves the ball a few pixels.
 */
function scanTouches(obs: ArcObsEx[], hw: number, hh: number): ArcObsEx[][] {
  const n = obs.length;
  if (n < SCAN_MIN_SIDE + SCAN_MIN_SHORT_SIDE) return [obs];
  const sigma = Math.max(SCAN_MIN_SIGMA_PX, _trackNoisePx(obs, hw, hh));
  const cuts: number[] = [];
  const score = new Array<number>(n).fill(0);
  const bounds = (b: number): [number, number] => {
    let lo = 0, hi = n;
    for (const c of cuts) { if (c <= b && c > lo) lo = c; if (c > b && c < hi) hi = c; }
    return [lo, hi];
  };
  const rescore = (from: number, to: number) => {
    for (let b = Math.max(SCAN_MIN_SHORT_SIDE, from); b <= Math.min(n - SCAN_MIN_SHORT_SIDE, to); b++) {
      score[b] = cuts.includes(b) ? 0 : _kinkScore(obs, b, ...bounds(b), hw, hh, sigma);
    }
  };
  rescore(0, n);
  for (;;) {
    let best = -1;
    for (let b = SCAN_MIN_SHORT_SIDE; b <= n - SCAN_MIN_SHORT_SIDE; b++) {
      if (score[b] < P.touchKinkScore || (best >= 0 && score[b] <= score[best])) continue;
      if (cuts.some((c) => Math.abs(c - b) < SCAN_MIN_SHORT_SIDE)) continue;
      best = b;
    }
    if (best < 0) break;
    const at = _refineCut(obs, best, ...bounds(best), hw, hh, sigma);
    const cut = cuts.some((c) => Math.abs(c - at) < SCAN_MIN_SHORT_SIDE) ? best : at;
    obs[cut].kink = score[best];
    cuts.push(cut);
    // Windows that reached across the new cut now stop at it.
    let from = best, to = best;
    while (from > 0 && obs[best].t - obs[from - 1].t <= 2 * SCAN_WINDOW_S) from--;
    while (to < n - 1 && obs[to + 1].t - obs[best].t <= 2 * SCAN_WINDOW_S) to++;
    rescore(from, to);
  }
  cuts.sort((x, y) => x - y);
  for (let k = 0; k + 1 < cuts.length; k++) {
    if (obs[cuts[k + 1]].t - obs[cuts[k]].t > SCAN_PAIR_MAX_S) continue;
    const [a, b] = _refinePair(obs, cuts[k], cuts[k + 1], k > 0 ? cuts[k - 1] : 0, k + 2 < cuts.length ? cuts[k + 2] : n, hw, hh, sigma);
    if ((k > 0 && a - cuts[k - 1] < SCAN_MIN_SHORT_SIDE) || (k + 2 < cuts.length && cuts[k + 2] - b < SCAN_MIN_SHORT_SIDE)) continue;
    const [ka, kb] = [obs[cuts[k]].kink, obs[cuts[k + 1]].kink];
    obs[cuts[k]].kink = undefined; obs[cuts[k + 1]].kink = undefined;
    cuts[k] = a; cuts[k + 1] = b;
    obs[a].kink = ka; obs[b].kink = kb;
  }
  const out: ArcObsEx[][] = [];
  let start = 0;
  for (const c of cuts) { out.push(obs.slice(start, c)); start = c; }
  out.push(obs.slice(start));
  return out;
}

/**
 * Split a continuous ball track at player touches.
 *
 * Between touches the ball is in free flight (see the flight model above). A
 * touch (dig, set, attack) breaks that: the next positions leave the flight
 * fitted to the points so far. Without this, a whole rally becomes one "arc"
 * that no single ballistic trajectory can explain, so the solver rejects it.
 * Two consecutive points must both deviate, so a single noisy detection does
 * not cut an arc.
 */
// ─── Rally solver: connected ballistic segments ──────────────────────────────
//
// Fitting each segment between touches on its own leaves its depth almost
// unobservable: from a camera behind the baseline, depth runs along the court,
// and a short arc has too little visible gravity drop to fix the scale. The
// result was arcs flying off at 100+ m/s or sitting 12 m behind a baseline.
//
// A rally is one continuous ball path: between touches it is in free flight,
// and at each touch the path is continuous in position (only the velocity
// jumps). So the whole rally is parameterised by the ball's 3D position at
// each touch ("knots"); a segment's trajectory is the unique ballistic arc
// joining its two knots in the time between them. Long, well-curved segments
// pin the depth of their knots, and continuity carries that to the short ones.
// A split where there was no real touch costs nothing: the knot simply lands
// on the arc.

/**
 * A knot this low (ball centre, m) is the ball on the floor: the play is over
 * and later knots are bounces or catches, not touches. A ball resting on the
 * floor has its centre at 0.105 m; real digs are played well above that.
 */
const FLOOR_CONTACT_Y_M = 0.3;
/** Longest detection gap (s) across which a touch is dated at the knot (see touch reporting). */
const TOUCH_FRAME_MAX_GAP_S = 0.4;
/** A flight whose arc gets this low (ball centre, m) has reached the floor. */
const FLOOR_ARC_Y_M = 0.15;
/** Touches happen at player height: floor of the soft bound on knot height (m). */
const KNOT_MIN_Y_M = 0;
/** Soft bounds on where the ball can be (m): court plus free zone. */
const RALLY_MAX_ABS_X_M = COURT_HALF_W + 5;
const RALLY_MAX_ABS_Z_M = COURT_HALF_L + 8;
const SEG_MAX_Y_M = 10;

/**
 * Where one type of touch can happen (see applyRallyRules). The side of the
 * net is derived from who served; these are the distances and heights.
 */
export interface TouchRule {
  /** Distance from the net (m), measured into the touching team's half. */
  minAbsZ: number;
  maxAbsZ: number;
  /** Contact height range (m). */
  minY: number;
  maxY: number;
  /**
   * Effective downward acceleration (multiples of g) of the flight that this
   * touch starts. Only serves and attacks carry heavy topspin; a toss, pass
   * or set falls at about g, and letting those drop faster makes them short
   * and flat instead of the high arcs they are.
   */
  minG: number;
  maxG: number;
  /**
   * A player this far from the net (m) may still make the touch when nobody
   * is in the usual zone; the zone is then stretched to them. Without a
   * player the usual zone holds, so an unanchored touch does not drift.
   */
  reachAbsZ?: number;
}

/**
 * Every tuned number of the rally solver, in one place, so tests and the
 * tuner (src/eval) can vary them without editing code. Defaults were set on
 * broadcast 1080p footage (France vs Japan); see each field for why.
 */
export interface RallyParams {
  /**
   * Deviation from a smooth curve that marks a candidate touch split, in
   * metres at the court (converted to pixels per clip; see courtMetresPerPx).
   * 0.071 m is 10 px on the France vs Japan broadcast camera.
   */
  touchSplitM: number;
  /** Observations a segment needs before touch detection can split it. */
  touchMinSeg: number;
  /**
   * Minimum jump in the ball's image velocity for a split to count as a
   * touch, in m/s at the court (converted per clip, so zoom and camera
   * distance don't change it). Measured on the France vs Japan broadcast
   * camera, where 1 px is ~7 mm: real touches (serve contact, reception, set,
   * attack) change it by ~950-1900 px/s (6.7-13 m/s); spurious splits from
   * detection jitter, perspective or the slow top of a serve toss by
   * ~70-420 px/s (0.5-3 m/s).
   */
  touchMinDvMps: number;
  /** The same for a split found as a kink in the track rather than at a detection gap (see mergeUntouchedSplits). */
  touchMinDvKinkMps: number;
  /**
   * How clearly the track must break for a touch split, in units of the
   * detection noise squared (see scanTouches). By chance alone a flight
   * scores above 28 once in 10 000 positions.
   */
  touchKinkScore: number;
  /** A kink this clear (with twice touchMinDvMps) is a touch even with no player detected in reach. */
  touchKinkCertain: number;
  /**
   * Max seconds without the ball for two pieces of track to belong to the
   * same rally. Real side-view broadcasts lose a fast serve for 0.5-1 s; at
   * 0.5 the reception started a "new rally" and was labelled a serve. Real
   * rallies are seconds apart. 1.0 scored best on the labelled real clips.
   */
  rallyMaxGapS: number;
  /** Ceiling of the soft bound on knot (touch) height (m). */
  knotMaxY: number;
  /** Speeds above this (m/s) are penalised; a hard men's jump serve is ~30 m/s. */
  softMaxSpeed: number;
  /**
   * Air drag: deceleration per m/s of speed, per m/s (rho·Cd·A / 2m). A
   * volleyball is light for its size: 1.2·0.47·0.035 / (2·0.27) ~ 0.036,
   * i.e. ~0.4 g of drag at 10 m/s. 0 is a drag-free arc. Off by default:
   * at 0.036 the labelled real clips moved by -1.8 points overall (better on
   * most, one fragile clip lost every arc) and the drag-free synthetic
   * scenarios cannot judge it.
   */
  dragPerM: number;
  /** Residual px per m/s of velocity change at a knot (touch impulse prior). */
  touchImpulseWeight: number;
  /**
   * Residual px per unit of log-depth error for the apparent-size prior. Size
   * is measured to roughly ±15% per frame and is biased in front of the crowd.
   */
  sizeDepthWeight: number;
  /** |log depth error| beyond which a size measurement is discarded (~35%). */
  sizeDepthOutlier: number;
  /**
   * Bounds (multiples of g) on the effective downward acceleration of flights
   * without a touch rule. Pure gravity is wrong for a volleyball: topspin adds
   * Magnus force and can roughly double the drop, and forcing exactly g made
   * the solver explain a spinning serve's extra curvature as "closer to the
   * camera", pulling it onto the wrong side of the net.
   */
  accelMinG: number;
  accelMaxG: number;
  /**
   * Weight of a touch point's own image position (per px). Below 1 because in
   * a gap it comes from the analyzer's straight-line fill, which cuts the
   * corner of the turn by a few px.
   */
  knotImageWeight: number;
  /** Rejected segments up to this many samples between accepted ones are bridged. */
  bridgeMaxFrames: number;
  /** Scale (px) of the robust reprojection loss in the rally solver. */
  robustReprojPx: number;
  /** Reprojection error (px) above which an observation is treated as an outlier. */
  rallyOutlierPx: number;
  /**
   * Per-segment acceptance: median error, not mean. TrackNet occasionally sits
   * ~15 px off the ball for a few frames, which says nothing about the arc.
   * Under a broadcast lens (f ~ 6600 px at 45-55 m) 10 px is ~7 cm at the ball.
   */
  segMaxMedianErrPx: number;
  /** Share of a segment's samples that must lie within rallyOutlierPx. */
  segMinInlierFrac: number;
  /** Frames either side searched for a player's lowest feet (take-off spot). */
  playerTrackFrames: number;
  /** Residual px per metre between a knot and its nearest candidate player. */
  anchorWeight: number;
  /** Distance (m) beyond which the anchor pull stops growing (robust). */
  anchorRobustM: number;
  /** A player beyond this |z| (m) at the start of a rally is serving. */
  serverMinAbsZ: number;
  /** Where each touch type can happen, in rally order. */
  rules: {
    toss: TouchRule;
    serve: TouchRule;
    reception: TouchRule;
    set: TouchRule;
    attack: TouchRule;
    block: TouchRule;
    dig: TouchRule;
  };
}

export const DEFAULT_RALLY_PARAMS: RallyParams = {
  touchSplitM: 0.071,
  touchMinSeg: 5,
  touchMinDvMps: 3.55,
  touchMinDvKinkMps: 2.0,
  touchKinkScore: 40,
  touchKinkCertain: 1000,
  rallyMaxGapS: 1.0,
  knotMaxY: 4.5,
  softMaxSpeed: 30,
  dragPerM: 0,
  touchImpulseWeight: 0.3,
  sizeDepthWeight: 25,
  sizeDepthOutlier: 0.3,
  accelMinG: 0.8,
  accelMaxG: 2.2,
  knotImageWeight: 0.4,
  bridgeMaxFrames: 6,
  robustReprojPx: 5,
  rallyOutlierPx: 25,
  segMaxMedianErrPx: 10,
  segMinInlierFrac: 0.7,
  playerTrackFrames: 10,
  anchorWeight: 25,
  anchorRobustM: 3,
  serverMinAbsZ: COURT_HALF_L - 1,
  rules: {
    // Heights: the toss leaves the hand at ~1-2.5 m; a jump serve, set and
    // attack are struck above the head; a reception can be anything from a
    // low dig to an overhead pass; a block is above the net tape.
    toss:      { minAbsZ: COURT_HALF_L - 1, maxAbsZ: RALLY_MAX_ABS_Z_M, minY: 1.0, maxY: 2.6, minG: 0.9, maxG: 1.15 },
    serve:     { minAbsZ: COURT_HALF_L - 2, maxAbsZ: RALLY_MAX_ABS_Z_M, minY: 2.4, maxY: 3.8, minG: 0.8, maxG: 2.2 },
    // Received in the back court: even a short serve lands 3-4 m from the net.
    reception: { minAbsZ: 2, maxAbsZ: RALLY_MAX_ABS_Z_M, minY: 0.2, maxY: 3.0, minG: 0.9, maxG: 1.15 },
    // Usually at the net, but a setter chasing a poor pass sets from the
    // back court (pol_ita_r3: 5.6 m out): a player up to reachAbsZ counts
    // when nobody is in the usual zone, and the zone stretches to them.
    set:       { minAbsZ: 0, maxAbsZ: 4,   minY: 2.2, maxY: 3.4, minG: 0.9, maxG: 1.15, reachAbsZ: 7 },
    // Front-row attacks are hit ~0.5-1 m from the net. The attacker is often
    // hidden behind the block (no player anchor), and from behind the baseline
    // the set's end depth is nearly unobservable: on broadcast footage, ending
    // it 1 m or 3 m from the net differed by ~3 px. Known limit: a back-row
    // attack (contact ~2-3 m out) is drawn too close.
    attack:    { minAbsZ: 0, maxAbsZ: 1.0, minY: 2.9, maxY: 3.8, minG: 0.8, maxG: 2.2 },
    block:     { minAbsZ: 0, maxAbsZ: 1.5, minY: 2.4, maxY: 3.6, minG: 0.8, maxG: 1.5 },
    // A dig is played low, anywhere from the 3 m line back (and beyond the
    // end line); the ball leaves it at about g, like a pass.
    dig:       { minAbsZ: 1.5, maxAbsZ: RALLY_MAX_ABS_Z_M, minY: 0.1, maxY: 2.6, minG: 0.9, maxG: 1.15 },
  },
};

/**
 * Parameters of the solveBallTrajectories call in progress. Module state
 * rather than an argument threaded through every helper; set and restored by
 * solveBallTrajectories, and JS runs one call at a time.
 */
let P: RallyParams = DEFAULT_RALLY_PARAMS;

/**
 * Metres at the court per image pixel for the clip being solved: camera
 * distance to the court centre over the focal length in pixels. Converts the
 * metre-based touch thresholds to pixels, so a wide side camera and a
 * telephoto broadcast camera are judged alike. Set by the physics block.
 */
let courtMetresPerPx = 0.00709;
/** Frame rate of the clip being solved; windows given in frames are meant at 30 fps. */
let clipFps = 30;
/** Every player on every frame, when the analyzer gives them: what a player is followed through. */
let trackBoxes: Map<number, PlayerBox[]> | null = null;

/** Combined projection, precomputed for speed: world (x,y,z) -> NDC (u,v). */
function _makeProjector(cam: THREE.PerspectiveCamera) {
  const m = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).elements;
  return (x: number, y: number, z: number): [number, number] => {
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
            (m[1] * x + m[5] * y + m[9] * z + m[13]) / w];
  };
}

/** Solve A·x = b by Gaussian elimination with partial pivoting. */
function _solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/**
 * A group of residuals and the only parameters it depends on. The Jacobian
 * is then differenced block by block: a rally's residuals each touch a few of
 * its knots, so perturbing one parameter re-evaluates a handful of terms
 * instead of the whole rally (that full re-evaluation was ~99% of the
 * analysis time on a long rally).
 */
interface ResidualBlock {
  params: number[];
  eval: (p: number[]) => number[];
}

function _blockResiduals(blocks: ResidualBlock[], p: number[]): number[] {
  const r: number[] = [];
  for (const b of blocks) for (const v of b.eval(p)) r.push(v);
  return r;
}

/** Levenberg-Marquardt on block residuals, numeric (forward-difference) Jacobian. */
function _levenbergMarquardt(p0: number[], blocks: ResidualBlock[], maxIter = 100): number[] {
  let p = p0.slice();
  let r = _blockResiduals(blocks, p);
  let cost = r.reduce((s, v) => s + v * v, 0);
  let lambda = 1e-2;
  const m = p.length;
  const h = 1e-4;
  for (let iter = 0; iter < maxIter; iter++) {
    // Normal equations: (JᵀJ + λ·diag) δ = −Jᵀr, summed block by block.
    const JTJ = Array.from({ length: m }, () => new Array<number>(m).fill(0));
    const JTr = new Array<number>(m).fill(0);
    const q = p.slice();
    for (const b of blocks) {
      const base = b.eval(q);
      const J = b.params.map((j) => {
        const v0 = q[j];
        q[j] = v0 + h;
        const rj = b.eval(q);
        q[j] = v0;
        return rj.map((v, i) => (v - base[i]) / h);
      });
      for (let a = 0; a < J.length; a++) {
        const ja = b.params[a];
        for (let i = 0; i < base.length; i++) JTr[ja] += J[a][i] * base[i];
        for (let c = a; c < J.length; c++) {
          let s = 0;
          for (let i = 0; i < base.length; i++) s += J[a][i] * J[c][i];
          JTJ[ja][b.params[c]] += s;
          if (c !== a) JTJ[b.params[c]][ja] += s;
        }
      }
    }
    let improved = false;
    for (let tries = 0; tries < 10; tries++) {
      const A = JTJ.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lambda) + 1e-9 : v)));
      const delta = _solveLinear(A, JTr.map((v) => -v));
      if (!delta) { lambda *= 10; continue; }
      const pn = p.map((v, k) => v + delta[k]);
      const rn = _blockResiduals(blocks, pn);
      const cn = rn.reduce((s, v) => s + v * v, 0);
      if (cn < cost) {
        const rel = (cost - cn) / Math.max(cost, 1e-12);
        p = pn; r = rn; cost = cn;
        lambda = Math.max(1e-7, lambda / 10);
        improved = true;
        if (rel < 1e-9) return p;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return p;
}

interface RallySegmentResult {
  obs: ArcObsEx[];
  /** Knot times bounding this segment (s). */
  tStart: number;
  tEnd: number;
  /** Position at time t (s, absolute), valid on [tStart, tEnd]. */
  posAt: (t: number) => [number, number, number];
  /** Launch speed (m/s) at tStart. */
  speedMps: number;
  accepted: boolean;
  rejectReason: string | null;
  errPx: number[];
  outlier: boolean[];
}

/** Ball this close to the top of the picture (NDC v) is leaving or entering it there. */
const TOP_EDGE_NDC = 0.85;
/** Longest flight above the picture that is joined into one arc (s). */
const OFFSCREEN_MAX_S = 4;

/**
 * Join the two pieces of a flight that went out of the top of the picture.
 * High balls (toss, serve, set, free ball) regularly leave the frame for
 * 1-2 s. Split there, the rally broke in two (gaps over rallyMaxGapS) or got
 * a knot in the gap, near the apex, where the knot-height cap and the
 * analyzer's straight-line gap fill pulled the arc down. Nobody touches the
 * ball above the picture, so exit and re-entry are one ballistic arc, and its
 * timing and gravity give the route above the frame.
 */
function joinOffscreenFlights(
  arcs: ArcObsEx[][],
  /** Frames of re-entry samples not to join at (their joined arc did not fit). */
  skip: Set<number> = new Set(),
  /** Receives the frame of each re-entry sample joined. */
  joined?: number[],
): ArcObsEx[][] {
  const vOf = (o: ArcObsEx) => o.frameV ?? o.v;
  const leftThroughTop = (a: ArcObsEx[]) =>
    a.length >= 2 && vOf(a[a.length - 1]) >= TOP_EDGE_NDC && vOf(a[a.length - 1]) > vOf(a[Math.max(0, a.length - 3)]);
  const cameThroughTop = (a: ArcObsEx[]) =>
    a.length >= 2 && vOf(a[0]) >= TOP_EDGE_NDC && vOf(a[0]) > vOf(a[Math.min(a.length - 1, 2)]);
  const out: ArcObsEx[][] = [];
  for (const arc of arcs) {
    const prev = out[out.length - 1];
    if (prev && arc.length > 0 && !skip.has(arc[0].fi) && leftThroughTop(prev) && cameThroughTop(arc) &&
        arc[0].t - prev[prev.length - 1].t <= OFFSCREEN_MAX_S) {
      out[out.length - 1] = [...prev, ...arc];
      joined?.push(arc[0].fi);
      continue;
    }
    out.push(arc);
  }
  return out;
}

/** Group touch-split segments into rallies (chains separated by short gaps). */
function groupIntoRallies(
  segments: ArcObsEx[][],
  /** Frames of first samples that continue the rally whatever the gap before them. */
  keepTogether: Set<number> = new Set(),
): ArcObsEx[][][] {
  const rallies: ArcObsEx[][][] = [];
  for (const seg of segments) {
    if (seg.length === 0) continue;
    const cur = rallies[rallies.length - 1];
    const prevSeg = cur?.[cur.length - 1];
    if (prevSeg && (seg[0].t - prevSeg[prevSeg.length - 1].t <= P.rallyMaxGapS || keepTogether.has(seg[0].fi))) cur.push(seg);
    else rallies.push([seg]);
  }
  return rallies;
}

// ─── Player anchors ──────────────────────────────────────────────────────────
//
// Every touch happens at a player, and a player's feet stand on the
// calibrated floor, so the touching player fixes the knot's position on the
// court - the one thing the ball alone cannot pin down from behind the
// baseline. From this camera the ball overlaps far-side players in the image
// almost all the time, so "ball inside a box" cannot say who touched it. Each
// knot instead gets every plausible player as a candidate, and the solver is
// pulled to the nearest one: ball size, gravity and continuity decide which.

/** Normalised person box [x1, y1, x2, y2, conf] from the analyzer. */
type PlayerBox = number[];
/** A possible toucher's position on the court: [x, z]. */
type TouchCandidate = [number, number];
/** Ball must clear the net tape by this much when it crosses (m). */
const NET_CLEAR_M = 2.43 + 0.1;
/** Referee zone: within this |z| of the net and this far beyond a sideline (m). */
const OFFICIALS_NET_Z_M = 1;
const OFFICIALS_OUT_M = 3;

function _groundPointOf(cam: THREE.PerspectiveCamera, xn: number, yn: number): [number, number] | null {
  const { origin, dir } = rayFromImagePos(cam, [xn, yn]);
  if (dir.y > -1e-6) return null;
  const t = -origin.y / dir.y;
  return [origin.x + dir.x * t, origin.z + dir.z * t];
}

function _boxIou(a: PlayerBox, b: PlayerBox): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** Court positions of players who could have touched the ball at frame `fi`. */
function touchCandidates(
  fi: number,
  ball: [number, number],
  playersByFrame: Map<number, PlayerBox[]>,
  cam: THREE.PerspectiveCamera,
  /** How far above the head (in body heights) the ball may be. */
  reachUp = 0.8,
): TouchCandidate[] {
  const boxes = playersByFrame.get(fi);
  if (!boxes) return [];
  const [bx, by] = ball;
  const out: TouchCandidate[] = [];
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box;
    const w = x2 - x1, h = y2 - y1;
    // Ball within reach: beside the body or above it (arms, jumps), never
    // below the feet.
    if (bx < x1 - 0.5 * w || bx > x2 + 0.5 * w) continue;
    if (by < y1 - reachUp * h || by > y2) continue;
    const g = takeOffSpot(fi, box, playersByFrame, cam);
    if (g) out.push(g);
  }
  return out;
}

/**
 * Court position of a player's take-off spot. A jumping server or attacker
 * has their feet off the floor, which would place them metres too deep, so
 * use the lowest feet of the same player over nearby frames. "Same player"
 * must be followed frame to frame by overlap - from behind the baseline,
 * far-side players and near-net players share image columns and sizes, so
 * matching on column alone grabbed someone else's feet.
 */
function takeOffSpot(
  fi: number,
  box: PlayerBox,
  playersByFrame: Map<number, PlayerBox[]>,
  cam: THREE.PerspectiveCamera,
): TouchCandidate | null {
  const [x1, y1, x2, y2] = box;
  const h = y2 - y1;
  // How much lower the same player's feet were d frames before and after.
  const window = Math.max(1, Math.round(P.playerTrackFrames * clipFps / 30));
  const drop: number[][] = [[], []];
  [-1, 1].forEach((step, side) => {
    let cur: PlayerBox = [x1, y1, x2, y2];
    let lowest = 0;
    for (let d = 1; d <= window; d++) {
      let next: PlayerBox | null = null;
      let bestIou = 0.5;
      for (const c of (trackBoxes ?? playersByFrame).get(fi + step * d) ?? []) {
        const iou = _boxIou(cur, c);
        if (iou > bestIou) { bestIou = iou; next = c; }
      }
      if (!next) break;
      cur = next;
      // Only while the box keeps its size. A jump lifts the whole box; a box
      // that grows as its feet come down is the player coming closer, or the
      // track slipping onto someone nearer the camera (pol_ita_r2: an
      // attacker 7 m out was placed at 3.6 m by a neighbour's feet).
      const hc = cur[3] - cur[1];
      if (Math.abs(hc / h - 1) <= TAKEOFF_SAME_SIZE) lowest = Math.max(lowest, cur[3] - y2);
      drop[side][d] = lowest;
    }
  });
  // A jump goes up and comes down: the feet were lower both before and
  // after. Lower on one side only is a player moving through (an attacker's
  // run-up comes from deeper in the court; taken as a jump, it put the
  // take-off spot where the player had been 0.4 s earlier, metres away).
  // Fixtures that only list the players near the ball lose a server or
  // attacker as the ball leaves; there a side whose track has ended cannot
  // say, and the other is taken alone. With every player listed, a track
  // that ends is a player lost in a crowd, and nothing beyond it is used.
  let lift = 0;
  for (let d = 1; d <= window; d++) {
    const before = drop[0][d], after = drop[1][d];
    if (before == null || after == null) {
      if (trackBoxes || (before == null && after == null)) break;
      lift = Math.max(lift, (before ?? after)!);
    } else lift = Math.max(lift, Math.min(before, after));
  }
  // Nobody jumps half their own height, and the lowest of many box edges is
  // always a little lower: below a hop's worth it is only that jitter, which
  // from a low camera is a metre or two on the floor.
  const feetY = lift > TAKEOFF_MAX_LIFT * h || lift < TAKEOFF_MIN_LIFT * h ? y2 : y2 + lift;
  const g = _groundPointOf(cam, (x1 + x2) / 2, feetY);
  if (!g || Math.abs(g[0]) > RALLY_MAX_ABS_X_M || Math.abs(g[1]) > RALLY_MAX_ABS_Z_M) return null;
  // The referees stand at the net posts, beyond the sidelines (the first on
  // a raised stand, so their "feet" land even further out). pol_ita_r6: the
  // first referee (9.4 m out) anchored a block and pulled it off the court.
  if (Math.abs(g[1]) < OFFICIALS_NET_Z_M && Math.abs(g[0]) > COURT_HALF_W + OFFICIALS_OUT_M) return null;
  return g;
}

/** How much a tracked box may differ in height (fraction) and still show the same player at the same distance. */
const TAKEOFF_SAME_SIZE = 0.15;
/** Highest jump, as a share of the box height. */
const TAKEOFF_MAX_LIFT = 0.45;
/** Smallest lift taken as a jump, as a share of the box height (~0.15 m). */
const TAKEOFF_MIN_LIFT = 0.08;

/** Arm keypoints of one player: [x, y, conf] for shoulders, elbows, wrists (left, right). */
type PlayerArms = number[][];
/** Wrist keypoints this confident are used. */
const WRIST_MIN_CONF = 0.3;
/** Longest detection gap (s) in which a contact is searched for at the arms. */
const HIDDEN_CONTACT_MAX_GAP_S = 0.8;
/**
 * How far a path extended into a gap may miss the wrists: sigma = base +
 * accel * dt^2 (px at 1080p), as an unknown acceleration (a serve diving
 * onto the receiver) bends the real path away from the extrapolation with
 * the square of the time. The two paths' misses in sigmas are summed.
 */
const CONTACT_SIGMA_BASE_PX = 20;
const CONTACT_SIGMA_ACCEL_PX = 1000;
const HIDDEN_CONTACT_MAX_SIGMA = 4;

/** A touch found at a player's arms inside a detection gap. */
interface HiddenContact {
  t: number;
  frame: number;
  /** Contact point in the image (NDC): the wrists. */
  img: [number, number];
  /** The player's take-off spot on the court. */
  spot: TouchCandidate;
}

/**
 * Where the ball was played while hidden: the frame and player at which the
 * incoming path (extrapolated forward) and the outgoing one (extrapolated
 * back) both pass the player's wrists. A reception is played off the
 * forearms, a set off the hands; either way the wrists are at the ball.
 */
function findHiddenContact(
  before: ArcObsEx[],
  after: ArcObsEx[],
  frames: { frame: number; timestamp: number }[],
  playersByFrame: Map<number, PlayerBox[]>,
  armsByFrame: Map<number, PlayerArms[]>,
  cam: THREE.PerspectiveCamera,
  hw: number,
  hh: number,
): HiddenContact | null {
  const a = before.slice(-6), b = after.slice(0, 6);
  if (a.length < 3 || b.length < 3) return null;
  const tA = a[a.length - 1].t, tB = b[0].t;
  if (tB - tA > HIDDEN_CONTACT_MAX_GAP_S) return null;
  const fit = (obs: ArcObsEx[], t0: number) => {
    const ts = obs.map((o) => o.t - t0);
    const cu = _fitQuadratic(ts, obs.map((o) => o.u));
    const cv = _fitQuadratic(ts, obs.map((o) => o.v));
    return (t: number): [number, number] => {
      const s = t - t0;
      return [cu[0] + cu[1] * s + cu[2] * s * s, cv[0] + cv[1] * s + cv[2] * s * s];
    };
  };
  const pathIn = fit(a, tA), pathOut = fit(b, tB);
  // In 1080p pixels, so the sigmas mean the same at any resolution.
  const px = (p: [number, number], q: [number, number]) => Math.hypot((p[0] - q[0]) * hw, (p[1] - q[1]) * hh) * (540 / hh);
  const sigma = (dt: number) => CONTACT_SIGMA_BASE_PX + CONTACT_SIGMA_ACCEL_PX * dt * dt;
  let best: HiddenContact | null = null;
  let bestCost = HIDDEN_CONTACT_MAX_SIGMA;
  for (const fr of frames) {
    if (fr.timestamp <= tA || fr.timestamp >= tB) continue;
    const boxes = playersByFrame.get(fr.frame) ?? [];
    const arms = armsByFrame.get(fr.frame) ?? [];
    const pIn = pathIn(fr.timestamp), pOut = pathOut(fr.timestamp);
    boxes.forEach((box, i) => {
      const wrists = (arms[i] ?? []).slice(4, 6).filter((k) => k[2] >= WRIST_MIN_CONF);
      if (!wrists.length) return;
      // Both wrists: their midpoint (forearms together); one: that wrist.
      const wx = wrists.reduce((s, k) => s + k[0], 0) / wrists.length;
      const wy = wrists.reduce((s, k) => s + k[1], 0) / wrists.length;
      const w: [number, number] = [wx * 2 - 1, 1 - wy * 2];
      const cost = px(pIn, w) / sigma(fr.timestamp - tA) + px(pOut, w) / sigma(tB - fr.timestamp);
      if (cost >= bestCost) return;
      const spot = takeOffSpot(fr.frame, box, playersByFrame, cam);
      if (!spot) return;
      bestCost = cost;
      best = { t: fr.timestamp, frame: fr.frame, img: w, spot };
    });
  }
  return best;
}

/** Frames sampled across a detection gap for players at a hidden touch. */
const HIDDEN_TOUCH_SAMPLES = 5;
/** Margin (m) around the court within which a player is in play. */
const IN_PLAY_MARGIN_M = 1.5;

/**
 * Players behind a baseline under the ball at the start of a rally: the
 * server. Uses every player in the frame, so the server is found even when
 * the ball (high on the toss) is far from them in the picture. Corner line
 * judges stand outside the sidelines and are left out.
 */
function serverCandidates(
  o: ArcObsEx,
  allPlayersByFrame: Map<number, PlayerBox[]>,
  cam: THREE.PerspectiveCamera,
): TouchCandidate[] {
  const bx = (o.u + 1) / 2;
  const out: TouchCandidate[] = [];
  for (const [x1, , x2, y2] of allPlayersByFrame.get(o.fi) ?? []) {
    const w = x2 - x1;
    if (bx < x1 - 1.5 * w || bx > x2 + 1.5 * w) continue;
    const g = _groundPointOf(cam, (x1 + x2) / 2, y2);
    if (!g || Math.abs(g[0]) > COURT_HALF_W + 0.5) continue;
    if (Math.abs(g[1]) < P.serverMinAbsZ || Math.abs(g[1]) > RALLY_MAX_ABS_Z_M) continue;
    out.push(g);
  }
  return out;
}

/** Candidates for a touch between two consecutive observations. */
function boundaryCandidates(
  a: ArcObsEx,
  b: ArcObsEx,
  playersByFrame: Map<number, PlayerBox[]>,
  cam: THREE.PerspectiveCamera,
  reachUp = 0.8,
): TouchCandidate[] {
  const toImg = (o: ArcObsEx): [number, number] => [(o.u + 1) / 2, (1 - o.v) / 2];
  return [
    ...touchCandidates(a.fi, toImg(a), playersByFrame, cam, reachUp),
    ...touchCandidates(b.fi, toImg(b), playersByFrame, cam, reachUp),
  ];
}

/**
 * Reach above the head (body heights) for a rally's first point. A clip
 * usually starts with the serve toss, which goes 2-4 m above the server's
 * head; with the normal 0.8 the server was never found on real broadcast
 * clips, so the volleyball rules never started and no touch got a label.
 */
const TOSS_REACH_UP = 2.5;


/** Image velocity jump (px/s) at the boundary between two pieces of track. */
function _velocityJumpPx(before: ArcObsEx[], after: ArcObsEx[], hw: number, hh: number): number {
  const a = before.slice(-6), b = after.slice(0, 6);
  if (a.length < 2 || b.length < 2) return Infinity; // one sample has no velocity: keep
  const tb = 0.5 * (a[a.length - 1].t + b[0].t);
  // Local quadratic fits either side; perspective is smooth over ~0.2 s.
  const velAt = (obs: ArcObsEx[]) => {
    // Two samples (between an attack and its block, say): their difference.
    if (obs.length === 2) {
      const dt = Math.max(1e-6, obs[1].t - obs[0].t);
      return [((obs[1].u - obs[0].u) / dt) * hw, ((obs[1].v - obs[0].v) / dt) * hh];
    }
    const ts = obs.map((o) => o.t - tb);
    const cu = _fitQuadratic(ts, obs.map((o) => o.u));
    const cv = _fitQuadratic(ts, obs.map((o) => o.v));
    return [cu[1] * hw, cv[1] * hh]; // derivative at t = tb
  };
  const [ua, va] = velAt(a);
  const [ub, vb] = velAt(b);
  return Math.hypot(ub - ua, vb - va);
}

/** Longest time (s) between the two touches of a double touch (attack and block). */
const DOUBLE_TOUCH_MAX_S = 0.3;
/** Samples either side of a split whose ball size says how far away it is, and the fewest needed. */
const SPLIT_DEPTH_SAMPLES = 8;
const SPLIT_DEPTH_MIN_SAMPLES = 4;
/** Slack (m) around the possible touchers' distances. */
const SPLIT_DEPTH_SLACK_M = 1.5;

/**
 * Metres per pixel at a split: the ball's distance there over the focal
 * length. Under a long lens that is the same everywhere on the court; from a
 * tripod behind the end line a pixel is 5 mm at the near baseline and 20 mm
 * at the far one, and one scale for both lost real touches far away and
 * kept jitter close by. The distance comes from the ball's apparent size
 * around the split (rough, but this needs no better), held within the
 * distances of the players who could have made the touch.
 */
function _metresPerPxAt(before: ArcObsEx[], after: ArcObsEx[], cands: TouchCandidate[]): number {
  const ctx = flightCam;
  if (!ctx) return courtMetresPerPx;
  const sizes = [...before.slice(-SPLIT_DEPTH_SAMPLES), ...after.slice(0, SPLIT_DEPTH_SAMPLES)]
    .flatMap((o) => (o.sizeDepth != null && o.sizeDepth > 0 ? [o.sizeDepth] : []));
  let depth = sizes.length >= SPLIT_DEPTH_MIN_SAMPLES ? _median(sizes) : ctx.courtDist;
  if (cands.length) {
    const c = ctx.cam.position;
    const ds = cands.map(([x, z]) => Math.hypot(x - c.x, 1.5 - c.y, z - c.z));
    depth = Math.min(Math.max(...ds) + SPLIT_DEPTH_SLACK_M, Math.max(Math.min(...ds) - SPLIT_DEPTH_SLACK_M, depth));
  }
  return depth / ctx.fPx;
}

/**
 * Drop touch splits that are not real touches. Each split adds a knot, and
 * unexplained knots are what made the path wobble. A split is kept only if a
 * player could have touched the ball there AND the path really changes: from
 * this camera players overlap the ball in the image all the time, so a
 * candidate alone would pin a ball still in flight to whoever is behind it.
 */
function mergeUntouchedSplits(
  arcs: ArcObsEx[][],
  playersByFrame: Map<number, PlayerBox[]>,
  cam: THREE.PerspectiveCamera,
  hw: number,
  hh: number,
  log?: TouchSplitDecision[],
  /**
   * Re-entry frames of flights out of the top of the picture that stay
   * split. The touch there (if any) happened out of view, so neither a
   * nearby player nor a velocity jump can confirm it, and merging would put
   * back the one-arc flight that already failed to fit.
   */
  keepSplit: Set<number> = new Set(),
  /** Kinks kept as the other half of a double touch (see below); set by the second pass. */
  paired: Set<number> = new Set(),
): ArcObsEx[][] {
  const out: ArcObsEx[][] = [];
  const decisions: (TouchSplitDecision & { t: number; fi: number })[] = [];
  for (const arc of arcs) {
    const prev = out[out.length - 1];
    if (prev && arc.length > 0 && !keepSplit.has(arc[0].fi)) {
      const last = prev[prev.length - 1];
      const sameRally = arc[0].t - last.t <= P.rallyMaxGapS;
      if (sameRally) {
        const cands = boundaryCandidates(last, arc[0], playersByFrame, cam);
        const candidates = cands.length;
        const dvMps = _velocityJumpPx(prev, arc, hw, hh) * _metresPerPxAt(prev, arc, cands);
        // A scanned kink is already a break no single flight explains, far
        // above the noise, so it needs less of a velocity jump than a split
        // at a detection gap, which has no such evidence. The jump is an
        // estimate from a few samples and is small for a real touch that
        // sends the ball along the line of sight (an attack seen from
        // behind: 2-3 m/s). It is not waived: where the ball passes the net
        // and the players the detector drifts by a few pixels, a real break
        // in the track of 0.5-1.2 m/s (France vs Japan).
        // Nobody in reach is normally the end of it, but a break this sharp
        // and this large is a touch by someone the detector missed (a
        // receiver hidden behind a team-mate).
        const certain = (arc[0].kink ?? 0) >= P.touchKinkCertain && dvMps >= 2 * P.touchMinDvMps;
        const kept = certain || (candidates > 0 &&
          (paired.has(arc[0].fi) || dvMps >= (arc[0].kink != null ? P.touchMinDvKinkMps : P.touchMinDvMps)));
        decisions.push({ frameBefore: last.fi, frameAfter: arc[0].fi, dvMps, candidates, kept, kink: arc[0].kink, t: 0.5 * (last.t + arc[0].t), fi: arc[0].fi });
        if (!kept) {
          out[out.length - 1] = [...prev, ...arc];
          continue;
        }
      }
    }
    out.push(arc);
  }
  // Double touches. An attack into the block is two touches a few frames
  // apart, and from behind the court one of them sends the ball almost
  // along the line of sight: a clear kink with next to no velocity change
  // in the picture. A kink with a player in reach right next to a kept
  // touch is taken as its other half. (Two weak kinks next to each other
  // stay dropped: neither has a touch to pair with.)
  const promote = decisions.filter((d) => !d.kept && d.kink != null && d.candidates > 0 &&
    decisions.some((o) => o.kept && Math.abs(o.t - d.t) <= DOUBLE_TOUCH_MAX_S));
  if (promote.length && paired.size === 0) {
    return mergeUntouchedSplits(arcs, playersByFrame, cam, hw, hh, log, keepSplit, new Set(promote.map((d) => d.fi)));
  }
  for (const { t: _t, fi: _fi, ...d } of decisions) log?.push(d);
  return out;
}

/** A split this many frames from a hand-marked touch is taken to be that touch. */
const EDIT_TOUCH_SNAP_FRAMES = 3;
/**
 * Residual px per metre between a knot and a hand-placed 3D position. Far
 * above the player anchor (anchorWeight): the user has seen where it was.
 */
const PIN_WEIGHT = 200;

/**
 * Split the track at hand-marked touches. A split the solver already made
 * near one is reused rather than doubled. Returns the arcs and the first
 * frame of each arc that starts at a marked touch.
 */
function splitAtMarkedTouches(arcs: ArcObsEx[][], touchAt: number[]): { arcs: ArcObsEx[][]; starts: Set<number> } {
  const starts = new Set<number>();
  if (!touchAt.length) return { arcs, starts };
  let out = arcs;
  for (const f of touchAt) {
    // Near an existing split, or inside the detection gap it spans.
    const near = out.find((a, i) => i > 0 && a.length > 0 && out[i - 1].length > 0 &&
      f >= out[i - 1][out[i - 1].length - 1].fi - EDIT_TOUCH_SNAP_FRAMES && f <= a[0].fi + EDIT_TOUCH_SNAP_FRAMES);
    if (near) { starts.add(near[0].fi); continue; }
    const next: ArcObsEx[][] = [];
    for (const a of out) {
      const i = a.findIndex((o) => o.fi >= f);
      if (i > 0) {
        next.push(a.slice(0, i), a.slice(i));
        starts.add(a[i].fi);
      } else next.push(a);
    }
    out = next;
  }
  return { arcs: out, starts };
}

/** Merge splits within a few frames of a frame where a touch was removed by hand. */
function mergeAtRemovedTouches(arcs: ArcObsEx[][], noTouchAt: number[], keep: Set<number>): ArcObsEx[][] {
  if (!noTouchAt.length) return arcs;
  const out: ArcObsEx[][] = [];
  for (const arc of arcs) {
    const prev = out[out.length - 1];
    if (prev && arc.length && !keep.has(arc[0].fi)) {
      const a = prev[prev.length - 1].fi, b = arc[0].fi;
      if (arc[0].t - prev[prev.length - 1].t <= P.rallyMaxGapS &&
          noTouchAt.some((f) => f >= a - EDIT_TOUCH_SNAP_FRAMES && f <= b + EDIT_TOUCH_SNAP_FRAMES)) {
        out[out.length - 1] = [...prev, ...arc];
        continue;
      }
    }
    out.push(arc);
  }
  return out;
}

/** Why a candidate touch split was kept or merged (diagnostics, see BallTrajectoryResult). */
export interface TouchSplitDecision {
  frameBefore: number;
  frameAfter: number;
  /** Jump in the ball's velocity at the split, m/s at the court. */
  dvMps: number;
  /** Players who could have touched the ball there. */
  candidates: number;
  kept: boolean;
  /** How clearly the track breaks there, in noise units; absent for a split at a detection gap. */
  kink?: number;
}

/** A touch rule placed on one knot: its type, its side of the net, and its limits. */
interface KnotZone extends TouchRule {
  label: string;
  /** +1 / -1: the half of the court (sign of z) the touch is on. */
  side: number;
  /**
   * A touch that can be one of several kinds (after an attack: a block at
   * the net or a dig in the back court). The zone spans them all; the label
   * is the first alternative the solved touch lies in (see zoneLabel).
   */
  alternatives?: (TouchRule & { label: string })[];
}

/**
 * Shortest time between two touches (s), except for the block or dig after
 * an attack. Labelled clips: the shortest is serve to reception, 0.56 s.
 */
const MIN_TOUCH_INTERVAL_S = 0.52;
/**
 * Longest time (s) the ball is carried up in the server's hand before the
 * toss leaves it. The shortest toss flight, a float serve's, is ~0.5 s.
 */
const TOSS_CARRY_MAX_S = 0.3;
/** A rally's first point this far (m) above the highest toss release is a ball already in the air. */
const TOSS_SEEN_SLACK_M = 0.3;
/** Distance from the net (m) beyond which a player on the serving side at the serve contact is the server, feet in the air. */
const SERVER_AIRBORNE_MIN_ABS_Z = 5;
/** Longest time (s) from an attack to its block. */
const BLOCK_MAX_AFTER_ATTACK_S = 0.4;
/** How far over the net (m) a blocker's hands reach into the attacker's side. */
const BLOCK_REACH_OVER_M = 0.75;

/** Does a court z fall in a rule's distance range (with slack, m)? */
function _inZoneZ(zone: TouchRule, side: number, z: number, slack: number, extraBack = 0): boolean {
  const into = side * z;
  return into >= zone.minAbsZ - slack && into <= zone.maxAbsZ + slack + extraBack;
}

/**
 * Label of a solved touch in a zone: its kind, or which alternative it lies
 * in. By distance from the net only: the alternatives do not overlap there
 * (a block is at the net, a dig never is), while the solved height of a
 * touch is the least reliable coordinate.
 */
function zoneLabel(zone: KnotZone, [, , z]: [number, number, number], sincePrevS = 0): string {
  if (!zone.alternatives) return zone.label;
  // A block is the attack coming straight back off the hands: the ball is
  // there within a few frames. A touch at the net longer after the last one
  // is a dig of a tip or of a ball off the tape (usa_jap_r3: 0.6 s).
  if (sincePrevS > BLOCK_MAX_AFTER_ATTACK_S) {
    const late = zone.alternatives.find((a) => a.label !== "block");
    if (late) return late.label;
  }
  // Blockers reach over the net: a block can be solved just past it.
  const hit = zone.alternatives.find((a) => _inZoneZ(a, zone.side, z, BLOCK_REACH_OVER_M));
  return (hit ?? zone.alternatives[zone.alternatives.length - 1]).label;
}

/** One zone covering several touch kinds, for a touch that can be any of them. */
function _eitherZone(side: number, alts: (TouchRule & { label: string })[]): KnotZone {
  return {
    label: alts.map((a) => a.label).join("/"), side, alternatives: alts,
    minAbsZ: Math.min(...alts.map((a) => a.minAbsZ)), maxAbsZ: Math.max(...alts.map((a) => a.maxAbsZ)),
    minY: Math.min(...alts.map((a) => a.minY)), maxY: Math.max(...alts.map((a) => a.maxY)),
    minG: Math.min(...alts.map((a) => a.minG)), maxG: Math.max(...alts.map((a) => a.maxG)),
  };
}


/**
 * How far behind the contact an attacker takes off (m). Measured on the
 * labelled clips: take-off 1.6-2.4 m from the net for contacts ~0.5-1 m out.
 */
const ATTACK_RUNUP_M = 1.5;
/** Distance from the net (m) beyond which a receiver is preferred: 3 m line + 1 m. */
const RECEPTION_BACK_Z_M = 4;

/**
 * Volleyball rules for one rally: which touch is which, and where it can be.
 *
 * A rally that starts with a player behind a baseline is a serve, and then
 * the order is fixed: toss and contact by the server, reception on the other
 * side (in the back court - even a short serve lands 3-4 m from the net, the
 * players at the net are blockers and the setter), set near the net on the
 * same side, attack on that side (front or back row), then a block at the net
 * by the serving team. Receptions and sets can both be overhead, so hand
 * height is not used. Later touches are left free.
 *
 * These matter because the ball alone cannot say how deep it is from behind
 * the baseline, and players overlap the ball in the image all the time: ball
 * size is biased while the ball passes the crowd, and without the rules a
 * serve gets "received" by whoever stands under it at the net.
 *
 * Returns one zone per knot (null = no rule) and filters `anchors` in place
 * to the candidates that satisfy their zone.
 */
function applyRallyRules(
  anchors: TouchCandidate[][],
  /**
   * Knots that are not touches (the ball was above the picture there).
   * Knots dropped as too soon after the previous touch are added to it.
   */
  skip: Set<number> = new Set(),
  /** Time of each knot (s), to drop splits too soon after a touch. */
  knotT: number[] = [],
  /** Where the previous rally's touch sequence stopped (see RulesState). */
  resume: RulesState | null = null,
  /** Receives where this rally's sequence stops. */
  out: { state?: RulesState } = {},
): (KnotZone | null)[] {
  const zones: (KnotZone | null)[] = anchors.map(() => null);
  const isServer = ([, z]: TouchCandidate) => Math.abs(z) >= P.serverMinAbsZ - 1;
  const servers0 = (anchors[0] ?? []).filter(isServer);
  const servers1 = anchors.length > 2 ? (anchors[1] ?? []).filter(isServer) : [];
  // Does the rally start with the toss, or does the clip begin at the serve
  // contact? A toss goes up and comes down over the same server, so the
  // same player stands at the first and the second point.
  // A jump server is in the air at the contact, and feet off the floor look
  // further away than they are: from behind the court, metres inside the
  // baseline (France vs Japan: 3 m). Nobody else on the serving side plays
  // the ball between toss and serve, so a player in that back court counts.
  const airborne1 = anchors.length > 2
    ? (anchors[1] ?? []).filter(([, z]) => Math.abs(z) >= SERVER_AIRBORNE_MIN_ABS_Z && servers0.some(([, z0]) => Math.sign(z0) === Math.sign(z)))
    : [];
  const sameServer = servers0.some(([x0, z0]) => servers1.some(([x1, z1]) => Math.hypot(x0 - x1, z0 - z1) < 2.5)) || airborne1.length > 0;
  let startsWithToss: boolean;
  let S: number;
  let step = 0;
  if (servers0.length && (sameServer || !servers1.length)) {
    startsWithToss = sameServer;
    S = Math.sign(servers0[0][1]);
  } else if (servers1.length) {
    // Server only seen at the contact: the first point is the toss.
    startsWithToss = true;
    S = Math.sign(servers1[0][1]);
  } else if (resume && knotT[0] - resume.lastT <= RULES_CARRY_S) {
    // No serve here: the play goes on from the previous piece of track,
    // which ended at a detection gap. Touches may have been missed in it,
    // so skip ahead to the next one on the side of the net where the
    // players at the first touch stand.
    ({ S, startsWithToss, step } = resume);
    const first = anchors.findIndex((a, k) => a.length > 0 && !skip.has(k));
    if (first >= 0) {
      const side = Math.sign(anchors[first].reduce((sum, [, z]) => sum + Math.sign(z), 0));
      const probe = rallyPlan(S, startsWithToss, step + 4);
      for (let i = 0; i < 3 && side && probe[step].side !== side; i++) step++;
    }
  } else {
    return zones;
  }
  // The ball is often seen in the server's hand before the toss: it rises
  // with the hand, then leaves it, and that release is a break in the path
  // like any touch (pol_ita_r4: first seen at frame 18, released at 21-22).
  // The release is the toss; the first sight of the ball is not a touch.
  if (startsWithToss && step === 0 && anchors.length > 2 && knotT.length > 1 && !skip.has(0) && !skip.has(1) &&
      knotT[1] - knotT[0] <= TOSS_CARRY_MAX_S && (anchors[1] ?? []).some(isServer)) {
    skip.add(0);
  }
  const plan = rallyPlan(S, startsWithToss, step + anchors.length);
  // The last knot is where the ball leaves view, not a touch.
  let lastT = -Infinity;
  const firstStep = step;
  for (let k = 0; step < plan.length && k < anchors.length - 1; k++) {
    if (skip.has(k)) continue;
    // A resumed rally's first point is only where the ball came back into
    // view, unless a player is there to have played it.
    if (k === 0 && resume && step === firstStep && !servers0.length && !servers1.length && !anchors[0].length) continue;
    // A split too soon after the previous touch is not the next one (a
    // reception and a "set" 0.28 s apart, say): it would shift every later
    // touch's rule by one. Only the block or dig follows an attack that fast.
    // The toss is no interval: its knot is wherever the clip first sees the
    // ball, already on its way up.
    if (step > firstStep && plan[step - 1].label !== "toss" && !plan[step].alternatives &&
        knotT[k] - lastT < MIN_TOUCH_INTERVAL_S) {
      skip.add(k);
      continue;
    }
    lastT = knotT[k] ?? lastT;
    // Re-sync by side of the net: players at this touch all on the other
    // side from the expected one means a touch was missed (hidden, or in a
    // detection gap). Jump to the next expected touch on their side.
    const side = Math.sign(anchors[k].reduce((sum, [, z]) => sum + Math.sign(z), 0));
    if (side && anchors[k].every(([, z]) => Math.sign(z) === side) && plan[step].side !== side) {
      for (let i = 1; i <= 2 && step + i < plan.length; i++) {
        if (plan[step + i].side === side) { step += i; break; }
      }
    }
    const zone = plan[step++];
    zones[k] = zone;
    // An attacker is anchored at the take-off spot, and the approach jump
    // carries them forward to the contact: allow that run-up behind the zone.
    const runUp = zone.label === "attack" ? ATTACK_RUNUP_M : 0;
    let kept = anchors[k].filter(([, z]) => zone.alternatives
      ? zone.alternatives.some((a) => _inZoneZ(a, zone.side, z, 0.5))
      : _inZoneZ(zone.reachAbsZ ? { ...zone, maxAbsZ: zone.reachAbsZ } : zone, zone.side, z, 0.5, runUp));
    if (zone.reachAbsZ) {
      // Someone in the usual zone is the likelier toucher.
      const usual = kept.filter(([, z]) => _inZoneZ(zone, zone.side, z, 0.5));
      if (usual.length) kept = usual;
      // Only someone beyond it: stretch the zone to them.
      const far = kept.length ? Math.max(...kept.map(([, z]) => zone.side * z)) : 0;
      if (far > zone.maxAbsZ) zones[k] = { ...zone, maxAbsZ: far + 0.5 };
    }
    // Serve receive is played from the back court: the front-row players
    // stand at the net to attack and block. Prefer anyone behind the 3 m line.
    if (zone.label === "reception") {
      const back = kept.filter(([, z]) => zone.side * z >= RECEPTION_BACK_Z_M);
      if (back.length) kept = back;
    }
    // No candidate fits: the toucher was not detected (an attacker hidden
    // behind the block, say). Use the zone alone rather than pull the knot
    // to someone who cannot have played the ball.
    anchors[k] = kept;
  }
  out.state = { S, startsWithToss, step, lastT: knotT[knotT.length - 1] ?? -Infinity };
  return zones;
}

/**
 * The touch sequence of a rally served from side S, at least `length` long:
 * toss (when seen), serve, reception, set, attack, then block or dig, set,
 * attack... with the sides alternating.
 */
function rallyPlan(S: number, startsWithToss: boolean, length: number): KnotZone[] {
  const R = P.rules;
  const plan: KnotZone[] = [
    ...(startsWithToss ? [{ label: "toss", side: S, ...R.toss }] : []),
    { label: "serve",     side: S,  ...R.serve },
    { label: "reception", side: -S, ...R.reception },
    { label: "set",       side: -S, ...R.set },
    { label: "attack",    side: -S, ...R.attack },
    // The attack is blocked at the net or dug in the back court: forcing a
    // block threw away the digger and let the knot (and the attack before
    // it) drift to the wrong depth.
    _eitherZone(S, [{ label: "block", ...R.block }, { label: "dig", ...R.dig }]),
  ];
  // The rally goes on in the same cycle, the sides alternating: the
  // defending team digs (or blocks), sets and attacks, then the other side
  // defends. Labelled long rallies follow it touch for touch.
  for (let side = S; plan.length < length; side = -side) {
    plan.push(
      { label: "set",    side, ...R.set },
      { label: "attack", side, ...R.attack },
      _eitherZone(-side, [{ label: "block", ...R.block }, { label: "dig", ...R.dig }]),
    );
  }
  return plan;
}

/** Where a rally's touch sequence stopped, to resume it after a gap. */
interface RulesState {
  S: number;
  startsWithToss: boolean;
  /** Index of the next touch in rallyPlan. */
  step: number;
  /** Time of the rally's last point (s). */
  lastT: number;
}
/** Longest gap (s) across which the touch sequence carries on. */
const RULES_CARRY_S = 4;

/**
 * Fit one rally as connected ballistic segments (see block comment above).
 * Knot k is the ball position at time tau[k]; segment k runs from knot k to k+1.
 */
function solveRally(
  segs: ArcObsEx[][],
  cam: THREE.PerspectiveCamera,
  videoWidth: number,
  videoHeight: number,
  /** Per knot (n+1 entries): court (x, z) of players who could have touched it. */
  anchors: TouchCandidate[][] = [],
  /** Per knot: volleyball-rule zone (side of the net, distance from it). */
  zones: (KnotZone | null)[] = [],
  /** Ball image position (NDC) at a time, incl. gap-filled frames, if known. */
  imageAt: (t: number) => [number, number] | null = () => null,
  /**
   * Knots that are not touches but the ball above the picture: no height
   * cap there, and no image target (the gap fill is a straight line that
   * cuts under the arc).
   */
  freeKnots: Set<number> = new Set(),
  /** Per knot: a contact found at a player's arms (time and image point). */
  contacts: (HiddenContact | null)[] = [],
  /** Per knot: a 3D position placed by hand, held by a stiff pull. */
  pins: ([number, number, number] | null)[] = [],
): RallySegmentResult[] {
  const hw = videoWidth / 2, hh = videoHeight / 2;
  const project = _makeProjector(cam);
  const n = segs.length;

  // Touch time = midway between the last sample of one segment and the first
  // of the next; the rally starts/ends at its first/last sample.
  const tau: number[] = [segs[0][0].t];
  for (let k = 1; k < n; k++) tau.push(0.5 * (segs[k - 1][segs[k - 1].length - 1].t + segs[k][0].t));
  const lastSeg = segs[n - 1];
  tau.push(lastSeg[lastSeg.length - 1].t);
  // A one-sample segment would have zero duration; give it a frame.
  for (let k = 1; k <= n; k++) if (tau[k] - tau[k - 1] < 1e-3) tau[k] = tau[k - 1] + 1 / 30;
  // A contact seen at the arms dates the touch better than mid-gap.
  for (let k = 1; k < n; k++) if (contacts[k]) tau[k] = contacts[k]!.t;

  // Parameter layout: 3 per knot, then one effective downward acceleration
  // per segment (see ACCEL_*), then the size log-scale when sizes exist.
  const GK = 3 * (n + 1);
  // Arc through both knots under gravity and air drag. Drag is linearised
  // (deceleration k·v) so the arc stays closed-form; k comes from the
  // segment's typical speed, as real drag grows with speed (see DRAG_PER_M).
  // Without it a long flight (a high pass out of the picture and back) had
  // no drag-free arc matching both ends.
  const segState = (p: number[], k: number, s: number): { pos: [number, number, number]; vel: [number, number, number] } => {
    const T = tau[k + 1] - tau[k];
    const g = p[GK + k];
    const d = [p[3 * k + 3] - p[3 * k], p[3 * k + 4] - p[3 * k + 1], p[3 * k + 5] - p[3 * k + 2]];
    const kd = P.dragPerM * Math.hypot(Math.hypot(d[0], d[1], d[2]) / T, 0.25 * g * T);
    let w: number, dw: number, sag: number, dsag: number;
    if (kd * T < 1e-4) {
      // Drag-free limit: linear in x/z, plus gravity sag in y.
      w = s / T; dw = 1 / T;
      sag = 0.5 * g * s * (T - s); dsag = 0.5 * g * (T - 2 * s);
    } else {
      const E = (u: number) => (1 - Math.exp(-kd * u)) / kd;
      const ET = E(T);
      w = E(s) / ET; dw = Math.exp(-kd * s) / ET;
      sag = (g / kd) * (T * w - s); dsag = (g / kd) * (T * dw - 1);
    }
    return {
      pos: [p[3 * k] + d[0] * w, p[3 * k + 1] + d[1] * w + sag, p[3 * k + 2] + d[2] * w],
      vel: [d[0] * dw, d[1] * dw + dsag, d[2] * dw],
    };
  };
  const segPos = (p: number[], k: number, t: number) => segState(p, k, t - tau[k]).pos;

  const allObs = segs.flatMap((seg, k) => seg.map((o) => ({ o, k })));
  const active = allObs.map(() => true);
  // At a contact the ball is at the wrists; elsewhere use the image track
  // (in a gap, the analyzer's straight-line fill).
  const knotTargets = tau.map((t, k) => contacts[k]?.img ?? imageAt(t));
  // Size-depth terms are rejected separately: a bad segmentation (ball merged
  // with a yellow shirt, say) should not also throw away a good 2D position.
  // Size fixes absolute depth; it is the only cue that does, since gravity's
  // scale is unreliable for a spinning ball (see ACCEL_*).
  const sizeActive = allObs.map(({ o }) => o.sizeDepth != null && o.sizeDepth > 0);
  const hasSize = sizeActive.some(Boolean);
  const camFwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
  const camDepthOf = (x: number, y: number, z: number) =>
    (x - cam.position.x) * camFwd.x + (y - cam.position.y) * camFwd.y + (z - cam.position.z) * camFwd.z;
  const sizeResidual = (p: number[], i: number): number | null => {
    const { o, k } = allObs[i];
    const [x, y, z] = segPos(p, k, o.t);
    const d = camDepthOf(x, y, z);
    if (!(d > 0.5)) return null;
    return Math.log(d) - Math.log(o.sizeDepth!);
  };

  // Parameters each term depends on: a knot's position, or a segment (its
  // two knots and its acceleration).
  const knotP = (k: number) => [3 * k, 3 * k + 1, 3 * k + 2];
  const segP = (k: number) => [...knotP(k), ...knotP(k + 1), GK + k];
  // Soft priors are in "pixels per metre of violation". They only bite
  // outside the plausible region, so they select among depths that fit the
  // image equally well rather than fighting the observations.
  const PRIOR = 20;
  // Built per solve: outlier rejection changes which observations are active.
  const residualBlocks = (): ResidualBlock[] => {
    const blocks: ResidualBlock[] = [];
    allObs.forEach(({ o, k }, i) => {
      if (!active[i]) return;
      blocks.push({ params: segP(k), eval: (p) => {
        const [x, y, z] = segPos(p, k, o.t);
        const [pu, pv] = project(x, y, z);
        const w = Math.sqrt(Math.max(0.1, o.conf));
        const du = (pu - o.u) * hw, dv = (pv - o.v) * hh;
        // Pseudo-Huber: quadratic up to ~P.robustReprojPx, linear beyond. A
        // messy stretch (a deflection off a block) has 20+ px errors, and with
        // a plain square those few samples dominate the whole rally and drag
        // well-anchored touches off their players.
        const e = Math.hypot(du, dv);
        const sc = e > 1e-9
          ? Math.sqrt(2 * P.robustReprojPx ** 2 * (Math.sqrt(1 + (e / P.robustReprojPx) ** 2) - 1)) / e
          : 1;
        return [w * du * sc, w * dv * sc];
      } });
    });
    // Apparent-size depth prior (log depth, so it is scale-free).
    if (hasSize) {
      for (let i = 0; i < allObs.length; i++) {
        if (!sizeActive[i]) continue;
        blocks.push({ params: segP(allObs[i].k), eval: (p) => {
          const e = sizeResidual(p, i);
          return [e == null ? P.sizeDepthWeight : P.sizeDepthWeight * e];
        } });
      }
    }
    for (let k = 0; k <= n; k++) {
      blocks.push({ params: knotP(k), eval: (p) => {
        const [x, y, z] = [p[3 * k], p[3 * k + 1], p[3 * k + 2]];
        return [
          PRIOR * Math.max(0, KNOT_MIN_Y_M - y, freeKnots.has(k) ? 0 : y - P.knotMaxY),
          PRIOR * Math.max(0, Math.abs(x) - RALLY_MAX_ABS_X_M),
          PRIOR * Math.max(0, Math.abs(z) - RALLY_MAX_ABS_Z_M),
        ];
      } });
    }
    // Effective gravity stays within what spin and drag can produce.
    for (let k = 0; k < n; k++) {
      const lo = zones[k] ? zones[k]!.minG * GRAVITY : (P.accelMinG * GRAVITY);
      const hi = zones[k] ? zones[k]!.maxG * GRAVITY : (P.accelMaxG * GRAVITY);
      blocks.push({ params: [GK + k], eval: (p) => [PRIOR * Math.max(0, lo - p[GK + k], p[GK + k] - hi)] });
    }
    // Speed prior. Depth along the viewing ray is the weakly observed
    // direction, and without this the knots zig-zag in depth (z = 17, 5, 17...)
    // with segments flying at 50-160 m/s, which still projects onto the image
    // track. Nothing in volleyball moves faster than a hard jump serve.
    for (let k = 0; k < n; k++) {
      blocks.push({ params: segP(k), eval: (p) => {
        // Fastest point of the arc is one of its ends.
        const vEnd = Math.max(
          Math.hypot(...segState(p, k, 0).vel),
          Math.hypot(...segState(p, k, tau[k + 1] - tau[k]).vel),
        );
        return [PRIOR * Math.max(0, vEnd - P.softMaxSpeed)];
      } });
    }
    // Touch impulse prior. Measured on broadcast footage, the reprojection
    // error of a short (<0.4 s) segment changes by ~0.1 px across 30 m of
    // depth: the image simply does not say where it is. A weak preference for
    // small velocity changes at knots lets such segments follow their
    // well-determined neighbours instead of zig-zagging, while a real touch
    // (10-40 m/s of change) is still far cheaper than misfitting its samples.
    for (let k = 1; k < n; k++) {
      blocks.push({ params: [...knotP(k - 1), ...knotP(k), ...knotP(k + 1), GK + k - 1, GK + k], eval: (p) => {
        const vIn = segState(p, k - 1, tau[k] - tau[k - 1]).vel;
        const vOut = segState(p, k, 0).vel;
        return [0, 1, 2].map((c) => P.touchImpulseWeight * (vOut[c] - vIn[c]));
      } });
    }
    // Player anchors: pull each knot towards the nearest candidate toucher.
    for (let k = 0; k <= n; k++) {
      const cands = anchors[k] ?? [];
      blocks.push({ params: knotP(k), eval: (p) => {
        if (cands.length === 0) return [0];
        let d = Infinity;
        for (const [ax, az] of cands) d = Math.min(d, Math.hypot(p[3 * k] - ax, p[3 * k + 2] - az));
        return [P.anchorWeight * d / Math.sqrt(1 + (d / P.anchorRobustM) ** 2)];
      } });
    }
    // Touch points must project onto the ball. A touch often falls in a short
    // detection gap (the ball is hidden by the player playing it), and without
    // this the two arcs meet wherever their extrapolations cross: at a set
    // that was a 0.4 m dip below the net tape, 50 px off the ball.
    for (let k = 0; k <= n; k++) {
      const target = freeKnots.has(k) ? null : knotTargets[k];
      blocks.push({ params: knotP(k), eval: (p) => {
        if (!target) return [0, 0];
        const [pu, pv] = project(p[3 * k], p[3 * k + 1], p[3 * k + 2]);
        return [P.knotImageWeight * (pu - target[0]) * hw, P.knotImageWeight * (pv - target[1]) * hh];
      } });
    }
    // Hand-placed positions.
    for (let k = 0; k <= n; k++) {
      const pin = pins[k];
      if (!pin) continue;
      blocks.push({ params: knotP(k), eval: (p) => [0, 1, 2].map((c) => PIN_WEIGHT * (p[3 * k + c] - pin[c])) });
    }
    // Rule zones: the side of the net and distance from it for known touches.
    // A hand-placed touch is where it is, whatever the rules expect.
    for (let k = 0; k <= n; k++) {
      const zone = pins[k] ? null : zones[k];
      blocks.push({ params: knotP(k), eval: (p) => {
        if (!zone) return [0, 0];
        const into = zone.side * p[3 * k + 2];
        const y = p[3 * k + 1];
        return [
          PRIOR * Math.max(0, zone.minAbsZ - into, into - zone.maxAbsZ),
          PRIOR * Math.max(0, zone.minY - y, y - zone.maxY),
        ];
      } });
    }
    // Net: a segment crossing the net plane between the antennae must clear it.
    for (let k = 0; k < n; k++) {
      blocks.push({ params: segP(k), eval: (p) => {
        const z0 = p[3 * k + 2], z1 = p[3 * k + 5];
        let viol = 0;
        if (z0 * z1 < 0) {
          const tc = tau[k] + (z0 / (z0 - z1)) * (tau[k + 1] - tau[k]);
          const [xc, yc] = segPos(p, k, tc);
          if (Math.abs(xc) <= COURT_HALF_W + 0.5) viol = Math.max(0, NET_CLEAR_M - yc);
        }
        return [PRIOR * viol];
      } });
    }
    return blocks;
  };

  // Image position at each knot time, from the nearest sample.
  const knotUV = tau.map((t) => {
    let best = allObs[0].o;
    for (const { o } of allObs) if (Math.abs(o.t - t) < Math.abs(best.t - t)) best = o;
    return [best.u, best.v];
  });
  // Seed: every knot on its viewing ray at a common touch height. Several
  // heights are tried because the depth along the ray is the unknown.
  const seedAtHeight = (h: number): number[] => [...knotUV.flatMap(([u, v]) => {
    const d = new THREE.Vector3(u, v, 0.5).unproject(cam).sub(cam.position).normalize();
    let t = Math.abs(d.y) > 1e-6 ? (h - cam.position.y) / d.y : 50;
    if (!(t > 0)) t = 50;
    // Keep the seed inside the playing area, along x and z alike: from the
    // side, overshooting the court happens across its width, not its length.
    const c = cam.position;
    for (const [axis, lim] of [["x", RALLY_MAX_ABS_X_M], ["z", RALLY_MAX_ABS_Z_M]] as const) {
      const at = c[axis] + d[axis] * t;
      if (Math.abs(at) > lim && Math.abs(d[axis]) > 1e-9) {
        const tb = (Math.sign(at) * lim - c[axis]) / d[axis];
        if (tb > 0 && tb < t) t = tb;
      }
    }
    const p = c.clone().addScaledVector(d, t);
    return [p.x, p.y, p.z];
  }), ...new Array<number>(n).fill(GRAVITY)];

  const cost = (p: number[]) => _blockResiduals(residualBlocks(), p).reduce((s, v) => s + v * v, 0);
  let best: number[] | null = null;
  let bestCost = Infinity;
  // Anchored seed: each touch starts where its most likely player stands, on
  // the knot's viewing ray. The anchor pull is capped (robust), so from a
  // height seed metres away it is too weak to drag the knot across.
  const anchoredSeed = (): number[] | null => {
    if (!anchors.some((a) => a.length > 0)) return null;
    const base = seedAtHeight(2.0);
    knotUV.forEach(([u, v], k) => {
      const cands = anchors[k] ?? [];
      if (cands.length === 0) return;
      const d = new THREE.Vector3(u, v, 0.5).unproject(cam).sub(cam.position).normalize();
      // Candidate closest to the ray in the floor plane; the knot goes to the
      // point of the ray above that player.
      let bestP: THREE.Vector3 | null = null;
      let bestD = Infinity;
      // Point of the ray passing closest to the player, measured on the
      // floor plane: works whichever way the camera looks along the court.
      const dh2 = d.x * d.x + d.z * d.z;
      if (dh2 < 1e-9) return;
      for (const [ax, az] of cands) {
        const t = ((ax - cam.position.x) * d.x + (az - cam.position.z) * d.z) / dh2;
        if (!(t > 0)) continue;
        const q = cam.position.clone().addScaledVector(d, t);
        const off = Math.hypot(q.x - ax, q.z - az);
        if (off < bestD) { bestD = off; bestP = q; }
      }
      if (bestP) {
        base[3 * k] = bestP.x;
        base[3 * k + 1] = Math.min(P.knotMaxY, Math.max(0.2, bestP.y));
        base[3 * k + 2] = bestP.z;
      }
    });
    return base;
  };
  const seeds = [0.5, 1.5, 2.5, 3.5].map(seedAtHeight);
  const anchored = anchoredSeed();
  if (anchored) seeds.push(anchored);
  // Every seed starts the hand-placed knots where they were placed.
  for (const seed of seeds) pins.forEach((pin, k) => { if (pin) seed.splice(3 * k, 3, ...pin); });
  const blocks = residualBlocks();
  for (const seed of seeds) {
    const sol = _levenbergMarquardt(seed, blocks);
    const c = cost(sol);
    if (c < bestCost) { bestCost = c; best = sol; }
  }
  let sol = best!;

  // One round of outlier rejection, then refine from the current solution.
  const errOf = (p: number[], o: ArcObsEx, k: number) => {
    const [x, y, z] = segPos(p, k, o.t);
    const [pu, pv] = project(x, y, z);
    return Math.hypot((pu - o.u) * hw, (pv - o.v) * hh);
  };
  let anyOutlier = false;
  allObs.forEach(({ o, k }, i) => {
    if (errOf(sol, o, k) > P.rallyOutlierPx) { active[i] = false; anyOutlier = true; }
    if (sizeActive[i]) {
      const e = sizeResidual(sol, i);
      if (e == null || Math.abs(e) > P.sizeDepthOutlier) { sizeActive[i] = false; anyOutlier = true; }
    }
  });
  if (anyOutlier) sol = _levenbergMarquardt(sol, residualBlocks());
  if (hasSize) {
    const used = sizeActive.filter(Boolean).length;
    console.log(`[physicsReproj] rally size-depth: ${used}/${allObs.length} sizes used`);
  }

  // Per-segment results and physical validation.
  let idx = 0;
  return segs.map((seg, k) => {
    const T = tau[k + 1] - tau[k];
    const speedMps = Math.hypot(...segState(sol, k, 0).vel);
    const errPx = seg.map((o) => errOf(sol, o, k));
    const outlier = seg.map(() => !active[idx++]);
    const inl = errPx.filter((_, i) => !outlier[i]);
    const medErr = inl.length ? _median(inl) : Infinity;
    const inlierFrac = inl.length / Math.max(1, seg.length);
    // Sample the arc to check its extent, apex included.
    const samples = Array.from({ length: 11 }, (_, i) => segPos(sol, k, tau[k] + (T * i) / 10));
    const ys = samples.map((q) => q[1]);
    const maxAbsX = Math.max(...samples.map((q) => Math.abs(q[0])));
    const maxAbsZ = Math.max(...samples.map((q) => Math.abs(q[2])));

    let rejectReason: string | null = null;
    if (inl.length < 2)                           rejectReason = `too-few-inliers (${inl.length})`;
    else if (Math.min(...ys) < MIN_FIT_Y_M - 0.2) rejectReason = `y-below-ground (min=${Math.min(...ys).toFixed(2)}m)`;
    else if (Math.max(...ys) > SEG_MAX_Y_M)       rejectReason = `y-too-high (max=${Math.max(...ys).toFixed(2)}m)`;
    else if (maxAbsX > RALLY_MAX_ABS_X_M + 1)     rejectReason = `x-out-of-bounds (${maxAbsX.toFixed(1)}m)`;
    else if (maxAbsZ > RALLY_MAX_ABS_Z_M + 1)     rejectReason = `z-out-of-bounds (${maxAbsZ.toFixed(1)}m)`;
    else if (speedMps > MAX_BALL_SPEED_MPS)       rejectReason = `speed-too-high (${speedMps.toFixed(1)} m/s)`;
    else if (medErr > P.segMaxMedianErrPx)      rejectReason = `median-err-too-high (${medErr.toFixed(1)}px)`;
    else if (inlierFrac < P.segMinInlierFrac)    rejectReason = `too-many-outliers (${(100 * inlierFrac).toFixed(0)}% inliers)`;

    return {
      obs: seg, tStart: tau[k], tEnd: tau[k + 1],
      posAt: (t: number) => segPos(sol, k, Math.min(tau[k + 1], Math.max(tau[k], t))),
      speedMps, accepted: rejectReason === null, rejectReason, errPx, outlier,
    };
  });
}

// ─── Line fill ───────────────────────────────────────────────────────────────
//
// The rally solver rightly refuses to call a poor fit a measurement, but a
// refused flight is not an absent ball: fast serves and attacks are rejected
// with most of their detections present. Their frames are filled here so the
// line runs through the rally, marked "estimated" so nothing reads them as
// fitted: touches are decided before this runs, and the score and point
// winners use fitted frames only.

/** A detection further than this (m) from the estimated arc is a false one. */
const FILL_MAX_CORRECTION_M = 4;
/** Frames over which a correction fades in from a knot, so the flights still meet. */
const FILL_TAPER_FRAMES = 3;
/** Half-width (frames) of the median over corrections: one wild detection must not kink the line. */
const FILL_MEDIAN_RADIUS = 2;
/** Longest hole (s) bridged between flights: as long as a very high pass stays up. */
const FILL_BRIDGE_MAX_S = 2.5;
/** A bridging arc rises no higher than this (m): a long hole is rarely one flight. */
const FILL_BRIDGE_MAX_Y_M = 8;
/** How long (s) the last flight of a rally is followed on after the ball is lost. */
const FILL_FOLLOW_MAX_S = 1.5;
/** Fastest launch a followed flight may have (m/s), as for the solver's speed prior. */
const FILL_FOLLOW_MAX_SPEED = 30;
/** Frames the launch velocity of a followed flight is averaged over. */
const FILL_FOLLOW_VEL_FRAMES = 4;

type V3 = [number, number, number];

function fillLineGaps(
  rallies: RallySegmentResult[][],
  frames: SolverFrame[],
  cam: THREE.PerspectiveCamera,
  touches: BallTouch[],
  positions: Map<number, V3>,
  methods: Map<number, BallReconstructionMethod>,
): void {
  const set = (fr: SolverFrame, p: V3) => {
    positions.set(fr.frame, [
      Math.max(-RALLY_MAX_ABS_X_M - 1, Math.min(RALLY_MAX_ABS_X_M + 1, p[0])),
      Math.max(HALF_BALL, Math.min(SEG_MAX_Y_M, p[1])),
      Math.max(-RALLY_MAX_ABS_Z_M - 1, Math.min(RALLY_MAX_ABS_Z_M + 1, p[2])),
    ]);
    methods.set(fr.frame, "estimated");
  };

  // Places a run of frames on `base`, moved onto the detections' lines of
  // sight: the depth comes from the arc, the rest from what the video shows.
  const follow = (run: SolverFrame[], base: (t: number) => V3, taperStart: boolean, taperEnd: boolean) => {
    if (run.length === 0) return;
    const corr: (V3 | null)[] = run.map((fr) => {
      if (!fr.ball_img_pos || fr.ball_interpolated) return null;
      const { origin, dir } = rayFromImagePos(cam, fr.ball_img_pos);
      const b = new THREE.Vector3(...base(fr.timestamp));
      const c = origin.clone().addScaledVector(dir, b.clone().sub(origin).dot(dir)).sub(b);
      return c.length() <= FILL_MAX_CORRECTION_M ? [c.x, c.y, c.z] : null;
    });
    const at = (i: number, axis: number): number => {
      const near: number[] = [];
      for (let j = Math.max(0, i - FILL_MEDIAN_RADIUS); j <= Math.min(run.length - 1, i + FILL_MEDIAN_RADIUS); j++) {
        const c = corr[j];
        if (c) near.push(c[axis]);
      }
      if (near.length) return _median(near);
      // No detection nearby: carry between the nearest on each side.
      let lo = i, hi = i;
      while (lo >= 0 && !corr[lo]) lo--;
      while (hi < run.length && !corr[hi]) hi++;
      const a = lo >= 0 ? corr[lo]![axis] : null, b = hi < run.length ? corr[hi]![axis] : null;
      if (a == null || b == null) return a ?? b ?? 0;
      return a + ((b - a) * (i - lo)) / (hi - lo);
    };
    run.forEach((fr, i) => {
      const fade = Math.min(
        taperStart ? (i + 1) / FILL_TAPER_FRAMES : 1,
        taperEnd ? (run.length - i) / FILL_TAPER_FRAMES : 1,
        1,
      );
      const b = base(fr.timestamp);
      set(fr, [b[0] + fade * at(i, 0), b[1] + fade * at(i, 1), b[2] + fade * at(i, 2)]);
    });
  };

  // Flights the solver refused: their arcs share knots with their neighbours,
  // so they already join up.
  for (const rally of rallies) {
    rally.forEach((seg, k) => {
      if (seg.accepted) return;
      const run = frames.filter((fr) =>
        fr.timestamp >= seg.tStart - 1e-6 && fr.timestamp <= seg.tEnd + 1e-6 && !positions.has(fr.frame));
      follow(run, seg.posAt, k > 0, k < rally.length - 1);
    });
  }

  // Holes left between rallies, and where the ball was lost for good.
  const solved = (i: number) => (i >= 0 && i < frames.length ? positions.get(frames[i].frame) : undefined);
  const opensPlay = (from: number, to: number) =>
    touches.some((t) => t.frame >= from && t.frame <= to && /^(toss|serve)/.test(t.label));
  for (let a = 0; a < frames.length; a++) {
    const pa = solved(a);
    if (!pa || solved(a + 1) || a + 1 >= frames.length) continue;
    let b = a + 1;
    while (b < frames.length && !solved(b)) b++;
    // The ball is on the floor, or the next point has started: nothing connects them.
    if (pa[1] <= FLOOR_CONTACT_Y_M) continue;
    const hasNext = b < frames.length;
    const T = hasNext ? frames[b].timestamp - frames[a].timestamp : Infinity;
    if (hasNext && T <= FILL_BRIDGE_MAX_S && !opensPlay(frames[b].frame, frames[b].frame + FILL_TAPER_FRAMES)) {
      const pb = solved(b)!;
      const t0 = frames[a].timestamp;
      const g = Math.min(GRAVITY, (8 * Math.max(0, FILL_BRIDGE_MAX_Y_M - Math.max(pa[1], pb[1]))) / (T * T));
      follow(frames.slice(a + 1, b), (t) => {
        const s = t - t0, w = s / T;
        return [pa[0] + (pb[0] - pa[0]) * w, pa[1] + (pb[1] - pa[1]) * w + 0.5 * g * s * (T - s), pa[2] + (pb[2] - pa[2]) * w];
      }, true, true);
      continue;
    }
    // Follow the last flight on under gravity until it reaches the floor.
    const back = Math.max(0, a - FILL_FOLLOW_VEL_FRAMES);
    const p0 = solved(back);
    const span = frames[a].timestamp - frames[back].timestamp;
    if (!p0 || span <= 0) continue;
    const v: V3 = [(pa[0] - p0[0]) / span, (pa[1] - p0[1]) / span - (GRAVITY * span) / 2, (pa[2] - p0[2]) / span];
    const speed = Math.hypot(...v);
    if (speed > FILL_FOLLOW_MAX_SPEED) { for (let c = 0; c < 3; c++) v[c] *= FILL_FOLLOW_MAX_SPEED / speed; }
    for (let i = a + 1; i < b; i++) {
      const s = frames[i].timestamp - frames[a].timestamp;
      if (s > FILL_FOLLOW_MAX_S) break;
      const y = pa[1] + v[1] * s - 0.5 * GRAVITY * s * s;
      const x = pa[0] + v[0] * s, z = pa[2] + v[2] * s;
      if (y < HALF_BALL || Math.abs(x) > RALLY_MAX_ABS_X_M + 1 || Math.abs(z) > RALLY_MAX_ABS_Z_M + 1) break;
      set(frames[i], [x, y, z]);
    }
    a = b - 1;
  }
}

// ─── Physics-reprojection arc solver ─────────────────────────────────────────

/** Extended arc observation: includes confidence and optional smoothed depth prior. */
interface ArcObsEx extends ArcObs {
  conf: number;
  depthZ: number | null;
  /**
   * Camera depth (m) implied by the measured ball diameter, up to a per-rally
   * scale. Kept apart from depthZ, which drives arc splitting and is too
   * noisy per frame for that.
   */
  sizeDepth?: number | null;
  /** On the first sample after a scanned cut: how clearly the track breaks there (see scanTouches). */
  kink?: number;
  /**
   * Vertical NDC position in the camera's own image (u/v are in the
   * reference image for a moving camera). Says whether the ball is at the
   * edge of the picture; see joinOffscreenFlights.
   */
  frameV?: number;
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
        // Soft physical bounds.
        //
        // These were flat constants, which give the optimiser no gradient: a
        // solution 3 m out of court paid exactly the same as one 30 m out, so
        // nothing pulled it back. Under a long lens depth is only weakly
        // observable and many depths fit the 2D track about equally well, so
        // this prior is what selects the physically plausible one. Penalties
        // are quadratic in the violation and scaled to be comparable with the
        // reprojection term (a few px is ~1e-5 in NDC²).
        // 3 m past the baseline is the FIVB minimum free zone and about as far
        // back as a server stands, so it is a real physical bound rather than a
        // tuning knob. It matters most for flat, fast arcs: gravity only fixes
        // the depth scale when the trajectory visibly curves, so a near-straight
        // serve is under-determined from one camera and this prior is what keeps
        // it on the court.
        const OUT_W = 2e-3;
        if (wy < 0) err += OUT_W * 4 * wy * wy;
        else if (wy > MAX_BALL_HEIGHT_M) err += OUT_W * (wy - MAX_BALL_HEIGHT_M) ** 2;
        const xOver = Math.abs(wx) - (COURT_HALF_W + 3);
        if (xOver > 0) err += OUT_W * xOver * xOver;
        const zOver = Math.abs(wz) - (COURT_HALF_L + 3);
        if (zOver > 0) err += OUT_W * zOver * zOver;
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
      // Depth multi-start anchored to court geometry.
      //
      // A single seed taken from apparent ball size is not usable: with
      // TrackNet the box is the heatmap blob rather than the ball's
      // silhouette, so the implied depth is meaningless. Under a long lens the
      // projection is close to orthographic, the cost surface is nearly flat
      // along depth, and one bad starting depth leaves the optimiser in a
      // local minimum hundreds of pixels away from the observations.
      //
      // Instead, walk the viewing ray of the first observation. Its ground
      // intersection bounds how far the ball can be — a ball in the air is
      // always nearer along that ray than the point where the ray meets the
      // floor — so seeding fractions of that distance brackets the true depth.
      const first = activeObs[0];
      const last = activeObs[activeObs.length - 1];
      const dur = Math.max(1e-3, last.t - first.t);

      const rayDir = (o: ArcObsEx) =>
        new THREE.Vector3(o.u, o.v, 0.5).unproject(cam).sub(cam.position).normalize();
      const groundT = (d: THREE.Vector3) =>
        (d.y < -1e-6 ? -cam.position.y / d.y : NaN);

      const d0 = rayDir(first);
      const dN = rayDir(last);
      const t0Ground = groundT(d0);
      const tNGround = groundT(dN);

      const seeded: number[][] = [];
      if (Number.isFinite(t0Ground) && Number.isFinite(tNGround) &&
          t0Ground > 0 && tNGround > 0) {
        for (const frac of [0.5, 0.65, 0.78, 0.88, 0.96, 1.0]) {
          const p0 = cam.position.clone().addScaledVector(d0, frac * t0Ground);
          const pN = cam.position.clone().addScaledVector(dN, frac * tNGround);
          if (p0.y < -1 || p0.y > MAX_BALL_HEIGHT_M + 2) continue;
          // Straight-line velocity between the two seeded endpoints, with the
          // gravity drop added back so the seed is a plausible arc.
          const vxs = (pN.x - p0.x) / dur;
          const vys = (pN.y - p0.y) / dur + 0.5 * GRAVITY * dur;
          const vzs = (pN.z - p0.z) / dur;
          seeded.push([p0.x, Math.max(0.05, p0.y), p0.z,
                       vxs / VSCALE, vys / VSCALE, vzs / VSCALE]);
        }
      }

      // Keep the original bbox-depth seed and generic velocity fans as
      // additional starts; the best fit across all of them wins.
      for (const [sivx, sivy, sivz] of [
        [0,  0.4, -0.5],
        [0,  0.4,  0.5],
        [0, -0.5, -0.5],
        [initVx / VSCALE, initVy / VSCALE, initVz / VSCALE],
      ] as [number, number, number][]) {
        seeded.push([initX, initY, initZ, sivx, sivy, sivz]);
      }

      for (const sp of seeded) tryP(sp);
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

type SolverFrame = Pick<FrameData, "frame" | "timestamp" | "ball_img_pos" | "ball_bbox" | "ball_interpolated" | "ball_conf" | "ball_diameter_px" | "players_near_ball"
  | "players" | "players_arms" | "ball_ref_pos" | "ball_ref_diameter_px" | "players_ref" | "players_all_ref"
  | "players_arms_ref"> & {
  /** Ball position in the camera's own image, kept when ball_img_pos is swapped for the reference one. */
  ball_frame_pos?: [number, number];
};

/**
 * For a camera that pans and zooms, use the measurements expressed in the
 * reference image (the analyzer's "virtual static camera"), which is the
 * image the calibration belongs to. A frame without a trusted motion estimate
 * loses its ball rather than mixing two coordinate systems. Static clips have
 * no ref fields and pass through unchanged.
 */
function toReferenceFrames(frames: SolverFrame[]): SolverFrame[] {
  if (!frames.some((f) => f.ball_ref_pos)) return frames;
  return frames.map((f) => ({
    ...f,
    ball_img_pos: f.ball_ref_pos,
    ball_frame_pos: f.ball_img_pos,
    ball_diameter_px: f.ball_ref_diameter_px,
    players_near_ball: f.players_ref,
    players: f.players_all_ref,
    players_arms: f.players_arms_ref,
    // The bbox is in raw image coords and is not used by this mode.
    ball_bbox: undefined,
  }));
}

export function solveBallTrajectories(
  frames: SolverFrame[],
  calibration: CalibrationResult,
  aspect: number,
  options?: BallReconstructionOptions,
): BallTrajectoryResult {
  frames = toReferenceFrames(frames);
  const prev = P;
  const o = options?.rallyParams;
  P = o
    ? { ...DEFAULT_RALLY_PARAMS, ...o, rules: { ...DEFAULT_RALLY_PARAMS.rules, ...o.rules } }
    : DEFAULT_RALLY_PARAMS;
  try {
    return _solveBallTrajectories(frames, calibration, aspect, options);
  } finally {
    P = prev;
  }
}

function _solveBallTrajectories(
  frames: SolverFrame[],
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
        usedInFit:          (trajectoryMode !== "physics-reprojection" && trajectoryMode !== "short-arc-debug")
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
    const touches: BallTouch[] = [];
    const touchSplits: TouchSplitDecision[] = [];

    if (trajectoryMode === "physics-reprojection") {
      // Detections reaching here have already passed the analyzer's motion
      // gate (_BallTracker), which is stronger evidence that a point is the
      // ball than a raw YOLO score is — a fast, blurred ball scores low but
      // tracks perfectly. Measured on broadcast footage, genuine tracked
      // detections land at conf 0.09-0.13, i.e. below the detector's own
      // _BALL_CONF_FLOOR of 0.08 plus a hair. Anything above that floor was
      // already validated, so keep it: the objective below weights each
      // residual by confidence, which down-weights weak points without
      // discarding them. A stricter gate here starves the solver and it
      // rejects every arc.
      const minFitConf = 0.05;
      const obsEx: ArcObsEx[] = detectedList
        .map((f) => ({
          t:      f.timestamp,
          u:      f.ball_img_pos[0] * 2 - 1,
          v:      1 - f.ball_img_pos[1] * 2,
          fi:     f.frame,
          conf:   f.ball_conf ?? 0,
          // Depth from apparent ball size is deliberately not supplied.
          // The detector reports a heatmap blob, not the ball's silhouette:
          // on real footage those boxes quantise to the 512-wide heatmap grid
          // and imply depths of 230-577 m for a ball on an 18 m court. Feeding
          // that in only corrupts the seed and splits arcs on noise.
          depthZ: null,
          // The real ball size, segmented by colour in the analyzer, does carry
          // depth: it is used as a soft prior by the rally solver.
          sizeDepth: f.ball_diameter_px && f.ball_diameter_px > 0
            ? fyPx * ballDiameterM / f.ball_diameter_px
            : null,
          frameV: f.ball_frame_pos ? 1 - f.ball_frame_pos[1] * 2 : undefined,
        }))
        .filter((o) => o.conf >= minFitConf);

      // Ball image position at any time, including the analyzer's gap fill.
      const ballTrack = frames
        .filter((f) => f.ball_img_pos)
        .map((f) => ({ t: f.timestamp, u: f.ball_img_pos![0] * 2 - 1, v: 1 - f.ball_img_pos![1] * 2 }));
      const frameDt = frames.length > 1 ? Math.max(1e-3, frames[1].timestamp - frames[0].timestamp) : 1 / 30;
      const nearestFrame = (t: number): number => {
        let best = frames[0];
        for (const f of frames) if (Math.abs(f.timestamp - t) < Math.abs(best.timestamp - t)) best = f;
        return best.frame;
      };
      const imageAt = (t: number): [number, number] | null => {
        let best: { t: number; u: number; v: number } | null = null;
        for (const b of ballTrack) if (!best || Math.abs(b.t - t) < Math.abs(best.t - t)) best = b;
        return best && Math.abs(best.t - t) <= 0.6 * frameDt ? [best.u, best.v] : null;
      };
      const playersByFrame = new Map<number, PlayerBox[]>();
      for (const f of frames) if (f.players_near_ball) playersByFrame.set(f.frame, f.players_near_ball);
      const hasPlayers = playersByFrame.size > 0;
      // Every player on every frame (newer analyzer output), for touches the
      // ball is hidden at and for finding the server.
      const allPlayersByFrame = new Map<number, PlayerBox[]>();
      for (const f of frames) if (f.players) allPlayersByFrame.set(f.frame, f.players);
      const armsByFrame = new Map<number, PlayerArms[]>();
      for (const f of frames) if (f.players_arms) armsByFrame.set(f.frame, f.players_arms);
      {
        const centre = new THREE.Vector3(0, 1.5, 0);
        const fPx = vH / 2 / Math.tan((effectiveFovY * Math.PI) / 360);
        courtMetresPerPx = cam.position.distanceTo(centre) / fPx;
        flightCam = { cam, fPx, courtDist: cam.position.distanceTo(centre) };
        clipFps = 1 / frameDt;
        trackBoxes = allPlayersByFrame.size ? allPlayersByFrame : null;
      }
      const touchPins = options?.touchEdits?.pins ?? [];
      const { arcs: rawArcs, starts: markedTouches } = splitAtMarkedTouches(
        splitArcsEx(obsEx, vW, vH),
        [...(options?.touchEdits?.touchAt ?? []), ...touchPins.map((pin) => pin.frame)],
      );
      const removedTouches = options?.touchEdits?.noTouchAt ?? [];
      const buildArcs = (skip: Set<number>, joined?: number[], log?: TouchSplitDecision[]) => {
        const split = joinOffscreenFlights(rawArcs, skip, joined);
        const keep = new Set([...skip, ...markedTouches]);
        const merged = hasPlayers ? mergeUntouchedSplits(split, playersByFrame, cam, vW / 2, vH / 2, log, keep) : split;
        return { split, arcs: mergeAtRemovedTouches(merged, removedTouches, markedTouches) };
      };
      // Volleyball rules and player anchors on top of the rally solver.
      // Knots that are the ball above the picture, not touches (see below).
      const offscreen = new Set<number>();
      const solveRuled = (rally: ArcObsEx[][], resume: RulesState | null = null) => {
        const free = new Set<number>();
        for (let k = 1; k < rally.length; k++) if (offscreen.has(rally[k][0].fi)) free.add(k);
        // Candidate touchers per knot: the rally's first sample (ball in the
        // server's hands) and every touch between segments. The last knot is
        // where the ball leaves view, not a touch.
        const anchors: TouchCandidate[][] = [];
        const contacts: (HiddenContact | null)[] = [];
        if (hasPlayers) {
          const first = rally[0];
          anchors.push(boundaryCandidates(first[0], first[Math.min(1, first.length - 1)], playersByFrame, cam, TOSS_REACH_UP));
          if (allPlayersByFrame.size && !anchors[0].some(([, z]) => Math.abs(z) >= P.serverMinAbsZ - 1)) {
            anchors[0].push(...serverCandidates(first[0], allPlayersByFrame, cam));
          }
          for (let k = 1; k < rally.length; k++) {
            if (free.has(k)) { anchors.push([]); continue; }
            const prevSeg = rally[k - 1];
            const a = prevSeg[prevSeg.length - 1], b = rally[k][0];
            // Hidden touch found at a player's arms: that player, alone.
            const contact = armsByFrame.size && b.fi - a.fi > 2
              ? findHiddenContact(prevSeg, rally[k], frames, allPlayersByFrame, armsByFrame, cam, vW / 2, vH / 2)
              : null;
            contacts[k] = contact;
            if (contact) { anchors.push([contact.spot]); continue; }
            const cands = boundaryCandidates(a, b, playersByFrame, cam);
            // The ball hidden at the touch (behind the receiver's arms, say):
            // the players around where it was at that moment.
            // The contact can be anywhere in the gap, so look across it.
            if (allPlayersByFrame.size && b.fi - a.fi > 2) {
              const n = Math.min(HIDDEN_TOUCH_SAMPLES, b.fi - a.fi - 1);
              for (let i = 1; i <= n; i++) {
                const t = a.t + ((b.t - a.t) * i) / (n + 1);
                const img = imageAt(t);
                if (!img) continue;
                for (const c of touchCandidates(nearestFrame(t), [(img[0] + 1) / 2, (1 - img[1]) / 2], allPlayersByFrame, cam)) {
                  if (Math.abs(c[0]) > COURT_HALF_W + IN_PLAY_MARGIN_M || Math.abs(c[1]) > COURT_HALF_L + IN_PLAY_MARGIN_M) continue;
                  // Same player seen again a few frames on: keep one.
                  if (!cands.some(([x, z]) => Math.hypot(x - c[0], z - c[1]) < 0.5)) cands.push(c);
                }
              }
            }
            anchors.push(cands);
          }
          anchors.push([]);
        }
        // Knot times as solveRally sets them: mid-gap, or the contact at the arms.
        const knotT = rally.map((seg, k) => k === 0 ? seg[0].t
          : contacts[k]?.t ?? 0.5 * (rally[k - 1][rally[k - 1].length - 1].t + seg[0].t));
        const rules: { state?: RulesState } = {};
        const zones = hasPlayers ? applyRallyRules(anchors, free, knotT, resume, rules) : [];
        // Hand-placed touches: knot k sits between the last sample before it
        // and the first after (the first sample itself for the rally start).
        const pins = rally.map((seg, k) => {
          const lo = (k === 0 ? seg[0].fi : rally[k - 1][rally[k - 1].length - 1].fi) - EDIT_TOUCH_SNAP_FRAMES;
          const hi = seg[0].fi + EDIT_TOUCH_SNAP_FRAMES;
          return touchPins.find((pin) => pin.frame >= lo && pin.frame <= hi)?.pos ?? null;
        });
        pins.forEach((pin, k) => { if (pin && anchors.length > k) anchors[k] = [[pin[0], pin[2]]]; });
        return {
          anchors, zones, free, contacts, rules: rules.state ?? null,
          results: solveRally(rally, cam, vW, vH, anchors, zones, imageAt, free, contacts, pins),
        };
      };
      // A flight joined across the top of the picture that the solver cannot
      // fit (the arc is rejected) is put back as two pieces, so it is not
      // lost from the path altogether.
      const joined: number[] = [];
      const firstPass = buildArcs(new Set(), joined);
      const badJoins = new Set<number>();
      if (joined.length) {
        for (const rally of groupIntoRallies(firstPass.arcs)) {
          if (rally.reduce((s, a) => s + a.length, 0) < MIN_ARC_FRAMES) continue;
          const segJoins = rally.map((seg) => joined.filter((fi) => seg.some((o) => o.fi === fi)));
          if (!segJoins.some((j) => j.length)) continue;
          solveRuled(rally).results.forEach((r, k) => {
            if (!r.accepted) segJoins[k].forEach((fi) => badJoins.add(fi));
          });
        }
        console.log(`[physicsReproj] off-screen flights joined: ${joined.length}, undone (no fit): ${badJoins.size}`);
      }
      const { split: splitArcs, arcs } = buildArcs(badJoins, undefined, touchSplits);
      // An undone join is still one flight: the halves stay in one rally, and
      // the knot between them is the ball above the picture, not a touch.
      badJoins.forEach((fi) => offscreen.add(fi));
      if (hasPlayers) {
        console.log(`[physicsReproj] touch splits: ${splitArcs.length} -> ${arcs.length} after dropping ones with no player`);
      }

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
      const solvedRallies: RallySegmentResult[][] = [];

      // Segments shorter than two samples cannot carry a touch-to-touch arc on
      // their own, but they still constrain the rally they sit in.
      const rallies = groupIntoRallies(arcs, offscreen);
      // The touch sequence carries over a detection gap into the next rally.
      let carry: RulesState | null = null;
      let segId = 0;
      for (const rally of rallies) {
        const nObs = rally.reduce((s, a) => s + a.length, 0);
        if (nObs < MIN_ARC_FRAMES) {
          for (const arc of rally) {
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
                arcId: segId, arcAccepted: false, arcRejectReason: "too-few-observations",
                speedMps: null, cameraDepthFitted: null, depthPriorError: null, isOutlier: false,
              });
            }
            segId++;
          }
          continue;
        }

        const { anchors, zones, free, contacts, rules, results } = solveRuled(rally, carry);
        carry = rules;
        solvedRallies.push(results);
        // Which knots are touches. Measured on 11 labelled broadcast clips,
        // false touches came from three places, each handled here:
        //  - a rally's first point is only where the ball came into view (or
        //    reappeared after a >1 s detection gap, usually high in the air):
        //    a touch only when the rules identified it as the toss or serve;
        //  - after the ball reaches the floor the play is over; bounces,
        //    catches and rolling balls are not touches;
        //  - the touch's frame is the contact itself (the knot), not the first
        //    detection after it, which is several frames late when the ball is
        //    hidden at the contact.
        let deadAfter = Infinity;
        results.forEach((seg, k) => {
          // Not the rally's first point, not a touch the rules identified (a
          // reception solved too low is a solver error, not the end of play),
          // and not below the floor (also a solver error).
          if (k === 0 || zones[k] || free.has(k)) return;
          const y = seg.posAt(seg.tStart)[1];
          if (y > -0.2 && y <= FLOOR_CONTACT_Y_M && k < deadAfter) deadAfter = k;
        });
        // The bounce itself is often not a knot: near the floor, among the
        // players, the ball goes undetected. But the flight into it still
        // reaches the floor, so a flight whose arc dips to floor height ends
        // play at its end. Same guards: not when the next point is a touch
        // the rules identified, not for arcs far below the floor (solver error).
        results.forEach((seg, k) => {
          if (!seg.accepted || zones[k + 1] || k + 1 >= deadAfter) return;
          let minY = Infinity;
          for (let i = 1; i < 10; i++) minY = Math.min(minY, seg.posAt(seg.tStart + ((seg.tEnd - seg.tStart) * i) / 10)[1]);
          if (minY > -0.5 && minY <= FLOOR_ARC_Y_M) deadAfter = k + 1;
        });
        results.forEach((seg, k) => {
          if (free.has(k)) return;
          // A touch the rules placed at a player stands even when the flights
          // on both sides of it fit badly (a reception between a blurred serve
          // and a pass that left the picture).
          // But not on a few stray detections: with no fitted flight on either
          // side, fewer samples after it than a flight needs say nothing left
          // it (usa_jap_r4: the tracker hopped to a player and then the floor
          // for three frames after a serve, and that became a reception).
          const ruled = !!zones[k] && (anchors[k]?.length ?? 0) > 0 && seg.obs.length >= MIN_ARC_FRAMES;
          if (!seg.accepted && !(k > 0 && results[k - 1].accepted) && !ruled) return;
          if (k === 0 && !zones[0]) return;
          if (k >= deadAfter) return;
          // A toss is the ball leaving the server's hand. A ball first seen
          // well above any reach is already in flight: the clip (or the
          // detector) picked it up after the toss, which nobody can date.
          if (zones[k]?.label === "toss" && seg.posAt(seg.tStart)[1] > zones[k]!.maxY + TOSS_SEEN_SLACK_M) return;
          // Short gap (an arm hides the ball at contact): the knot is the
          // contact. Long gap (a high toss leaves the frame and only comes
          // back at the serve): the ball reappears right after the contact,
          // while the knot, midway through the gap, would be far too early.
          const prevObs = k > 0 ? results[k - 1].obs : null;
          const gap = prevObs ? seg.obs[0].t - prevObs[prevObs.length - 1].t : 0;
          const frame = contacts[k]?.frame ?? (gap <= TOUCH_FRAME_MAX_GAP_S ? nearestFrame(seg.tStart) : seg.obs[0].fi);
          const position = seg.posAt(seg.tStart);
          const sincePrev = k > 0 ? seg.tStart - results[k - 1].tStart : 0;
          touches.push({ frame, position, label: zones[k] ? zoneLabel(zones[k]!, position, sincePrev) : "touch", candidates: anchors[k] ?? [] });
        });
        console.log(
          "[rally] " + results.map((seg, k) => {
            const [x, y, z] = seg.posAt(seg.tStart);
            return `${zones[k]?.label ?? "touch"} f${seg.obs[0].fi} (${x.toFixed(1)},${y.toFixed(1)},${z.toFixed(1)})` +
              ` cands=${(anchors[k] ?? []).map(([ax, az]) => `${ax.toFixed(1)}/${az.toFixed(1)}`).join(" ")} -> ${seg.speedMps.toFixed(0)}m/s`;
          }).join("  |  "),
        );
        const camFwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
        for (const seg of results) {
          const ai = segId++;
          console.log(
            `[physicsReproj] seg[${ai}] frames ${seg.obs[0].fi}–${seg.obs[seg.obs.length - 1].fi}` +
            ` n=${seg.obs.length} speed=${seg.speedMps.toFixed(1)}m/s` +
            ` start=[${seg.posAt(seg.tStart).map((v) => v.toFixed(1)).join(",")}]` +
            ` end=[${seg.posAt(seg.tEnd).map((v) => v.toFixed(1)).join(",")}]` +
            ` ${seg.accepted ? "ACCEPTED" : `REJECTED: ${seg.rejectReason}`}`,
          );
          // A rejected sliver between two touches (attack straight into the
          // block: two touches in five frames) is drawn along its solved arc
          // between its touch points, so the path does not break there. Not a
          // straight line: that drops the gravity sag the net-clearance term
          // was applied to (pol_ita_r3: a fast serve drawn under the net tape).
          const segIdx = results.indexOf(seg);
          const bridge = !seg.accepted && seg.obs.length <= P.bridgeMaxFrames &&
            segIdx > 0 && segIdx < results.length - 1 &&
            results[segIdx - 1].accepted && results[segIdx + 1].accepted;
          if (bridge) {
            for (const fr of frames) {
              if (fr.timestamp < seg.tStart - 1e-6 || fr.timestamp > seg.tEnd + 1e-6) continue;
              if (positions.has(fr.frame)) continue;
              positions.set(fr.frame, seg.posAt(fr.timestamp));
              methods.set(fr.frame, "physics-reprojection");
            }
          }
          if (seg.accepted) {
            totalFittedArcs++;
            for (const fr of frames) {
              if (fr.timestamp < seg.tStart - 1e-6 || fr.timestamp > seg.tEnd + 1e-6) continue;
              if (positions.has(fr.frame)) continue;
              positions.set(fr.frame, seg.posAt(fr.timestamp));
              methods.set(fr.frame, "physics-reprojection");
            }
          }
          const project = _makeProjector(cam);
          seg.obs.forEach((o, i) => {
            const pos = seg.posAt(o.t);
            const [projU, projV] = project(...pos);
            const isOutlier = seg.outlier[i];
            if (!isOutlier) allErrPx.push(seg.errPx[i]);
            if (seg.accepted && !isOutlier) {
              const rec = debugRecords.find((d) => d.frame === o.fi);
              if (rec) {
                rec.fittedPosition      = pos;
                rec.projectedU          = projU;
                rec.projectedV          = projV;
                rec.reprojectionErrorPx = seg.errPx[i];
                rec.worldPosition       = pos;
                rec.accepted            = true;
                rec.rejectedReason      = null;
                rec.method              = "physics-reprojection";
                rec.usedInFit           = true;
              }
            }
            const origIdx = detectedList.findIndex((df) => df.frame === o.fi);
            const cameraDepthFitted =
              (pos[0] - cam.position.x) * camFwd.x +
              (pos[1] - cam.position.y) * camFwd.y +
              (pos[2] - cam.position.z) * camFwd.z;
            physicsDebug.push({
              frame:               o.fi,
              observedU:           o.u,
              observedV:           o.v,
              projectedU:          projU,
              projectedV:          projV,
              reprojectionErrorPx: seg.errPx[i],
              fittedX:             pos[0],
              fittedY:             pos[1],
              fittedZ:             pos[2],
              rawBboxDepthZ:       origIdx >= 0 ? rawZFromDiam[origIdx]      : null,
              smoothedBboxDepthZ:  origIdx >= 0 ? smoothedZFromDiam[origIdx] : null,
              confidence:          o.conf,
              usedInFit:           seg.accepted && !isOutlier,
              arcId:               ai,
              arcAccepted:         seg.accepted,
              arcRejectReason:     seg.rejectReason,
              speedMps:            seg.speedMps,
              cameraDepthFitted,
              depthPriorError:     null,
              isOutlier,
            });
          });
        }
      }
      const fitted = positions.size;
      fillLineGaps(solvedRallies, frames, cam, touches, positions, methods);
      console.log(`[physicsReproj] line fill: ${positions.size - fitted} estimated frames on ${fitted} fitted`);

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

    // ── Short-arc-debug arc solver (post-loop) ────────────────────────────────
    if (trajectoryMode === "short-arc-debug") {
      const arcStart      = options?.arcStartFrame ?? 0;
      const arcEnd        = options?.arcEndFrame   ?? Number.MAX_SAFE_INTEGER;
      const disableDepth  = options?.disableBboxDepthPrior ?? false;
      const minFitConf    = 0.0; // accept all detections in range for short-arc mode

      const sadObsEx: ArcObsEx[] = detectedList
        .map((f, i) => ({
          t:      f.timestamp,
          u:      f.ball_img_pos[0] * 2 - 1,
          v:      1 - f.ball_img_pos[1] * 2,
          fi:     f.frame,
          conf:   Math.max(f.ball_conf ?? 0, 0.01), // treat 0 as 0.01 so it's not filtered
          depthZ: disableDepth ? null : smoothedZFromDiam[i],
        }))
        .filter((o) => o.fi >= arcStart && o.fi <= arcEnd && o.conf >= minFitConf);

      const sadPhysicsDebug: PhysicsReprojectionDebug[] = [];
      let shortArcAccepted = false;
      let shortArcRejectReason: string | null = "no-observations-in-range";

      if (sadObsEx.length < MIN_ARC_FRAMES) {
        shortArcRejectReason = `too-few-observations (${sadObsEx.length} < ${MIN_ARC_FRAMES})`;
        console.warn(`[shortArcDebug] ${shortArcRejectReason}`);
      } else {
        const solved = solveArcReprojection(sadObsEx, cam, vW, vH, 0);

        if (!solved) {
          shortArcRejectReason = `solver-returned-null`;
        } else {
          const { arcParams, t0: arcT0, observations: solvedObs } = solved;
          const [ax0, ay0, az0, avx, avy, avz] = arcParams;

          const inlierObs   = solvedObs.filter((o) => !o.isOutlier);
          const allX        = solvedObs.map((o) => o.pos[0]);
          const allZ        = solvedObs.map((o) => o.pos[2]);
          const allY        = solvedObs.map((o) => o.pos[1]);
          const inlierErrs  = inlierObs.map((o) => o.errPx);
          const avgErrPx    = inlierErrs.length > 0
            ? inlierErrs.reduce((s, e) => s + e, 0) / inlierErrs.length : Infinity;
          const maxErrPx    = inlierErrs.length > 0 ? Math.max(...inlierErrs) : Infinity;
          const maxSpeed    = Math.max(0, ...solvedObs.map((o) => o.speedMps));

          // Stricter short-arc validation
          let sadRejectReason: string | null = null;
          if (allX.some((x) => Math.abs(x) > SAD_MAX_X_M))
            sadRejectReason = `x-out-of-bounds (max|x|=${Math.max(...allX.map(Math.abs)).toFixed(2)}m > ${SAD_MAX_X_M}m)`;
          else if (allZ.some((z) => Math.abs(z) > SAD_MAX_Z_M))
            sadRejectReason = `z-out-of-bounds (max|z|=${Math.max(...allZ.map(Math.abs)).toFixed(2)}m > ${SAD_MAX_Z_M}m)`;
          else if (allY.some((y) => y < 0))
            sadRejectReason = `y-below-zero (min=${Math.min(...allY).toFixed(2)}m)`;
          else if (allY.some((y) => y > 6))
            sadRejectReason = `y-too-high (max=${Math.max(...allY).toFixed(2)}m > 6m)`;
          else if (avgErrPx > SAD_MAX_AVG_ERR_PX)
            sadRejectReason = `avg-reprojection-err-${avgErrPx.toFixed(1)}px > ${SAD_MAX_AVG_ERR_PX}px`;
          else if (maxErrPx > SAD_MAX_MAX_ERR_PX)
            sadRejectReason = `max-reprojection-err-${maxErrPx.toFixed(1)}px > ${SAD_MAX_MAX_ERR_PX}px`;
          else if (maxSpeed > MAX_BALL_SPEED_MPS)
            sadRejectReason = `speed-${maxSpeed.toFixed(1)}m/s > ${MAX_BALL_SPEED_MPS}m/s`;

          shortArcAccepted    = sadRejectReason === null;
          shortArcRejectReason = sadRejectReason;

          console.log(
            `[shortArcDebug] frames ${arcStart}–${arcEnd}` +
            ` obs=${sadObsEx.length} inliers=${inlierObs.length}` +
            ` avgErr=${avgErrPx.toFixed(1)}px maxErr=${maxErrPx.toFixed(1)}px` +
            ` yRange=[${Math.min(...allY).toFixed(2)},${Math.max(...allY).toFixed(2)}]` +
            ` maxSpeed=${maxSpeed.toFixed(1)}m/s` +
            ` ${shortArcAccepted ? "ACCEPTED" : `REJECTED: ${shortArcRejectReason}`}`,
          );

          // Only populate positions for accepted arc — rejected arc stays out of yellow path
          if (shortArcAccepted) {
            for (const f of detectedList) {
              if (f.frame < arcStart || f.frame > arcEnd) continue;
              const t  = f.timestamp - arcT0;
              const wx = ax0 + avx * t;
              const wy = ay0 + avy * t - 0.5 * GRAVITY * t * t;
              const wz = az0 + avz * t;
              positions.set(f.frame, [wx, wy, wz]);
              methods.set(f.frame, "physics-reprojection");
            }
          }

          // Build physicsDebug records for this arc
          for (const { fi, pos, projU, projV, errPx, speedMps, cameraDepthFitted,
                       depthPriorError, isOutlier, obsRef } of solvedObs) {
            if (shortArcAccepted && !isOutlier) {
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
            sadPhysicsDebug.push({
              frame:               fi,
              observedU:           obsRef.u,
              observedV:           obsRef.v,
              projectedU:          projU,
              projectedV:          projV,
              reprojectionErrorPx: errPx,
              fittedX:             pos[0],
              fittedY:             pos[1],
              fittedZ:             pos[2],
              rawBboxDepthZ:       null,
              smoothedBboxDepthZ:  null,
              confidence:          obsRef.conf,
              usedInFit:           shortArcAccepted && !isOutlier,
              arcId:               0,
              arcAccepted:         shortArcAccepted,
              arcRejectReason:     shortArcRejectReason,
              speedMps,
              cameraDepthFitted,
              depthPriorError,
              isOutlier,
            });
          }
        }
      }

      const solved3dFrms = positions.size;
      const rejectedFrms = debugRecords.filter((d) => !d.accepted).length;
      const mCounts: Partial<Record<BallReconstructionMethod, number>> = {};
      for (const m of methods.values()) mCounts[m] = (mCounts[m] ?? 0) + 1;

      return {
        positions, methods, debugRecords,
        physicsDebug: sadPhysicsDebug.length > 0 ? sadPhysicsDebug : undefined,
        shortArcAccepted,
        shortArcRejectReason,
        stats: { detectedFrames, solved3dFrames: solved3dFrms, rejectedFrames: rejectedFrms, methodCounts: mCounts, avgConf },
      };
    }

    return {
      positions, methods, debugRecords,
      physicsDebug: physicsDebug.length > 0 ? physicsDebug : undefined,
      touches: touches.length > 0 ? touches : undefined,
      touchSplits: touchSplits.length > 0 ? touchSplits : undefined,
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

/** Internal helpers exposed for unit tests (src/eval). Not part of the app API. */
/** Kink score at every position of a track (tests and diagnostics). */
function kinkScoresFor(obs: ArcObsEx[], vW: number, vH: number): number[] {
  const sigma = Math.max(SCAN_MIN_SIGMA_PX, _trackNoisePx(obs, vW / 2, vH / 2));
  return obs.map((_, b) => _kinkScore(obs, b, 0, obs.length, vW / 2, vH / 2, sigma));
}

export const __test__ = { kinkScoresFor, fitFlight, fitQuadratic: _fitQuadratic, applyRallyRules, velocityJumpPx: _velocityJumpPx };
