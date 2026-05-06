import * as THREE from "three";
import type { FrameData } from "../hooks/useAnalysis";
import type { CalibrationResult } from "./cameraSolver";
import { applyHomography, imageToCourtHomography } from "./courtHomography";

const BALL_DIAMETER_M = 0.21;
const HALF_BALL = BALL_DIAMETER_M / 2;
const COURT_HALF_W = 4.5;
const COURT_HALF_L = 9;
const MAX_BALL_HEIGHT_M = 8;
const COURT_WIDTH = 9;
const COURT_LENGTH = 18;

export interface BallWorldEstimate {
  position: [number, number, number];
  ground: [number, number, number];
  height: number;
  method: "homography-ray" | "bbox-depth" | "ground-intersection" | "bbox-bottom-ray";
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

function createCamera(calibration: CalibrationResult, aspect: number): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(calibration.fov, aspect, 0.1, 500);
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

function bboxDepthCam(
  bbox: [number, number, number, number],
  fovDeg: number,
): number {
  const [x1, y1, x2, y2] = bbox;
  const w = Math.max(1e-4, Math.abs(x2 - x1));
  const h = Math.max(1e-4, Math.abs(y2 - y1));
  const fovRad = (fovDeg * Math.PI) / 180;
  return BALL_DIAMETER_M / Math.max(2 * ((w + h) * 0.5) * Math.tan(fovRad * 0.5), 1e-5);
}

function tryBboxBottomRay(
  frame: Pick<FrameData, "ball_img_pos" | "ball_bbox">,
  calibration: CalibrationResult,
  cam: THREE.PerspectiveCamera,
  origin: THREE.Vector3,
  dir: THREE.Vector3,
): BallWorldEstimate | null {
  if (!frame.ball_bbox || !frame.ball_img_pos) return null;
  const [x1, , x2, y2] = frame.ball_bbox;
  const { origin: bOrig, dir: bDir } = rayFromImagePos(cam, [(x1 + x2) * 0.5, y2]);
  if (Math.abs(bDir.y) <= 1e-6) return null;
  const tGnd = -bOrig.y / bDir.y;
  if (tGnd <= 0) return null;
  const groundPt = bOrig.clone().addScaledVector(bDir, tGnd);
  const depth = bboxDepthCam(frame.ball_bbox, calibration.fov);
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
  calibration: CalibrationResult,
  cam: THREE.PerspectiveCamera,
  origin: THREE.Vector3,
  dir: THREE.Vector3,
): BallWorldEstimate | null {
  if (!frame.ball_bbox) return null;
  const depth = bboxDepthCam(frame.ball_bbox, calibration.fov);
  const invQ = cam.quaternion.clone().invert();
  const forward = -dir.clone().applyQuaternion(invQ).z;
  if (forward <= 1e-5) return null;
  const pos = origin.clone().addScaledVector(dir, depth / forward);
  pos.y = Math.max(HALF_BALL, pos.y);
  if (!isPlausibleInCourtVolume(pos)) return null;
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
): BallWorldEstimate | null {
  if (!frame.ball_img_pos) return null;
  if (aspect <= 0) return null;

  const cam = createCamera(calibration, aspect);
  const { origin, dir } = rayFromImagePos(cam, frame.ball_img_pos);

  const r1 = tryBboxBottomRay(frame, calibration, cam, origin, dir);
  if (r1) return r1;

  if (courtCorners?.length === 4) {
    const r2 = tryHomographyRay(frame, origin, dir, courtCorners);
    if (r2) return r2;
  }

  const r3 = tryBboxDepth(frame, calibration, cam, origin, dir);
  if (r3) return r3;

  // ── Fallback: ray-ground intersection ─────────────────────────────────────
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
