import * as THREE from "three";
import { solveCameraIterative, type CalibrationResult } from "../utils/cameraSolver";
import {
  solveBallTrajectories,
  type BallReconstructionOptions,
  type BallTrajectoryResult,
} from "../utils/ballReconstruction";
import type { CourtLabel, Fixture } from "./types";

/** Apply a row-major 3x3 homography to a normalised point. */
function applyH(H: number[] | undefined, [x, y]: number[]): [number, number] {
  if (!H) return [x, y];
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/**
 * Hand-placed court -> matching (reference image point, court point) pairs.
 * Each point is mapped from the frame it was clicked on into the reference
 * image with that frame's homography (identity for a static camera).
 */
export function courtFromLabel(fx: Fixture, court: CourtLabel): { img: [number, number][]; world: [number, number][] } {
  let points = court.points ?? [];
  if (!points.length && court.corners && court.frame != null) {
    // Old format: 4 corners on one frame, image order far-left, far-right,
    // near-right, near-left. Side view: camera at -x, so the picture's left
    // end is z = -9 and the far sideline is x = +4.5 (not mirrored).
    const world: [number, number][] = court.view === "side"
      ? [[4.5, -9], [4.5, 9], [-4.5, 9], [-4.5, -9]]
      : [[-4.5, -9], [4.5, -9], [4.5, 9], [-4.5, 9]];
    points = court.corners.map((pt, i) => ({ id: i, name: "", world: world[i], frame: court.frame!, pt: pt as [number, number] }));
  }
  return {
    img: points.map((p) => applyH(fx.frames[p.frame]?.ref_from_frame, p.pt)),
    world: points.map((p) => p.world),
  };
}

/** Calibrate from the court (hand-placed if given), exactly as App.tsx does. */
export function calibrate(fx: Fixture, court?: CourtLabel): CalibrationResult | null {
  const aspect = fx.video.width / fx.video.height;
  const toNdc = ([x, y]: number[]) => ({ x: x * 2 - 1, y: -(y * 2 - 1) });
  if (court) {
    const { img, world } = courtFromLabel(fx, court);
    if (img.length < 4) return null;
    return solveCameraIterative(img.map(toNdc), aspect, world.map(([x, z]) => ({ x, y: 0, z }))).result;
  }
  if (!fx.courtCorners) return null;
  return solveCameraIterative(fx.courtCorners.map(toNdc), aspect).result;
}

/** A three.js camera matching a calibration (same construction as the solver). */
export function cameraFor(cal: CalibrationResult, aspect: number): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(cal.fov, aspect, 0.1, 500);
  cam.position.set(...cal.position);
  cam.lookAt(new THREE.Vector3(...cal.lookAt));
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  return cam;
}

/** World point -> normalised image coords [0,1]². */
export function projectNorm(cam: THREE.PerspectiveCamera, p: [number, number, number]): [number, number] {
  const v = new THREE.Vector3(...p).project(cam);
  return [(v.x + 1) / 2, (1 - v.y) / 2];
}

/** Normalised image point -> point on the floor (y = 0), or null above the horizon. */
export function floorPoint(cam: THREE.PerspectiveCamera, img: [number, number]): [number, number] | null {
  const d = new THREE.Vector3(img[0] * 2 - 1, 1 - img[1] * 2, 0.5).unproject(cam).sub(cam.position).normalize();
  if (d.y > -1e-6) return null;
  const t = -cam.position.y / d.y;
  return [cam.position.x + d.x * t, cam.position.z + d.z * t];
}

export interface PipelineRun {
  calibration: CalibrationResult;
  result: BallTrajectoryResult;
  millis: number;
}

/**
 * Run the 3D solver on a fixture with the app's options. The solver logs a
 * lot; that is silenced unless `verbose`.
 */
export function runFixture(
  fx: Fixture,
  rallyParams?: BallReconstructionOptions["rallyParams"],
  verbose = false,
  court?: CourtLabel,
): PipelineRun | null {
  const calibration = calibrate(fx, court);
  if (!calibration) return null;
  const aspect = fx.video.width / fx.video.height;
  const log = console.log, warn = console.warn;
  if (!verbose) { console.log = () => {}; console.warn = () => {}; }
  const t0 = Date.now();
  try {
    // Same options as App.tsx's solverOptions.
    const result = solveBallTrajectories(fx.frames, calibration, aspect, {
      useBallisticSolver: false,
      debugMode: true,
      diameterMode: "min-side",
      videoWidth: fx.video.width,
      videoHeight: fx.video.height,
      trajectoryMode: "physics-reprojection",
      useCalibrationFov: true,
      manualFovYDeg: 30,
      minBallConfidence: 0.6,
      rallyParams,
    });
    return { calibration, result, millis: Date.now() - t0 };
  } finally {
    console.log = log; console.warn = warn;
  }
}
