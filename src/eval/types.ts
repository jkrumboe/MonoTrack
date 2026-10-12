import type { FrameData } from "../hooks/useAnalysis";

/** Touch types the rally rules know, plus "dig" for later touches. */
export type TouchType = "toss" | "serve" | "reception" | "set" | "attack" | "block" | "dig";

/**
 * Analyzer output for one clip, saved by python/export_fixture.py (or made by
 * synth.ts). Enough to run the 3D solver without video, Python or a GPU.
 */
export interface Fixture {
  clip: string;
  video: { width: number; height: number; fps: number };
  /**
   * Court corners in normalised reference-image coords, in world order
   * (court (x,z): (-4.5,-9), (4.5,-9), (4.5,9), (-4.5,9)). Null when the
   * analyzer found no court: then the labels must supply it.
   */
  courtCorners: number[][] | null;
  /** Frame the corners belong to; the reference image when the camera moves. */
  courtFrame?: number;
  cameraMoving?: boolean;
  frames: FrameData[];
}

/** One human-labelled (or synthetic) touch. */
export interface TouchLabel {
  frame: number;
  type: TouchType;
  /** Toucher's feet in normalised image coords, if marked. */
  feet?: [number, number];
  /** Exact 3D ball position at the touch (synthetic only). */
  position?: [number, number, number];
}

/** One court point placed by hand in the labeller. */
export interface CourtPointLabel {
  id: number;
  name: string;
  /**
   * Court coordinates (x across, z along; net at z = 0), metres, plus the
   * height above the floor for points that are not on it (antennas).
   */
  world: [number, number] | [number, number, number];
  /** Frame it was clicked on, and where (normalised raw image coords). */
  frame: number;
  pt: [number, number];
}

/**
 * Court placed by hand in label_touches.py: 4+ named points (corners, or
 * where the 3 m / centre lines meet the outer lines), possibly on different
 * frames when the camera pans. Older labels have 4 `corners` on one `frame`
 * in image order far-left, far-right, near-right, near-left.
 */
export interface CourtLabel {
  view: "behind" | "side";
  points?: CourtPointLabel[];
  frame?: number;
  corners?: number[][];
}

/** Ground truth for a clip. Real clips have touches only; synthetic ones also have the path. */
export interface Truth {
  clip: string;
  touches: TouchLabel[];
  /** Hand-placed court, preferred over the analyzer's when present. */
  court?: CourtLabel;
  /** frame -> true 3D ball position (synthetic only). */
  path?: Record<number, [number, number, number]>;
}

export interface ClipMetrics {
  clip: string;
  synthetic: boolean;
  touchesTrue: number;
  touchesFound: number;
  /** Found touches matched to a true one within the frame tolerance. */
  touchesMatched: number;
  recall: number;
  precision: number;
  /** Matched touches whose label is right. */
  typeAccuracy: number;
  /** Mean horizontal (court) distance of matched touches to the truth (m). */
  touchPosErrM: number | null;
  /** Mean height error of matched touches (m), synthetic only. */
  touchHeightErrM: number | null;
  /** Mean |found - true| touch frame of matched touches (frames). Reported, not scored. */
  touchFrameErr: number | null;
  /** RMS 3D error over frames that have both truth and a solved position (m). */
  pathRmsM: number | null;
  /** Share of true-path frames that got a 3D position. */
  pathCoverage: number | null;
  /** Median reprojection error of solved positions onto detections (px). */
  reprojMedianPx: number | null;
  /** Combined 0-100 score used for gating and tuning (higher is better). */
  score: number;
}
