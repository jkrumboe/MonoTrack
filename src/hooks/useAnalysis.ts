import { useState, useEffect, useCallback, useRef } from "react";
import { getBackend, type AnalysisMessage } from "../platform/backend";

/** Identifies which reconstruction method produced a ball_3d_pos value. */
export type BallReconstructionMethod =
  | "homography-ray"
  | "bbox-depth"
  | "ground-intersection"
  | "bbox-bottom-ray"
  | "ballistic-arc"
  | "physics-reprojection"
  /** Filling a hole in the fitted path, so the line runs through the rally. Not a fit. */
  | "estimated";

export interface FrameData {
  action: "frame";
  frame: number;
  timestamp: number;
  pose_player1?: [number, number, number][];
  pose_player2?: [number, number, number][];
  pose_player1_world?: [number, number, number][];
  pose_player2_world?: [number, number, number][];
  court_corners?: number[][];
  court_keypoints?: (number[] | null)[];
  player1_img_pos?: [number, number];
  player2_img_pos?: [number, number];
  team1_img_positions?: [number, number][];
  team2_img_positions?: [number, number][];
  /** Normalised [0,1] image-space ball centre from YOLO (or interpolated). */
  ball_img_pos?: [number, number];
  /** True when the position was filled in by trajectory interpolation. */
  ball_interpolated?: boolean;
  /** YOLO detection confidence (0 when interpolated). */
  ball_conf?: number;
  /** Normalised [x1,y1,x2,y2] bounding box from YOLO (absent when interpolated). */
  ball_bbox?: [number, number, number, number];
  /**
   * Apparent ball diameter (px) measured from the ball's colour in the frame.
   * TrackNet only; absent when interpolated or the ball could not be segmented.
   */
  ball_diameter_px?: number;
  /**
   * Person boxes near the ball, normalised [x1, y1, x2, y2, conf]. Used to
   * anchor each touch to the player who made it. Detected frames only.
   */
  players_near_ball?: number[][];
  /**
   * Every person in the frame, same format. Taken on every frame, so the
   * players are known when the ball is hidden at a touch.
   */
  players?: number[][];
  /**
   * Arm keypoints per entry of `players` (pose model): [x, y, conf] for the
   * left/right shoulder, elbow and wrist, normalised.
   */
  players_arms?: number[][][];
  /**
   * Camera motion (pan/zoom from a fixed spot): 3x3 homography, row-major,
   * mapping this frame's pixels to the reference frame's pixels. Absent for a
   * static camera. The fields below are the same measurements expressed in the
   * reference image ("virtual static camera"), which is what the 3D solver
   * uses; absent when this frame's motion estimate was not trusted.
   */
  ref_from_frame?: number[];
  ball_ref_pos?: [number, number];
  ball_ref_diameter_px?: number;
  players_ref?: number[][];
  players_all_ref?: number[][];
  players_arms_ref?: number[][][];
  /**
   * Relative depth at ball centre from Depth Anything V2 [0,1] (0=near, 1=far).
   * Present only for YOLO-detected (non-interpolated) frames when the depth model is available.
   */
  ball_depth_estimate?: number;
  /**
   * Physics-solver 3D position [x, y, z] in world metres after ballistic trajectory fitting.
   * Populated in-browser after analysis + calibration are both available.
   */
  ball_3d_pos?: [number, number, number];
  /**
   * Which reconstruction method produced ball_3d_pos.
   * Present only when ball_3d_pos is set.
   */
  ball_3d_method?: BallReconstructionMethod;
}

export interface CourtDataEntry {
  frame: number;
  timestamp: number;
  corners: number[][];
  keypoints?: (number[] | null)[];
}

export interface AnalysisStats {
  avg_ball_speed?: number;
  total_shots?: number;
}

export interface AnalysisComplete {
  action: "complete";
  total_frames: number;
  fps: number;
  court_data?: CourtDataEntry[];
  score: { team1: number; team2: number };
  stats: AnalysisStats;
}

export interface ProgressData {
  action: "progress";
  percent: number;
  message: string;
}

export type AnalysisStatus = "idle" | "running" | "complete" | "error";

export function useAnalysis() {
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [result, setResult] = useState<AnalysisComplete | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [courtData, setCourtData] = useState<CourtDataEntry[]>([]);
  const framesRef = useRef<FrameData[]>([]);
  const courtDataRef = useRef<CourtDataEntry[]>([]);
  // Wall-clock timing of the current/last run
  const startTimeRef = useRef<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [analysisDurationMs, setAnalysisDurationMs] = useState<number | null>(null);
  /** Stops listening to the current run's messages. */
  const unlistenRef = useRef<(() => void) | null>(null);
  const videoRef = useRef("");
  /** Last time the received frames were pushed to the UI. */
  const lastFramesPushRef = useRef(0);

  // Tick the elapsed clock while running
  useEffect(() => {
    if (status !== "running") return;
    const id = window.setInterval(() => {
      if (startTimeRef.current !== null) setElapsedMs(performance.now() - startTimeRef.current);
    }, 500);
    return () => window.clearInterval(id);
  }, [status]);

  useEffect(() => () => unlistenRef.current?.(), []);

  const handleMessage = useCallback((msg: AnalysisMessage) => {
    switch (msg.action) {
      case "frame": {
        framesRef.current.push(msg);
        // Collect court corners from frame data if present
        if (msg.court_corners) {
          courtDataRef.current.push({
            frame: msg.frame,
            timestamp: msg.timestamp,
            corners: msg.court_corners,
            keypoints: msg.court_keypoints,
          });
        }
        // A replayed match delivers thousands of frames at once; copying the
        // array for each would be quadratic, so the UI catches up a few times a second.
        const now = performance.now();
        if (now - lastFramesPushRef.current > 250) {
          lastFramesPushRef.current = now;
          setFrames([...framesRef.current]);
        }
        break;
      }
      case "progress":
        setProgress(msg.percent);
        setProgressMessage(msg.message);
        break;
      case "complete": {
        setFrames([...framesRef.current]);
        // Use court_data from complete message if available, otherwise use incrementally collected data
        setCourtData(msg.court_data ?? courtDataRef.current);
        setResult(msg);
        setStatus("complete");
        setProgress(100);
        if (startTimeRef.current !== null) {
          const ms = performance.now() - startTimeRef.current;
          setElapsedMs(ms);
          setAnalysisDurationMs(ms);
          console.log(
            `[analysis] ${msg.total_frames} frames in ${(ms / 1000).toFixed(1)}s` +
            ` (${(msg.total_frames / (ms / 1000)).toFixed(1)} fps)`
          );
        }
        unlistenRef.current?.();
        unlistenRef.current = null;
        break;
      }
      case "error":
        setError(msg.message);
        setStatus("error");
        unlistenRef.current?.();
        unlistenRef.current = null;
        break;
    }
  }, []);

  /** Analyse a video, or open its finished analysis (server). `rerun` analyses it again. */
  const startAnalysis = useCallback(async (videoPath: string, opts?: { rerun?: boolean }) => {
    unlistenRef.current?.();
    unlistenRef.current = null;
    videoRef.current = videoPath;
    setStatus("running");
    setProgress(0);
    setProgressMessage("Starting...");
    setFrames([]);
    setResult(null);
    setError(null);
    setCourtData([]);
    framesRef.current = [];
    courtDataRef.current = [];
    startTimeRef.current = performance.now();
    setElapsedMs(0);
    setAnalysisDurationMs(null);

    try {
      const backend = await getBackend();
      const unlisten = await backend.analyze(videoPath, handleMessage, opts);
      // Another analysis may have started while this one was starting.
      if (videoRef.current === videoPath) unlistenRef.current = unlisten;
      else unlisten();
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }, [handleMessage]);

  const stopAnalysis = useCallback(async () => {
    unlistenRef.current?.();
    unlistenRef.current = null;
    try {
      await (await getBackend()).stopAnalysis(videoRef.current);
    } catch (e) {
      console.error("Failed to stop analysis:", e);
    }
    setStatus("idle");
  }, []);

  /** Back to the start screen. */
  const reset = useCallback(() => {
    unlistenRef.current?.();
    unlistenRef.current = null;
    videoRef.current = "";
    setStatus("idle");
    setFrames([]);
    setResult(null);
    setError(null);
    setCourtData([]);
  }, []);

  return {
    status,
    progress,
    progressMessage,
    frames,
    result,
    courtData,
    error,
    elapsedMs,
    analysisDurationMs,
    startAnalysis,
    stopAnalysis,
    reset,
  };
}
