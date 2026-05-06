import { useState, useEffect, useCallback, useRef } from "react";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

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

  useEffect(() => {
    const unlisteners: Promise<UnlistenFn>[] = [];

    unlisteners.push(
      listen<FrameData>("analysis-frame", (event) => {
        framesRef.current = [...framesRef.current, event.payload];
        // Collect court corners from frame data if present
        const payload = event.payload;
        if (payload.court_corners) {
          courtDataRef.current = [...courtDataRef.current, {
            frame: payload.frame,
            timestamp: payload.timestamp,
            corners: payload.court_corners,
            keypoints: payload.court_keypoints,
          }];
        }
        // Batch UI updates — update every 50 frames to avoid rendering overhead
        if (framesRef.current.length % 50 === 0) {
          setFrames([...framesRef.current]);
        }
      })
    );

    unlisteners.push(
      listen<ProgressData>("analysis-progress", (event) => {
        setProgress(event.payload.percent);
        setProgressMessage(event.payload.message);
      })
    );

    unlisteners.push(
      listen<AnalysisComplete>("analysis-complete", (event) => {
        setFrames([...framesRef.current]);
        // Use court_data from complete message if available, otherwise use incrementally collected data
        const cd = event.payload.court_data ?? courtDataRef.current;
        setCourtData(cd);
        setResult(event.payload);
        setStatus("complete");
        setProgress(100);
      })
    );

    unlisteners.push(
      listen<{ message: string }>("analysis-error", (event) => {
        setError(event.payload.message);
        setStatus("error");
      })
    );

    return () => {
      unlisteners.forEach((p) => p.then((unlisten) => unlisten()));
    };
  }, []);

  const startAnalysis = useCallback(async (videoPath: string) => {
    setStatus("running");
    setProgress(0);
    setProgressMessage("Starting...");
    setFrames([]);
    setResult(null);
    setError(null);
    setCourtData([]);
    framesRef.current = [];
    courtDataRef.current = [];

    try {
      await invoke("start_analysis", { videoPath });
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }, []);

  const stopAnalysis = useCallback(async () => {
    try {
      await invoke("stop_analysis");
    } catch (e) {
      console.error("Failed to stop analysis:", e);
    }
    setStatus("idle");
  }, []);

  return {
    status,
    progress,
    progressMessage,
    frames,
    result,
    courtData,
    error,
    startAnalysis,
    stopAnalysis,
  };
}
