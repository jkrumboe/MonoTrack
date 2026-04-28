import { useState, useEffect, useCallback, useRef } from "react";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

export interface BallData {
  x: number;
  y: number;
  z: number;
}

export interface FrameData {
  action: "frame";
  frame: number;
  timestamp: number;
  pose_player1: [number, number, number][];
  pose_player2: [number, number, number][];
  ball: BallData;
  ball_detected: boolean;
}

export interface AnalysisStats {
  avg_ball_speed: number;
  total_shots: number;
}

export interface AnalysisComplete {
  action: "complete";
  total_frames: number;
  fps: number;
  score: { player1: number; player2: number };
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
  const framesRef = useRef<FrameData[]>([]);

  useEffect(() => {
    const unlisteners: Promise<UnlistenFn>[] = [];

    unlisteners.push(
      listen<FrameData>("analysis-frame", (event) => {
        framesRef.current = [...framesRef.current, event.payload];
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
    framesRef.current = [];

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
    error,
    startAnalysis,
    stopAnalysis,
  };
}
