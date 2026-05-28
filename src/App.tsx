import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import * as THREE from "three";
import { useAnalysis } from "./hooks/useAnalysis";
import SkeletonViewer from "./components/SkeletonViewer";
import VideoPlayer from "./components/VideoPlayer";
import { type CalibrationResult, solveCameraIterative } from "./utils/cameraSolver";
import CourtKeypointOverlay from "./components/CourtKeypointOverlay";
import CourtCalibration from "./components/CourtCalibration";
import CourtMapView from "./components/CourtMapView";
import ProgressBar from "./components/ProgressBar";
import StatsDisplay from "./components/StatsDisplay";
import PlaybackControls from "./components/PlaybackControls";
import PlayerTrackingOverlay from "./components/PlayerTrackingOverlay";
import BallDebugView from "./components/BallDebugView";
import DepthFilterView from "./components/DepthFilterView";
import { solveBallTrajectories, type BallFrameDebug, type PhysicsReprojectionDebug, type TrajectoryMode } from "./utils/ballReconstruction";
import type { FrameData, BallReconstructionMethod } from "./hooks/useAnalysis";
import "./App.css";

type ViewMode = "side-by-side" | "overlay";

const COURT_KEYPOINTS_3D: [number, number, number][] = [
  [-4.5, 0, -9.0], // far-left corner
  [ 4.5, 0, -9.0], // far-right corner
  [-4.5, 0,  9.0], // near-left corner
  [ 4.5, 0,  9.0], // near-right corner
  [-4.5, 0, -3.0], // far attack left
  [ 4.5, 0, -3.0], // far attack right
  [-4.5, 0,  3.0], // near attack left
  [ 4.5, 0,  3.0], // near attack right
  [-4.5, 0,  0.0], // net left
  [ 4.5, 0,  0.0], // net right
];

/**
 * Convert video-frame-normalized corners [0,1] to container NDC [-1,1],
 * accounting for object-fit:contain letterboxing.
 */
function projectKeypointsFromCalibration(
  calibration: CalibrationResult,
  videoSize: { width: number; height: number },
): (number[] | null)[] {
  const vw = videoSize.width;
  const vh = videoSize.height;
  if (!vw || !vh) return COURT_KEYPOINTS_3D.map(() => null);

  const videoAR = vw / vh;
  const cam = new THREE.PerspectiveCamera(calibration.fov, videoAR, 0.1, 500);
  cam.position.set(...calibration.position);
  cam.lookAt(new THREE.Vector3(...calibration.lookAt));
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();

  return COURT_KEYPOINTS_3D.map(([x, y, z]) => {
    const v = new THREE.Vector3(x, y, z).project(cam);
    // NDC [-1,1] from videoAR camera → video-frame normalized [0,1]
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  });
}

/** Export the solved 3D ball trajectory as a JSON file download. */
async function exportBallTrajectory(exportFrames: FrameData[]): Promise<void> {
  const rows = exportFrames
    .filter((f) => f.ball_3d_pos != null)
    .map((f) => ({
      frame: f.frame,
      timestamp: f.timestamp,
      x: f.ball_3d_pos![0],
      y: f.ball_3d_pos![1],
      z: f.ball_3d_pos![2],
      method: f.ball_3d_method ?? null,
      conf: f.ball_conf ?? null,
      interpolated: f.ball_interpolated ?? false,
    }));
  const json = JSON.stringify(rows, null, 2);
  const path = await save({
    defaultPath: "ball_trajectory.json",
    filters: [{ name: "JSON", extensions: ["json"] }],
  });
  if (path) {
    await invoke("write_text_file", { path, content: json });
  }
}

/** Export all per-frame reconstruction debug records as a JSON file download. */
async function exportDebugJSON(records: BallFrameDebug[]): Promise<void> {
  const json = JSON.stringify(records, null, 2);
  const path = await save({
    defaultPath: "ball_reconstruction_debug.json",
    filters: [{ name: "JSON", extensions: ["json"] }],
  });
  if (path) {
    await invoke("write_text_file", { path, content: json });
  }
}

function App() {
  const {
    status,
    progress,
    progressMessage,
    frames,
    result,
    courtData,
    error,
    startAnalysis,
    stopAnalysis,
  } = useAnalysis();

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoPath, setVideoPath] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [sidebarTab, setSidebarTab] = useState<"stats" | "court" | "ball-debug" | "depth">("stats");
  const [_videoDuration, setVideoDuration] = useState<number | null>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [manualCalibrationLocked, setManualCalibrationLocked] = useState(false);
  const [resetViewTrigger, setResetViewTrigger] = useState(0);
  const [showCalibration, setShowCalibration] = useState(false);
  /** Solved 3D ball positions per frame index from the physics trajectory solver. */
  const [ball3dMap, setBall3dMap] = useState<Map<number, [number, number, number]>>(new Map());
  /** Reconstruction method used per frame index (parallel to ball3dMap). */
  const [ball3dMethods, setBall3dMethods] = useState<Map<number, BallReconstructionMethod>>(new Map());
  /** Per-frame debug records from the last solver run. */
  const [ball3dDebug, setBall3dDebug] = useState<BallFrameDebug[]>([]);
  /** Physics-reprojection diagnostic (only populated in physics-reprojection mode). */
  const [physicsDebug, setPhysicsDebug] = useState<PhysicsReprojectionDebug[] | undefined>(undefined);
  /** True when physics-reprojection mode ran but produced zero fitted arcs. */
  const [physicsFailedWarning, setPhysicsFailedWarning] = useState(false);
  // ── Solver option state ──────────────────────────────────────────────────
  const [trajectoryMode, setTrajectoryMode] = useState<TrajectoryMode>("physics-reprojection");
  const [useCalibrationFov, setUseCalibrationFov] = useState(true);
  const [manualFovYDeg, setManualFovYDeg] = useState(30);
  const [minBallConfidence, setMinBallConfidence] = useState(0.6);
  const [calibrationViewTransform, setCalibrationViewTransform] = useState({ zoom: 1, panX: 0, panY: 0 });
  const [overlayEl, setOverlayEl] = useState<HTMLDivElement | null>(null);
  const [splitVideoEl, setSplitVideoEl] = useState<HTMLDivElement | null>(null);
  const [splitVideoSize, setSplitVideoSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [videoZoomEl, setVideoZoomEl] = useState<HTMLDivElement | null>(null);
  const [videoZoomTransform, setVideoZoomTransform] = useState({ zoom: 1, panX: 0, panY: 0 });
  const videoZoomStateRef = useRef({ zoom: 1, panX: 0, panY: 0 });
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const playIntervalRef = useRef<number | null>(null);
  const prevCalibrationRef = useRef<CalibrationResult | null>(null);
  const lastCalibratedCourtFrameRef = useRef<number | null>(null);

  /** Smooth a new calibration against the previous one to prevent 3D camera jumps. */
  const smoothCalibration = useCallback((newCal: CalibrationResult): CalibrationResult => {
    const prev = prevCalibrationRef.current;
    if (!prev) {
      prevCalibrationRef.current = newCal;
      return newCal;
    }
    const a = 0.35; // blend factor toward new value
    const lerp = (p: number, n: number) => p + a * (n - p);
    const smoothed: CalibrationResult = {
      position: [
        lerp(prev.position[0], newCal.position[0]),
        lerp(prev.position[1], newCal.position[1]),
        lerp(prev.position[2], newCal.position[2]),
      ],
      lookAt: [
        lerp(prev.lookAt[0], newCal.lookAt[0]),
        lerp(prev.lookAt[1], newCal.lookAt[1]),
        lerp(prev.lookAt[2], newCal.lookAt[2]),
      ],
      fov: lerp(prev.fov, newCal.fov),
    };
    prevCalibrationRef.current = smoothed;
    return smoothed;
  }, []);

  const isRealVideo = videoPath !== "" && videoPath !== "demo.mp4";
  const videoAspect = useMemo(() => {
    if (!videoSize.width || !videoSize.height) return undefined;
    return videoSize.width / videoSize.height;
  }, [videoSize]);

  /**
   * After analysis completes and camera calibration is available, run the
   * physics-based ballistic trajectory solver over all detected ball frames.
   * The solver is CPU-intensive so it runs in a timeout to avoid blocking the UI.
   */
  const solverOptions = useMemo(() => ({
    useBallisticSolver:  false as const,
    debugMode:           true,
    diameterMode:        "min-side" as const,
    videoWidth:          videoSize.width  > 0 ? videoSize.width  : undefined,
    videoHeight:         videoSize.height > 0 ? videoSize.height : undefined,
    trajectoryMode,
    useCalibrationFov,
    manualFovYDeg,
    minBallConfidence,
  }), [videoSize, trajectoryMode, useCalibrationFov, manualFovYDeg, minBallConfidence]);

  useEffect(() => {
    if (status !== "complete" || !calibration || !videoAspect || frames.length === 0) return;
    const handle = setTimeout(() => {
      const { positions, methods, stats, debugRecords, physicsDebug: pd } = solveBallTrajectories(
        frames, calibration, videoAspect, solverOptions,
      );
      setBall3dMap(positions);
      setBall3dMethods(methods);
      setBall3dDebug(debugRecords);
      setPhysicsDebug(pd);

      // Warn when physics mode produced zero fitted arcs despite having detections.
      if (solverOptions.trajectoryMode === "physics-reprojection") {
        setPhysicsFailedWarning(positions.size === 0 && stats.detectedFrames > 0);
      } else {
        setPhysicsFailedWarning(false);
      }

      // Console summary: raw counts per method for quick diagnosis.
      const physicsCount   = [...methods.values()].filter((m) => m === "physics-reprojection").length;
      const bboxDepthCount = [...methods.values()].filter((m) => m === "bbox-depth").length;
      const groundCount    = [...methods.values()].filter((m) => m === "ground-intersection").length;
      console.log(
        `[ballSolver] trajectoryMode=${solverOptions.trajectoryMode}` +
        ` total=${methods.size}` +
        ` physics-reprojection=${physicsCount}` +
        ` bbox-depth=${bboxDepthCount}` +
        ` ground-intersection=${groundCount}` +
        ` stats: detected=${stats.detectedFrames} solved=${stats.solved3dFrames}` +
        ` rejected=${stats.rejectedFrames} avgConf=${stats.avgConf.toFixed(3)}`
      );

      // Hard assertion: in physics mode every exported position must be physics-fitted.
      if (solverOptions.trajectoryMode === "physics-reprojection" && methods.size > 0) {
        const nonPhysics = [...methods.values()].filter((m) => m !== "physics-reprojection");
        if (nonPhysics.length > 0) {
          console.error(
            "BUG: physics mode exported non-physics trajectory —",
            nonPhysics.length,
            "frames with methods:",
            [...new Set(nonPhysics)].join(", "),
          );
        }
      }
    }, 0);
    return () => clearTimeout(handle);
  }, [status, calibration, videoAspect, frames, solverOptions]);

  /** Frames augmented with physics-solver 3D positions and methods where available. */
  const solvedFrames = useMemo<FrameData[]>(() => {
    // In physics-reprojection mode, ONLY physics positions are valid.
    // If the solver produced no positions (arc fitting failed), clear all stale
    // ball_3d_pos so the export reflects the failure rather than old bbox-depth data.
    if (trajectoryMode === "physics-reprojection") {
      if (ball3dMap.size === 0) {
        // Physics solver ran but found nothing — return frames with all 3D positions cleared.
        return frames.map((f) => ({ ...f, ball_3d_pos: undefined, ball_3d_method: undefined }));
      }
      return frames.map((f) => {
        const pos = ball3dMap.get(f.frame);
        const method = ball3dMethods.get(f.frame);
        if (pos) return { ...f, ball_3d_pos: pos, ball_3d_method: method };
        // Frame not covered by physics — clear any stale position.
        return { ...f, ball_3d_pos: undefined, ball_3d_method: undefined };
      });
    }
    // Other modes: keep raw frames when solver has no results.
    if (ball3dMap.size === 0) return frames;
    return frames.map((f) => {
      const pos = ball3dMap.get(f.frame);
      const method = ball3dMethods.get(f.frame);
      return pos ? { ...f, ball_3d_pos: pos, ball_3d_method: method } : f;
    });
  }, [frames, ball3dMap, ball3dMethods, trajectoryMode]);

  /** Raw bbox-depth 3D positions for debug cloud in SkeletonViewer. */
  const rawBboxPoints = useMemo<ReadonlyArray<[number, number, number]>>(
    () => ball3dDebug.flatMap((d) => (d.worldPositionRaw ? [d.worldPositionRaw] : [])),
    [ball3dDebug],
  );

  /** Rejected physics-reprojection arcs grouped by arcId, for red trail in SkeletonViewer. */
  const rejectedArcSegments = useMemo<ReadonlyArray<ReadonlyArray<[number, number, number]>>>(() => {
    if (!physicsDebug) return [];
    const arcMap = new Map<number, [number, number, number][]>();
    for (const rec of physicsDebug) {
      if (rec.arcAccepted !== false) continue;
      if (rec.fittedX == null || rec.fittedY == null || rec.fittedZ == null) continue;
      const id = rec.arcId ?? -1;
      if (!arcMap.has(id)) arcMap.set(id, []);
      arcMap.get(id)!.push([rec.fittedX, rec.fittedY, rec.fittedZ]);
    }
    return [...arcMap.values()];
  }, [physicsDebug]);

  useEffect(() => {
    if (!splitVideoEl) {
      setSplitVideoSize({ width: 0, height: 0 });
      return;
    }
    const update = () => {
      setSplitVideoSize({ width: splitVideoEl.clientWidth, height: splitVideoEl.clientHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(splitVideoEl);
    return () => ro.disconnect();
  }, [splitVideoEl]);

  /** Look up the nearest precomputed court corners for a given frame index. */
  const getCornersForFrame = useCallback((frameIdx: number): number[][] | null => {
    if (courtData.length === 0) return null;
    // Find the closest court data entry at or before this frame
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= frameIdx) {
        best = entry;
      } else {
        break; // courtData is sorted by frame
      }
    }
    return best.corners;
  }, [courtData]);

  /** Look up the nearest precomputed court keypoints for a given frame index. */
  const getKeypointsForFrame = useCallback((frameIdx: number): (number[] | null)[] | null => {
    if (courtData.length === 0) return null;
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= frameIdx) {
        best = entry;
      } else {
        break;
      }
    }
    return best.keypoints ?? null;
  }, [courtData]);

  /** Get the nearest court-data sample at or before the current frame. */
  const currentCourtEntry = useMemo(() => {
    if (courtData.length === 0) return null;
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= currentFrame) {
        best = entry;
      } else {
        break;
      }
    }
    return best;
  }, [courtData, currentFrame]);

  const currentKeypoints = (() => {
    if (manualCalibrationLocked && calibration && videoSize.width > 0 && videoSize.height > 0) {
      return projectKeypointsFromCalibration(calibration, videoSize);
    }
    return getKeypointsForFrame(currentFrame);
  })();

  /** Compute calibration from precomputed court corners for the current frame. */
  const updateCalibrationForFrame = useCallback((frameIdx: number) => {
    if (manualCalibrationLocked) return;
    const video = videoElRef.current;
    if (!video || video.readyState < 2) return;

    const corners = getCornersForFrame(frameIdx);
    if (!corners) return;

    const cornerObjs = corners.map(([x, y]: number[]) => ({ x, y }));
    // Court detection returns video-frame-normalised [0,1]; convert to video-frame NDC.
    const ndcPoints = cornerObjs.map(p => ({ x: p.x * 2 - 1, y: -(p.y * 2 - 1) }));
    if (!video.videoWidth || !video.videoHeight) return;
    const aspectRatio = video.videoWidth / video.videoHeight;
    const { result: cal } = solveCameraIterative(ndcPoints, aspectRatio);
    if (cal) {
      setCalibration(smoothCalibration(cal));
    }
  }, [getCornersForFrame, smoothCalibration, manualCalibrationLocked]);

  // Store a reference to the video element when it mounts
  const handleVideoElement = useCallback((el: HTMLVideoElement | null) => {
    videoElRef.current = el;
    if (!el) {
      setVideoSize({ width: 0, height: 0 });
      return;
    }
    setVideoSize({ width: el.videoWidth, height: el.videoHeight });
  }, []);

  const handleCanPlay = useCallback(() => {
    if (manualCalibrationLocked) return;
    // Calibrate from precomputed data if available
    if (courtData.length > 0) {
      updateCalibrationForFrame(currentFrame);
    }
  }, [courtData, currentFrame, updateCalibrationForFrame, manualCalibrationLocked]);

  // When courtData arrives (analysis complete) or video becomes ready, calibrate for frame 0
  useEffect(() => {
    if (courtData.length === 0 || status !== "complete" || !isRealVideo || manualCalibrationLocked) return;

    let cancelled = false;
    const pollId = window.setInterval(() => {
      if (cancelled) return;
      const video = videoElRef.current;
      if (!video) return;

      window.clearInterval(pollId);

      const doCalibrate = () => {
        if (!cancelled) {
          prevCalibrationRef.current = null;
          lastCalibratedCourtFrameRef.current = null;
          updateCalibrationForFrame(0);
        }
      };

      if (video.readyState >= 2) {
        doCalibrate();
      } else {
        const onReady = () => {
          video.removeEventListener("canplay", onReady);
          doCalibrate();
        };
        video.addEventListener("canplay", onReady);
      }
    }, 100);

    return () => {
      cancelled = true;
      window.clearInterval(pollId);
    };
  }, [courtData, status, isRealVideo, updateCalibrationForFrame, manualCalibrationLocked]);

  // Update calibration only when we cross into a new court-data sample frame.
  // This avoids running the expensive camera solver on every playback frame.
  useEffect(() => {
    if (!isRealVideo || manualCalibrationLocked || !currentCourtEntry) return;
    if (lastCalibratedCourtFrameRef.current === currentCourtEntry.frame) return;
    lastCalibratedCourtFrameRef.current = currentCourtEntry.frame;
    updateCalibrationForFrame(currentCourtEntry.frame);
  }, [currentCourtEntry, isRealVideo, manualCalibrationLocked, updateCalibrationForFrame]);

  // Reset calibration, zoom, and ball solver state when video changes
  useEffect(() => {
    prevCalibrationRef.current = null;
    lastCalibratedCourtFrameRef.current = null;
    setCalibration(null);
    setManualCalibrationLocked(false);
    setCalibrationViewTransform({ zoom: 1, panX: 0, panY: 0 });
    videoZoomStateRef.current = { zoom: 1, panX: 0, panY: 0 };
    setVideoZoomTransform({ zoom: 1, panX: 0, panY: 0 });
    setBall3dMap(new Map());
    setBall3dMethods(new Map());
    setBall3dDebug([]);
  }, [videoPath]);

  // Wheel zoom on video pane (non-passive so we can preventDefault)
  useEffect(() => {
    const el = splitVideoEl;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (showCalibration) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const { zoom: prevZoom, panX: prevPanX, panY: prevPanY } = videoZoomStateRef.current;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newZoom = Math.max(1, Math.min(10, prevZoom * factor));
      const scale = newZoom / prevZoom;
      const rawPanX = mouseX - scale * (mouseX - prevPanX);
      const rawPanY = mouseY - scale * (mouseY - prevPanY);
      const newPanX = newZoom <= 1 ? 0 : Math.min(0, Math.max(el.clientWidth * (1 - newZoom), rawPanX));
      const newPanY = newZoom <= 1 ? 0 : Math.min(0, Math.max(el.clientHeight * (1 - newZoom), rawPanY));
      videoZoomStateRef.current = { zoom: newZoom, panX: newPanX, panY: newPanY };
      setVideoZoomTransform({ zoom: newZoom, panX: newPanX, panY: newPanY });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [splitVideoEl, showCalibration]);

  const handleVideoDragStart = useCallback((e: React.MouseEvent) => {
    if (videoZoomStateRef.current.zoom <= 1 || showCalibration) return;
    e.preventDefault();
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      panX: videoZoomStateRef.current.panX,
      panY: videoZoomStateRef.current.panY,
    };
  }, [showCalibration]);

  const handleVideoDragMove = useCallback((e: React.MouseEvent) => {
    if (!isDraggingRef.current || !splitVideoEl) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    const rawPanX = dragStartRef.current.panX + dx;
    const rawPanY = dragStartRef.current.panY + dy;
    const { zoom } = videoZoomStateRef.current;
    const newPanX = Math.min(0, Math.max(splitVideoEl.clientWidth * (1 - zoom), rawPanX));
    const newPanY = Math.min(0, Math.max(splitVideoEl.clientHeight * (1 - zoom), rawPanY));
    videoZoomStateRef.current = { zoom, panX: newPanX, panY: newPanY };
    setVideoZoomTransform({ zoom, panX: newPanX, panY: newPanY });
  }, [splitVideoEl]);

  const handleVideoDragEnd = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  useEffect(() => {
    if (!showCalibration) {
      setCalibrationViewTransform({ zoom: 1, panX: 0, panY: 0 });
    }
  }, [showCalibration]);

  const handleSelectVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "avi", "mov", "mkv"] }],
    });

    if (selected) {
      setVideoPath(selected as string);
      startAnalysis(selected as string);
    }
  }, [startAnalysis]);

  const handleDemoMode = useCallback(() => {
    setVideoPath("demo.mp4");
    startAnalysis("demo.mp4");
  }, [startAnalysis]);

  // Video reports its duration when metadata loads
  const handleMetadataLoaded = useCallback((duration: number) => {
    setVideoDuration(duration);
    const video = videoElRef.current;
    if (video) {
      setVideoSize({ width: video.videoWidth, height: video.videoHeight });
    }
  }, []);

  // Video reports its currentTime during playback — sync 3D frame to it
  const handleVideoTimeUpdate = useCallback(
    (currentTime: number) => {
      if (!result) return;
      const frameIdx = Math.min(
        Math.floor(currentTime * result.fps),
        frames.length - 1
      );
      if (frameIdx >= 0) {
        setCurrentFrame(frameIdx);
      }
    },
    [result, frames.length]
  );

  // Playback loop — only used for demo mode (no real video to drive timing)
  useEffect(() => {
    if (!isPlaying || isRealVideo || frames.length === 0) {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
      return;
    }

    playIntervalRef.current = window.setInterval(() => {
      setCurrentFrame((prev) => {
        if (prev >= frames.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 33); // ~30fps playback

    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, isRealVideo, frames.length]);

  const togglePlay = useCallback(() => {
    if (currentFrame >= frames.length - 1) {
      setCurrentFrame(0);
    }
    setIsPlaying((prev) => !prev);
  }, [currentFrame, frames.length]);

  const videoInnerTransform = useMemo(() => {
    if (showCalibration) {
      return `translate(${calibrationViewTransform.panX}px, ${calibrationViewTransform.panY}px) scale(${calibrationViewTransform.zoom})`;
    }
    if (videoZoomTransform.zoom > 1) {
      return `translate(${videoZoomTransform.panX}px, ${videoZoomTransform.panY}px) scale(${videoZoomTransform.zoom})`;
    }
    return undefined;
  }, [showCalibration, calibrationViewTransform, videoZoomTransform]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1>Volleyball 3D Analyzer</h1>

        </div>
        <div className="header-actions">
          {status === "idle" && (
            <>
              <button className="btn btn-primary" onClick={handleSelectVideo}>
                Open Video
              </button>
              <button className="btn btn-secondary" onClick={handleDemoMode}>
                Demo Mode
              </button>
            </>
          )}
          {status === "running" && (
            <button className="btn btn-danger" onClick={stopAnalysis}>
              Stop Analysis
            </button>
          )}
          {(status === "complete" || status === "error") && (
            <button className="btn btn-primary" onClick={handleSelectVideo}>
              Analyze Another
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {status === "idle" && (
          <div className="welcome">
            <div className="welcome-icon">�</div>
            <h2>Welcome to Volleyball 3D Analyzer</h2>
            <p>
              Open a volleyball match video to analyze player poses, ball
              trajectory, and match statistics.
            </p>
            <p className="hint">
              Or try <strong>Demo Mode</strong> to see sample data without a
              video file.
            </p>
          </div>
        )}

        {status === "running" && (
          <div className="analysis-running">
            <h2>Analyzing Video...</h2>
            <ProgressBar percent={progress} message={progressMessage} />
            <p className="frame-count">{frames.length} frames processed</p>
          </div>
        )}

        {status === "error" && (
          <div className="error-panel">
            <h2>Analysis Error</h2>
            <p className="error-message">{error}</p>
          </div>
        )}

        {status === "complete" && result && (
          <div className="results-layout">
            <div className="viewer-panel">
              <div className="view-toolbar">
                <button
                  className={`view-btn ${viewMode === "side-by-side" ? "active" : ""}`}
                  onClick={() => setViewMode("side-by-side")}
                >
                  Side by Side
                </button>
                <button
                  className={`view-btn ${viewMode === "overlay" ? "active" : ""}`}
                  onClick={() => setViewMode("overlay")}
                >
                  Overlay
                </button>
                {isRealVideo && (status === "complete") && (
                  <button
                    className={`view-btn ${showCalibration ? "active" : ""}`}
                    onClick={() => setShowCalibration((v) => !v)}
                  >
                    {showCalibration ? "Close Calibration" : "Align Court"}
                  </button>
                )}
                {viewMode === "side-by-side" && isRealVideo && calibration && !showCalibration && (
                  <span className="view-btn disabled" style={{ opacity: 0.6, cursor: "default" }}>
                    Court Aligned ✓
                  </span>
                )}
              </div>

              {viewMode === "side-by-side" && (
                <div className="split-view">
                  <div
                    className="split-pane"
                    ref={setSplitVideoEl}
                    style={{
                      overflow: "hidden",
                      cursor: !showCalibration && videoZoomTransform.zoom > 1 ? "grab" : undefined,
                    }}
                    onMouseMove={handleVideoDragMove}
                    onMouseUp={handleVideoDragEnd}
                    onMouseLeave={handleVideoDragEnd}
                  >
                    <div
                      ref={setVideoZoomEl}
                      onMouseDown={handleVideoDragStart}
                      style={{
                        width: "100%",
                        height: "100%",
                        position: "relative",
                        transformOrigin: "0 0",
                        transform: videoInnerTransform,
                      }}
                    >
                      <VideoPlayer
                        videoPath={videoPath}
                        currentFrame={currentFrame}
                        fps={result.fps}
                        isPlaying={isPlaying}
                        onTimeUpdate={handleVideoTimeUpdate}
                        onMetadataLoaded={handleMetadataLoaded}
                        onVideoElement={handleVideoElement}
                        onCanPlay={handleCanPlay}
                      />
                      {currentKeypoints && !showCalibration && (
                        <CourtKeypointOverlay
                          keypoints={currentKeypoints}
                          videoEl={videoElRef.current}
                          containerEl={videoZoomEl}
                        />
                      )}
                      {!showCalibration && frames.length > 0 && (
                        <PlayerTrackingOverlay
                          frames={frames}
                          currentFrame={currentFrame}
                          videoEl={videoElRef.current}
                          containerEl={videoZoomEl}
                        />
                      )}
                      {showCalibration && splitVideoSize.width > 0 && splitVideoSize.height > 0 && (
                        <CourtCalibration
                          width={splitVideoSize.width}
                          height={splitVideoSize.height}
                          videoPath={videoPath}
                          adjustMode={!!calibration}
                          initialCalibration={calibration ?? undefined}
                          onViewTransformChange={setCalibrationViewTransform}
                          onCalibrated={(cal) => {
                            setCalibration(cal);
                            prevCalibrationRef.current = cal;
                            setManualCalibrationLocked(true);
                            setShowCalibration(false);
                          }}
                          onCancel={() => setShowCalibration(false)}
                        />
                      )}
                    </div>
                  </div>
                  <div className="split-pane">
                    <SkeletonViewer
                      calibration={calibration}
                      frames={solvedFrames}
                      currentFrame={currentFrame}
                      videoAspect={videoAspect}
                      courtData={courtData}
                      isRealVideo={isRealVideo}
                      resetViewTrigger={resetViewTrigger}
                      rawBboxPoints={rawBboxPoints}
                      trajectoryMode={trajectoryMode}
                      rejectedArcSegments={rejectedArcSegments}
                    />
                    <button
                      className="reset-view-btn"
                      title="Reset to video-aligned view"
                      onClick={() => setResetViewTrigger((v) => v + 1)}
                    >
                      ⟳ Reset View
                    </button>
                  </div>
                </div>
              )}

              {viewMode === "overlay" && (
                <div className="overlay-view" ref={setOverlayEl}>
                  <div className="overlay-video">
                    <VideoPlayer
                      videoPath={videoPath}
                      currentFrame={currentFrame}
                      fps={result.fps}
                      isPlaying={isPlaying}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onMetadataLoaded={handleMetadataLoaded}
                      onVideoElement={handleVideoElement}
                      onCanPlay={handleCanPlay}
                    />
                  </div>
                  <div className="overlay-3d">
                    <SkeletonViewer
                      overlay
                      calibration={calibration}
                      frames={solvedFrames}
                      currentFrame={currentFrame}
                      videoAspect={videoAspect}
                      courtData={courtData}
                      isRealVideo={isRealVideo}
                      overlayVideoAspect={videoAspect}
                      rawBboxPoints={rawBboxPoints}
                      trajectoryMode={trajectoryMode}
                      rejectedArcSegments={rejectedArcSegments}
                    />
                  </div>
                  {currentKeypoints && (
                    <div className="overlay-lines">
                      <CourtKeypointOverlay
                        keypoints={currentKeypoints}
                        videoEl={videoElRef.current}
                        containerEl={overlayEl}
                      />
                    </div>
                  )}
                  {frames.length > 0 && (
                    <PlayerTrackingOverlay
                      frames={frames}
                      currentFrame={currentFrame}
                      videoEl={videoElRef.current}
                      containerEl={overlayEl}
                    />
                  )}
                </div>
              )}

              <PlaybackControls
                currentFrame={currentFrame}
                totalFrames={frames.length}
                isPlaying={isPlaying}
                onFrameChange={setCurrentFrame}
                onTogglePlay={togglePlay}
              />
            </div>
            <div className="sidebar">
              <div className="sidebar-tabs">
                <button
                  className={`sidebar-tab ${sidebarTab === "stats" ? "active" : ""}`}
                  onClick={() => setSidebarTab("stats")}
                >Stats</button>
                <button
                  className={`sidebar-tab ${sidebarTab === "court" ? "active" : ""}`}
                  onClick={() => setSidebarTab("court")}
                >Court Map</button>
                <button
                  className={`sidebar-tab ${sidebarTab === "ball-debug" ? "active" : ""}`}
                  onClick={() => setSidebarTab("ball-debug")}
                >Ball Debug</button>
                <button
                  className={`sidebar-tab ${sidebarTab === "depth" ? "active" : ""}`}
                  onClick={() => setSidebarTab("depth")}
                >Depth</button>
              </div>

              {sidebarTab === "stats" && <StatsDisplay result={result} />}
              {sidebarTab === "court" && (
                <CourtMapView
                  frames={solvedFrames}
                  currentFrame={currentFrame}
                  courtData={courtData}
                  isRealVideo={isRealVideo}
                  calibration={calibration}
                  videoAspect={videoAspect}
                />
              )}
              {sidebarTab === "ball-debug" && (
                <>
                  {/* ── Solver controls ──────────────────────────────── */}
                  <div className="ball-solver-controls">
                    <div className="control-row">
                      <label htmlFor="traj-mode-select">Trajectory mode</label>
                      <select
                        id="traj-mode-select"
                        value={trajectoryMode}
                        onChange={(e) => setTrajectoryMode(e.target.value as TrajectoryMode)}
                      >
                        <option value="raw-bbox-depth">Raw bbox-depth</option>
                        <option value="smoothed-bbox-depth">Smoothed bbox-depth</option>
                        <option value="physics-reprojection">Physics reprojection</option>
                      </select>
                    </div>
                    <div className="control-row">
                      <label>
                        <input
                          type="checkbox"
                          checked={useCalibrationFov}
                          onChange={(e) => setUseCalibrationFov(e.target.checked)}
                        />
                        {" "}Use calibration FoV{calibration ? ` (${calibration.fov.toFixed(1)}°)` : ""}
                      </label>
                    </div>
                    {!useCalibrationFov && (
                      <div className="control-row fov-row">
                        <label htmlFor="fov-range">Manual FoV: {manualFovYDeg.toFixed(1)}°</label>
                        <input
                          id="fov-range"
                          type="range" min={3} max={90} step={0.5}
                          value={manualFovYDeg}
                          onChange={(e) => setManualFovYDeg(Number(e.target.value))}
                        />
                        <div className="fov-presets">
                          {[8.78, 15, 30, 45, 60].map((p) => (
                            <button key={p} onClick={() => setManualFovYDeg(p)}>{p}°</button>
                          ))}
                        </div>
                      </div>
                    )}
                    <div className="control-row">
                      <label htmlFor="conf-range">
                        Min confidence: {minBallConfidence.toFixed(2)}
                      </label>
                      <input
                        id="conf-range"
                        type="range" min={0} max={1} step={0.05}
                        value={minBallConfidence}
                        onChange={(e) => setMinBallConfidence(Number(e.target.value))}
                      />
                    </div>
                  </div>
                  {physicsFailedWarning && (
                    <div className="physics-failed-warning">
                      ⚠ Physics reprojection failed — no arc could be fitted from the detected ball positions.
                      The 3D trajectory is empty. Check confidence threshold, camera calibration, and ensure
                      at least 4 consecutive ball detections exist.
                    </div>
                  )}
                  <BallDebugView
                    frames={solvedFrames}
                    currentFrame={currentFrame}
                    videoEl={videoElRef.current}
                    calibration={calibration}
                    courtData={courtData}
                    videoAspect={videoAspect}
                    ball3dDebug={ball3dDebug}
                  />
                  {ball3dMap.size > 0 && (
                    <div className="trajectory-export">
                      <button
                        className="btn btn-secondary"
                        onClick={() => exportBallTrajectory(solvedFrames)}
                      >
                        Export Trajectory JSON ({ball3dMap.size} frames)
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => exportDebugJSON(ball3dDebug)}
                        style={{ marginTop: 4 }}
                      >
                        Export Debug JSON ({ball3dDebug.length} records)
                      </button>
                      {physicsDebug && physicsDebug.length > 0 && (
                        <button
                          className="btn btn-secondary"
                          onClick={() => {
                            const json = JSON.stringify(physicsDebug, null, 2);
                            void save({ defaultPath: "physics_reprojection_debug.json", filters: [{ name: "JSON", extensions: ["json"] }] })
                              .then((path) => path ? invoke("write_text_file", { path, content: json }) : undefined);
                          }}
                          style={{ marginTop: 4 }}
                        >
                          Export Physics Debug JSON ({physicsDebug.length} records)
                        </button>
                      )}
                    </div>
                  )}
                </>
              )}
              {sidebarTab === "depth" && (
                <DepthFilterView
                  videoEl={videoElRef.current}
                  calibration={calibration}
                  currentFrame={currentFrame}
                  videoAspect={videoAspect}
                />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
