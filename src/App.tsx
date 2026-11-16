import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as THREE from "three";
import { useAnalysis } from "./hooks/useAnalysis";
import SkeletonViewer from "./components/SkeletonViewer";
import VideoPlayer from "./components/VideoPlayer";
import { aimCamera, type CalibrationResult, solveCameraIterative } from "./utils/cameraSolver";
import CourtKeypointOverlay from "./components/CourtKeypointOverlay";
import CourtCalibration from "./components/CourtCalibration";
import CourtMapView from "./components/CourtMapView";
import ProgressBar from "./components/ProgressBar";
import StatsDisplay from "./components/StatsDisplay";
import PlaybackControls from "./components/PlaybackControls";
import PlayerTrackingOverlay from "./components/PlayerTrackingOverlay";
import BallDebugView from "./components/BallDebugView";
import DepthFilterView from "./components/DepthFilterView";
import RallyInspector, { type SaveState } from "./components/RallyInspector";
import { MatchPanel, Scoreboard } from "./components/MatchTimeline";
import RecordView from "./components/RecordView";
import HomeScreen from "./components/HomeScreen";
import { BrandMark, IconAlert, IconCheck, IconClose, IconEdit, IconLayers, IconRefresh, IconSplit, IconTarget } from "./components/Icons";
import {
  backendKind, canRecord, getBackend, loadSettings, type Backend,
} from "./platform/backend";
import { buildMatchTimeline, nextRallyAfter, rallyAt, type Side } from "./utils/matchTimeline";
import { type BallFrameDebug, type BallTouch, type PhysicsReprojectionDebug, type SolverTrace, type TouchSplitDecision, type TrajectoryMode } from "./utils/ballReconstruction";
import {
  applyBallEdits, applyTouchEdits, EMPTY_EDITS, loadEdits, saveEdits, touchEditsForSolver, type RallyEdits,
} from "./utils/rallyEdits";
import type { FrameData, BallReconstructionMethod } from "./hooks/useAnalysis";
import type { SolverRequest, SolverResponse } from "./utils/solverWorker";
import { historyOf, historyPush, historyRedo, historyUndo, refitEditedPath, type History } from "./utils/rallyRefit";
import "./App.css";

type ViewMode = "side-by-side" | "overlay" | "inspect";

/** Width of the video pane in the side-by-side view, as a share of the stage. */
const DEFAULT_SPLIT_PCT = 56;
const MIN_SPLIT_PCT = 25;

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
  aimCamera(cam, calibration);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();

  return COURT_KEYPOINTS_3D.map(([x, y, z]) => {
    const v = new THREE.Vector3(x, y, z).project(cam);
    // NDC [-1,1] from videoAR camera → video-frame normalized [0,1]
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  });
}

/**
 * Mapping between a video frame and the reference frame for a camera that
 * pans and zooms (from the analyzer's per-frame `ref_from_frame`), in
 * video-normalised [0,1] coords. Frames without a trusted estimate borrow the
 * nearest one within a few frames. Null for a static camera.
 */
function referenceMapping(
  frames: FrameData[],
  frameIdx: number,
): { toReference: (p: { x: number; y: number }) => { x: number; y: number }; fromReference: (p: { x: number; y: number }) => { x: number; y: number } } | null {
  if (!frames.length || !frames.some((f) => f.ref_from_frame)) return null;
  const idx = Math.min(Math.max(0, frameIdx), frames.length - 1);
  let H: number[] | undefined;
  for (let d = 0; d <= 5 && !H; d++) H = frames[idx - d]?.ref_from_frame ?? frames[idx + d]?.ref_from_frame;
  if (!H) return null;
  const M = new THREE.Matrix3().set(H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]);
  const Minv = M.clone().invert();
  const apply = (m: THREE.Matrix3) => (p: { x: number; y: number }) => {
    const v = new THREE.Vector3(p.x, p.y, 1).applyMatrix3(m);
    return { x: v.x / v.z, y: v.y / v.z };
  };
  return { toReference: apply(M), fromReference: apply(Minv) };
}

/**
 * Remember a hand-placed court so later clips of the same match calibrate
 * themselves: the analyzer matches their venue against this frame
 * (python/court_store.py). Stored as the four court corners on the
 * reference frame, the calibration's own image, like a detected court.
 */
function saveCourtForMatch(videoPath: string, cal: CalibrationResult, aspect: number, refFrame: number) {
  const cam = new THREE.PerspectiveCamera(cal.fov, aspect, 0.1, 500);
  aimCamera(cam, cal);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  const corners = [[-4.5, -9], [4.5, -9], [4.5, 9], [-4.5, 9]].map(([x, z]) => {
    const v = new THREE.Vector3(x, 0, z).project(cam);
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  });
  getBackend()
    .then((b) => b.saveCourt(videoPath, refFrame, corners))
    .then((r) => console.log("[court] saved for this match:", r))
    .catch((e) => console.warn("[court] could not save court:", e));
}

/** Overlay showing manual ball-centre correction markers on the video pane. */
function BallCorrectionOverlay({
  videoEl,
  containerEl,
  corrections,
  currentFrame,
}: Readonly<{
  videoEl: HTMLVideoElement;
  containerEl: HTMLDivElement;
  corrections: Map<number, [number, number]>;
  currentFrame: number;
}>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cw = containerEl.clientWidth;
    const ch = containerEl.clientHeight;
    canvas.width = cw;
    canvas.height = ch;
    ctx.clearRect(0, 0, cw, ch);

    const vw = videoEl.videoWidth || 1;
    const vh = videoEl.videoHeight || 1;
    const videoAR = vw / vh;
    const containerAR = cw / ch;
    let renderW: number, renderH: number, offsetX: number, offsetY: number;
    if (videoAR > containerAR) {
      renderW = cw; renderH = cw / videoAR; offsetX = 0; offsetY = (ch - renderH) / 2;
    } else {
      renderH = ch; renderW = ch * videoAR; offsetX = (cw - renderW) / 2; offsetY = 0;
    }

    for (const [frame, [nx, ny]] of corrections) {
      const px = offsetX + nx * renderW;
      const py = offsetY + ny * renderH;
      const isCurrent = frame === currentFrame;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.strokeStyle = isCurrent ? "#00ff88" : "#ffff00";
      ctx.lineWidth = 2;
      ctx.stroke();
      // Crosshair
      const S = 10;
      ctx.beginPath();
      ctx.moveTo(px - S, py); ctx.lineTo(px + S, py);
      ctx.moveTo(px, py - S); ctx.lineTo(px, py + S);
      ctx.strokeStyle = isCurrent ? "#00ff88" : "#ffff00";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Frame label
      ctx.fillStyle = isCurrent ? "#00ff88" : "#ffff00";
      ctx.font = "10px monospace";
      ctx.fillText(`f${frame}`, px + 9, py - 5);
    }
  }, [videoEl, containerEl, corrections, currentFrame]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    />
  );
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
    elapsedMs,
    analysisDurationMs,
    startAnalysis,
    stopAnalysis,
    reset: resetAnalysis,
  } = useAnalysis();

  /** Where analysis runs: this machine (desktop app) or the analysis server (web, phone). */
  const [backend, setBackend] = useState<Backend | null>(null);
  const [settings, setSettings] = useState(loadSettings);
  useEffect(() => {
    let cancelled = false;
    getBackend().then((b) => { if (!cancelled) setBackend(b); });
    return () => { cancelled = true; };
  }, [settings]);
  const onServer = backendKind() === "server";
  /** The record screen, or the analysis (default; the start screen until a video is open). */
  const [screen, setScreen] = useState<"analysis" | "record">("analysis");
  /** Upload in progress (0..1), before the analysis starts. */
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  /** Touch kinds picked in the Match tab; all when empty. */
  const [matchKinds, setMatchKinds] = useState<Set<string>>(new Set());
  /** Playback jumps over the breaks between rallies. */
  const [ralliesOnly, setRalliesOnly] = useState(false);

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoPath, setVideoPath] = useState("");
  /** The match's name when it is known; a server video's reference does not tell it. */
  const [matchName, setMatchName] = useState("");
  const [splitPct, setSplitPct] = useState(DEFAULT_SPLIT_PCT);
  /** Surfaces failures from the file dialog, which otherwise fail silently. */
  const [openError, setOpenError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [sidebarTab, setSidebarTab] = useState<"match" | "stats" | "court" | "ball-debug" | "depth">("match");
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
  /** Player touches (serve, reception, set, ...) placed by the rally solver. */
  const [ballTouches, setBallTouches] = useState<BallTouch[]>([]);
  /** The solver's touch split decisions (for the problems list). */
  const [touchSplits, setTouchSplits] = useState<TouchSplitDecision[] | undefined>(undefined);
  /** How the solver put the path together, for the developer view in Inspect & Edit. */
  const [solverTrace, setSolverTrace] = useState<SolverTrace | undefined>(undefined);
  /** A solve is running in the worker. */
  const [solving, setSolving] = useState(false);
  /** True when physics-reprojection mode ran but produced zero fitted arcs. */
  const [physicsFailedWarning, setPhysicsFailedWarning] = useState(false);
  // ── Solver option state ──────────────────────────────────────────────────
  // Fixed: physics reprojection is the only mode that can place the ball in
  // depth from a single camera, so it is no longer user-selectable.
  const trajectoryMode: TrajectoryMode = "physics-reprojection";
  // The solved calibration always supplies the FoV; there is no manual override
  // now that the calibration is reliable.
  const useCalibrationFov = true;
  const manualFovYDeg = 30;
  const [minBallConfidence, setMinBallConfidence] = useState(0.6);
  /** Manual 2D correction mode — clicking the video sets ball centre for current frame. */
  const [correctionMode, setCorrectionMode] = useState(false);
  /**
   * Hand corrections for this video (ball positions, touches, dismissed
   * problems), saved per video and applied on every analysis of it.
   */
  const [editHistory, setEditHistory] = useState<History<RallyEdits>>(() => historyOf(EMPTY_EDITS));
  const edits = editHistory.present;
  /** Edits for the current video were loaded, so changes from here on are the user's and get saved. */
  const editsReadyRef = useRef(false);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  /** Ball-centre corrections as a map, for the Ball Debug tab's overlay. */
  const manualBallCorrections = useMemo(
    () => new Map(Object.entries(edits.ball).flatMap(([f, p]) => (p ? [[Number(f), p] as [number, [number, number]]] : []))),
    [edits.ball],
  );
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
  const lastCalibratedCourtFrameRef = useRef<number | null>(null);

  /** Smooth a new calibration against the previous one to prevent 3D camera jumps. */
  const isRealVideo = videoPath !== "" && videoPath !== "demo.mp4";
  const videoAspect = useMemo(() => {
    if (!videoSize.width || !videoSize.height) return undefined;
    return videoSize.width / videoSize.height;
  }, [videoSize]);

  /** Frames with the hand ball corrections applied (passed to the solver). */
  const framesForSolver = useMemo(() => applyBallEdits(frames, edits.ball), [frames, edits.ball]);
  const solverTouchEdits = useMemo(() => touchEditsForSolver(edits.touches, edits.pins), [edits.touches, edits.pins]);

  /**
   * After analysis completes and camera calibration is available, run the
   * physics-based ballistic trajectory solver over all detected ball frames.
   * The solver is CPU-intensive so it runs in a Web Worker, keeping the UI live.
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
    touchEdits:          solverTouchEdits,
  }), [videoSize, minBallConfidence, solverTouchEdits]);

  useEffect(() => {
    if (status !== "complete" || !calibration || !videoAspect || frames.length === 0) return;
    // One worker per solve: when the inputs change mid-solve (a manual ball
    // correction, say) the stale solve is terminated rather than awaited.
    const worker = new Worker(new URL("./utils/solverWorker.ts", import.meta.url), { type: "module" });
    setSolving(true);
    worker.onmessage = (e: MessageEvent<SolverResponse>) => {
      worker.terminate();
      setSolving(false);
      if ("error" in e.data) {
        console.error("[ballSolver] failed:", e.data.error);
        return;
      }
      const { positions, methods, stats, debugRecords, physicsDebug: pd, touches, touchSplits: splits, trace } = e.data.result;
      setSolverTrace(trace);
      setBall3dMap(positions);
      setBall3dMethods(methods);
      setBall3dDebug(debugRecords);
      setPhysicsDebug(pd);
      setBallTouches(touches ?? []);
      setTouchSplits(splits);

      // Warn when physics mode produced zero fitted arcs despite having detections.
      if (solverOptions.trajectoryMode === "physics-reprojection") {
        setPhysicsFailedWarning(![...methods.values()].includes("physics-reprojection") && stats.detectedFrames > 0);
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
        const nonPhysics = [...methods.values()].filter((m) => m !== "physics-reprojection" && m !== "estimated");
        if (nonPhysics.length > 0) {
          console.error(
            "BUG: physics mode exported non-physics trajectory —",
            nonPhysics.length,
            "frames with methods:",
            [...new Set(nonPhysics)].join(", "),
          );
        }
      }
    };
    worker.onerror = (e) => { setSolving(false); console.error("[ballSolver] worker error:", e.message); };
    worker.postMessage({ id: 0, args: [framesForSolver, calibration, videoAspect, solverOptions] } satisfies SolverRequest);
    return () => { worker.terminate(); setSolving(false); };
  }, [status, calibration, videoAspect, framesForSolver, solverOptions]);

  /** The solver's touches with the hand touch corrections applied. */
  const displayTouches = useMemo(
    () => applyTouchEdits(ballTouches, edits, ball3dMap, result?.fps ?? 30),
    [ballTouches, edits, ball3dMap, result?.fps],
  );

  /**
   * The path as edited: flights at a moved, added or removed touch, or with
   * a reshaped arc, are rebuilt here at once (see utils/rallyRefit); the rest
   * is the solver's. The background solve then catches up with the same edits.
   */
  const refit = useMemo(() => {
    const moving = framesForSolver.some((f) => f.ref_from_frame);
    const observations = new Map<number, [number, number]>();
    for (const f of framesForSolver) {
      const q = moving ? f.ball_ref_pos : f.ball_img_pos;
      if (q && !f.ball_interpolated) observations.set(f.frame, [q[0] * 2 - 1, 1 - q[1] * 2]);
    }
    let project: ((q: [number, number, number]) => [number, number]) | undefined;
    if (calibration && videoAspect && videoAspect > 0) {
      const cam = new THREE.PerspectiveCamera(calibration.fov, videoAspect, 0.1, 500);
      aimCamera(cam, calibration);
      cam.updateMatrixWorld();
      cam.updateProjectionMatrix();
      const v = new THREE.Vector3();
      project = (q) => { v.set(q[0], q[1], q[2]).project(cam); return [v.x, v.y]; };
    }
    return refitEditedPath({
      touches: displayTouches, positions: ball3dMap, methods: ball3dMethods, fps: result?.fps ?? 30,
      arcs: edits.arcs,
      removedAt: Object.entries(edits.touches).flatMap(([f, k]) => (k === "none" ? [Number(f)] : [])),
      observations, project,
    });
  }, [displayTouches, ball3dMap, ball3dMethods, result?.fps, edits.arcs, edits.touches, framesForSolver, calibration, videoAspect]);
  const shownBall3dMap = refit.positions;
  const shownBall3dMethods = refit.methods;

  /** Positions the solver fitted, without the estimates that keep the line going. */
  const fittedBall3dMap = useMemo(
    () => new Map([...shownBall3dMap].filter(([f]) => shownBall3dMethods.get(f) !== "estimated")),
    [shownBall3dMap, shownBall3dMethods],
  );

  /** Rallies, points and score, from the touches (see utils/matchTimeline). */
  const timeline = useMemo(
    () => buildMatchTimeline({
      touches: displayTouches,
      positions: fittedBall3dMap,
      fps: result?.fps ?? 30,
      totalFrames: frames.length,
      pointOverrides: edits.points,
    }),
    [displayTouches, fittedBall3dMap, result?.fps, frames.length, edits.points],
  );

  /** Frames augmented with physics-solver 3D positions and methods where available. */
  const solvedFrames = useMemo<FrameData[]>(() => {
    // In physics-reprojection / short-arc-debug mode, ONLY arc-fitted positions are valid.
    // Rejected or unfitted arcs must not bleed stale bbox-depth positions into the output.
    if (trajectoryMode === "physics-reprojection" || trajectoryMode === "short-arc-debug") {
      if (shownBall3dMap.size === 0) {
        // Solver ran but found nothing — return frames with all 3D positions cleared.
        return frames.map((f) => ({ ...f, ball_3d_pos: undefined, ball_3d_method: undefined }));
      }
      return frames.map((f) => {
        const pos = shownBall3dMap.get(f.frame);
        const method = shownBall3dMethods.get(f.frame);
        if (pos) return { ...f, ball_3d_pos: pos, ball_3d_method: method };
        // Frame not covered by accepted arc — clear any stale position.
        return { ...f, ball_3d_pos: undefined, ball_3d_method: undefined };
      });
    }
    // Other modes: keep raw frames when solver has no results.
    if (shownBall3dMap.size === 0) return frames;
    return frames.map((f) => {
      const pos = shownBall3dMap.get(f.frame);
      const method = shownBall3dMethods.get(f.frame);
      return pos ? { ...f, ball_3d_pos: pos, ball_3d_method: method } : f;
    });
  }, [frames, shownBall3dMap, shownBall3dMethods, trajectoryMode]);


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

  // Pan/zoom camera: this frame's mapping to the reference frame (the one the
  // calibration and court keypoints belong to), and back. Null when static.
  const frameMapping = useMemo(() => referenceMapping(frames, currentFrame), [frames, currentFrame]);

  const currentKeypoints = (() => {
    const kps = manualCalibrationLocked && calibration && videoSize.width > 0 && videoSize.height > 0
      ? projectKeypointsFromCalibration(calibration, videoSize)
      : getKeypointsForFrame(currentFrame);
    if (!frameMapping || !kps) return kps;
    return kps.map((k) => {
      if (!k) return k;
      const p = frameMapping.fromReference({ x: k[0], y: k[1] });
      return [p.x, p.y];
    });
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
      // No temporal smoothing. The court is a single consensus detection for
      // the whole clip, so there is nothing to smooth across — and lerping
      // position, lookAt and fov independently does not interpolate a camera
      // pose, it produces one that satisfies none of them. That was turning an
      // exact solve into a ~300 px reprojection error.
      setCalibration(cal);
    }
  }, [getCornersForFrame, manualCalibrationLocked]);

  // Store a reference to the video element when it mounts
  const handleVideoElement = useCallback((el: HTMLVideoElement | null) => {
    videoElRef.current = el;
    if (!el) {
      setVideoSize({ width: 0, height: 0 });
      return;
    }
    setVideoSize({ width: el.videoWidth, height: el.videoHeight });
  }, []);

  // canplay fires after every seek. Only recalibrate when the court sample
  // actually changes — a new calibration object reruns the whole ball solver,
  // which is what made scrubbing the timeline crawl.
  const handleCanPlay = useCallback(() => {
    if (manualCalibrationLocked || !currentCourtEntry) return;
    if (calibration && lastCalibratedCourtFrameRef.current === currentCourtEntry.frame) return;
    lastCalibratedCourtFrameRef.current = currentCourtEntry.frame;
    updateCalibrationForFrame(currentCourtEntry.frame);
  }, [calibration, currentCourtEntry, updateCalibrationForFrame, manualCalibrationLocked]);

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
    lastCalibratedCourtFrameRef.current = null;
    setCalibration(null);
    setManualCalibrationLocked(false);
    setCalibrationViewTransform({ zoom: 1, panX: 0, panY: 0 });
    videoZoomStateRef.current = { zoom: 1, panX: 0, panY: 0 };
    setVideoZoomTransform({ zoom: 1, panX: 0, panY: 0 });
    setBall3dMap(new Map());
    setBall3dMethods(new Map());
    setBall3dDebug([]);
    setBallTouches([]);
    setTouchSplits(undefined);
    setCorrectionMode(false);
  }, [videoPath]);

  // Load this video's saved corrections. Until they are in, nothing is saved,
  // so an empty start cannot overwrite them.
  useEffect(() => {
    editsReadyRef.current = false;
    setEditHistory(historyOf(EMPTY_EDITS));
    setSaveState("idle");
    if (!isRealVideo) return;
    let cancelled = false;
    loadEdits(videoPath)
      .then((e) => { if (!cancelled) setEditHistory(historyOf(e)); })
      .catch((err) => console.warn("[edits] could not load:", err))
      .finally(() => { if (!cancelled) editsReadyRef.current = true; });
    return () => { cancelled = true; };
  }, [videoPath, isRealVideo]);

  /** Change the edits; they are saved shortly after. */
  const updateEdits = useCallback((change: (e: RallyEdits) => RallyEdits) => {
    if (!editsReadyRef.current && isRealVideo) return;
    setEditHistory((h) => historyPush(h, change(h.present)));
    if (isRealVideo) setSaveState("saving");
  }, [isRealVideo]);
  /** Step back or forward through the edits made in this session. */
  const stepEdits = useCallback((dir: -1 | 1) => {
    setEditHistory((h) => (dir < 0 ? historyUndo(h) : historyRedo(h)));
    if (isRealVideo) setSaveState("saving");
  }, [isRealVideo]);

  useEffect(() => {
    if (saveState !== "saving" || !isRealVideo) return;
    const id = window.setTimeout(() => {
      saveEdits(videoPath, edits)
        .then(() => { setSaveState("saved"); setSaveError(null); })
        .catch((err) => { setSaveState("error"); setSaveError(String(err)); });
    }, 500);
    return () => window.clearTimeout(id);
  }, [edits, saveState, videoPath, isRealVideo]);

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

  /** Handle click on the video pane in correction mode: record ball centre. */
  const handleVideoPaneClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!correctionMode) return;
    const videoEl = videoElRef.current;
    if (!videoEl) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const cw = rect.width;
    const ch = rect.height;
    const vw = videoEl.videoWidth || 1;
    const vh = videoEl.videoHeight || 1;
    // Compute object-fit:contain letterbox rect
    const videoAR = vw / vh;
    const containerAR = cw / ch;
    let displayW: number, displayH: number, offsetX: number, offsetY: number;
    if (containerAR > videoAR) {
      displayH = ch; displayW = ch * videoAR;
      offsetX = (cw - displayW) / 2; offsetY = 0;
    } else {
      displayW = cw; displayH = cw / videoAR;
      offsetX = 0; offsetY = (ch - displayH) / 2;
    }
    const nx = (clickX - offsetX) / displayW;
    const ny = (clickY - offsetY) / displayH;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return; // clicked letterbox
    updateEdits((e) => ({ ...e, ball: { ...e.ball, [currentFrame]: [nx, ny] } }));
  }, [correctionMode, currentFrame, updateEdits]);

  /** Calibration log: project court keypoints through camera and measure reprojection error. */
  const calibrationLog = useMemo(() => {
    if (!calibration || !videoSize.width || !videoSize.height) return null;
    const effectiveFovY = useCalibrationFov ? calibration.fov : manualFovYDeg;
    const vw = videoSize.width;
    const vh = videoSize.height;
    const aspect = vw / vh;
    const cam = new THREE.PerspectiveCamera(effectiveFovY, aspect, 0.1, 500);
    aimCamera(cam, calibration);
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();

    const detectedKps = currentCourtEntry?.keypoints ?? null;
    const errors: number[] = [];
    if (detectedKps) {
      for (let i = 0; i < COURT_KEYPOINTS_3D.length; i++) {
        const kp2d = detectedKps[i];
        if (!kp2d) continue;
        const [wx, wy, wz] = COURT_KEYPOINTS_3D[i];
        const proj = new THREE.Vector3(wx, wy, wz).project(cam);
        const projNx = (proj.x + 1) / 2;
        const projNy = (1 - proj.y) / 2;
        const dx = (projNx - kp2d[0]) * vw;
        const dy = (projNy - kp2d[1]) * vh;
        errors.push(Math.hypot(dx, dy));
      }
    }
    const avgKpErr = errors.length > 0 ? errors.reduce((s, e) => s + e, 0) / errors.length : null;
    const maxKpErr = errors.length > 0 ? Math.max(...errors) : null;
    return {
      fovY: effectiveFovY,
      position: calibration.position,
      lookAt: calibration.lookAt,
      courtKpErrors: errors,
      avgKpErr,
      maxKpErr,
    };
  }, [calibration, videoSize, useCalibrationFov, manualFovYDeg, currentCourtEntry]);

  /** Open a video: pick a local file (desktop), or upload one to the analysis server. */
  const handleSelectVideo = useCallback(async () => {
    setOpenError(null);
    if (onServer) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const selected = await (await getBackend()).pickLocalVideo?.();
      if (selected) {
        setScreen("analysis");
        setVideoPath(selected);
        startAnalysis(selected);
      }
    } catch (e) {
      console.error("Failed to open video picker:", e);
      setOpenError(`Could not open the file picker: ${String(e)}`);
    }
  }, [startAnalysis, onServer]);

  const handleFileChosen = useCallback(async (file: File) => {
    setOpenError(null);
    setScreen("analysis");
    setMatchName(file.name);
    setUploadProgress(0);
    try {
      const b = await getBackend();
      if (!b.uploadVideo) throw new Error("This backend cannot upload videos.");
      const video = await b.uploadVideo(file, { name: file.name, onProgress: setUploadProgress });
      setVideoPath(video);
      startAnalysis(video);
    } catch (e) {
      setOpenError(`Upload failed: ${String(e)}`);
      setMatchName("");
    } finally {
      setUploadProgress(null);
    }
  }, [startAnalysis]);

  /** Open a match from the server: replays its analysis, or follows one in progress. */
  const openMatch = useCallback((video: string, name = "") => {
    setOpenError(null);
    setMatchName(name);
    setScreen("analysis");
    setVideoPath(video);
    startAnalysis(video);
  }, [startAnalysis]);

  const goHome = useCallback(() => {
    if (status === "running" && !onServer) void stopAnalysis();
    resetAnalysis();
    setVideoPath("");
    setMatchName("");
    setOpenError(null);
    setScreen("analysis");
  }, [status, onServer, stopAnalysis, resetAnalysis]);

  const setPointWinner = useCallback((rallyStart: number, side: Side | null) => {
    updateEdits((e) => {
      const points = { ...(e.points ?? {}) };
      if (side) points[rallyStart] = side;
      else delete points[rallyStart];
      return { ...e, points };
    });
  }, [updateEdits]);

  const renameTeam = useCallback((side: Side, name: string) => {
    updateEdits((e) => ({ ...e, teams: { ...(e.teams ?? {}), [side]: name } }));
  }, [updateEdits]);

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
    if (!isPlaying && (currentFrame >= frames.length - 1 || videoElRef.current?.ended)) {
      setCurrentFrame(0);
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying, currentFrame, frames.length]);

  const stopPlayback = useCallback(() => setIsPlaying(false), []);

  // "Play rallies only": while playing, jump over the breaks between rallies.
  useEffect(() => {
    if (!ralliesOnly || !isPlaying || !isRealVideo || !result || timeline.rallies.length === 0) return;
    if (rallyAt(timeline.rallies, currentFrame)) return;
    const next = nextRallyAfter(timeline.rallies, currentFrame);
    if (!next) {
      setIsPlaying(false);
      return;
    }
    // The video owns the clock while playing, so move the video itself.
    const video = videoElRef.current;
    if (video) video.currentTime = (next.clipStart + 0.5) / result.fps;
    setCurrentFrame(next.clipStart);
  }, [ralliesOnly, isPlaying, isRealVideo, result, timeline, currentFrame]);

  // Scrubbing or stepping always pauses, so the video doesn't fight the slider.
  // The slider can fire several input events per screen refresh; re-rendering
  // the whole app for each is wasted work, so apply at most one per frame.
  const pendingScrubRef = useRef<number | null>(null);
  const scrubRafRef = useRef(0);
  const seekToFrame = useCallback(
    (frame: number) => {
      setIsPlaying(false);
      pendingScrubRef.current = Math.max(0, Math.min(frame, frames.length - 1));
      if (scrubRafRef.current) return;
      scrubRafRef.current = requestAnimationFrame(() => {
        scrubRafRef.current = 0;
        if (pendingScrubRef.current !== null) setCurrentFrame(pendingScrubRef.current);
        pendingScrubRef.current = null;
      });
    },
    [frames.length]
  );
  useEffect(() => () => cancelAnimationFrame(scrubRafRef.current), []);

  // Keyboard: Space toggles playback, ←/→ step one frame (Shift = 10 frames)
  useEffect(() => {
    if (frames.length === 0) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          (target.tagName === "INPUT" && (target as HTMLInputElement).type !== "range"))
      ) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
        e.preventDefault();
        const step = (e.shiftKey ? 10 : 1) * (e.code === "ArrowRight" ? 1 : -1);
        seekToFrame(currentFrame + step);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [frames.length, currentFrame, togglePlay, seekToFrame]);

  const videoInnerTransform = useMemo(() => {
    if (showCalibration) {
      return `translate(${calibrationViewTransform.panX}px, ${calibrationViewTransform.panY}px) scale(${calibrationViewTransform.zoom})`;
    }
    if (videoZoomTransform.zoom > 1) {
      return `translate(${videoZoomTransform.panX}px, ${videoZoomTransform.panY}px) scale(${videoZoomTransform.zoom})`;
    }
    return undefined;
  }, [showCalibration, calibrationViewTransform, videoZoomTransform]);

  /** The court alignment editor, laid over a video pane of this size (side by side, or Inspect & edit). */
  const renderCalibration = (width: number, height: number) => (
    <CourtCalibration
      width={width}
      height={height}
      videoPath={videoPath}
      adjustMode={!!calibration}
      initialCalibration={calibration ?? undefined}
      onViewTransformChange={setCalibrationViewTransform}
      onCalibrated={(cal) => {
        setCalibration(cal);
        setManualCalibrationLocked(true);
        setShowCalibration(false);
        if (isRealVideo && videoAspect) saveCourtForMatch(videoPath, cal, videoAspect, courtData[0]?.frame ?? 0);
      }}
      onCancel={() => setShowCalibration(false)}
      toReference={frameMapping?.toReference}
      fromReference={frameMapping?.fromReference}
    />
  );

  const matchTitle = matchName
    || (videoPath === "demo.mp4" ? "Demo data" : videoPath && !videoPath.startsWith("job:") ? videoPath.split(/[\\/]/).pop() ?? "" : videoPath ? "Match" : "");
  const inMatch = screen === "analysis" && status === "complete" && !!result;
  const diagnosticsTab = sidebarTab === "ball-debug" || sidebarTab === "depth";

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <button className={`brand ${matchTitle ? "has-match" : ""}`} onClick={goHome} title="Home">
            <BrandMark />
            <span className="brand-name">Volleyball Analyzer</span>
          </button>
          {matchTitle && screen === "analysis" && (
            <>
              <span className="crumb-sep" aria-hidden>/</span>
              <span className="crumb" title={matchTitle}>{matchTitle}</span>
            </>
          )}
        </div>
        <div className="header-actions">
          {status === "complete" && screen === "analysis" && onServer && videoPath.startsWith("job:") && (
            <button className="btn btn-ghost" onClick={() => startAnalysis(videoPath, { rerun: true })}>
              <IconRefresh size={14} /><span className="btn-label">Re-analyse</span>
            </button>
          )}
          {inMatch && (
            <button className="btn btn-secondary" onClick={goHome} title="Close this match and go back to the start">
              <IconClose size={14} /><span className="btn-label">Close match</span>
            </button>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void handleFileChosen(f);
          }}
        />
      </header>

      <main className="app-main">
        {screen === "record" && (
          <RecordView backend={backend} onUploaded={openMatch} onClose={() => setScreen("analysis")} />
        )}

        {screen === "analysis" && status === "idle" && uploadProgress === null && (
          <HomeScreen
            onServer={onServer}
            canRecord={canRecord()}
            backend={backend}
            error={openError}
            settings={settings}
            onSettingsChange={setSettings}
            onRecord={() => setScreen("record")}
            onSelectVideo={handleSelectVideo}
            onDemo={handleDemoMode}
            onOpenMatch={openMatch}
          />
        )}

        {screen === "analysis" && uploadProgress !== null && (
          <div className="task-card">
            <h2>Uploading video</h2>
            {matchTitle && <p className="task-subject">{matchTitle}</p>}
            <ProgressBar percent={Math.round(uploadProgress * 100)} message="Sending the video to the analysis server" />
          </div>
        )}

        {screen === "analysis" && status === "running" && (
          <div className="task-card">
            <h2>Analysing match</h2>
            {matchTitle && <p className="task-subject">{matchTitle}</p>}
            <ProgressBar percent={progress} message={progressMessage} elapsedMs={elapsedMs} />
            {/* Frames arrive all at once at the end; until then the progress message counts them. */}
            {frames.length > 0 && <p className="frame-count">{frames.length} frames received</p>}
            <div className="task-actions">
              {onServer && (
                <button className="btn btn-secondary" onClick={goHome} title="The server carries on; open the match later from Matches">
                  Leave running
                </button>
              )}
              <button className="btn btn-danger" onClick={stopAnalysis}>Stop analysis</button>
            </div>
          </div>
        )}

        {screen === "analysis" && status === "error" && (
          <div className="task-card task-card-error">
            <h2><IconAlert size={20} />Analysis failed</h2>
            <p className="error-message">{error}</p>
            <div className="task-actions">
              <button className="btn btn-secondary" onClick={goHome}>Back to start</button>
              {videoPath && <button className="btn btn-primary" onClick={() => startAnalysis(videoPath)}>Try again</button>}
            </div>
          </div>
        )}

        {screen === "analysis" && status === "complete" && result && (
          <div className="results-layout">
            <div className="viewer-panel">
              <div className="view-toolbar">
                <div className="segmented" role="group" aria-label="View">
                  <button
                    className={`view-btn ${viewMode === "side-by-side" ? "active" : ""}`}
                    onClick={() => setViewMode("side-by-side")}
                  >
                    <IconSplit size={14} />Side by side
                  </button>
                  <button
                    className={`view-btn ${viewMode === "overlay" ? "active" : ""}`}
                    onClick={() => { setViewMode("overlay"); setShowCalibration(false); }}
                  >
                    <IconLayers size={14} />Overlay
                  </button>
                  <button
                    className={`view-btn ${viewMode === "inspect" ? "active" : ""}`}
                    onClick={() => { setViewMode("inspect"); setShowCalibration(false); }}
                    title="Check the rally and correct the ball path and touches"
                  >
                    <IconEdit size={14} />Inspect &amp; edit
                  </button>
                </div>
                <div className="view-toolbar-status">
                  {solving && (
                    <span className="status-chip"><span className="spinner" />Solving ball path</span>
                  )}
                  {isRealVideo && (
                    <button
                      className={`court-status ${showCalibration ? "open" : calibration ? "ok" : "warn"}`}
                      onClick={() => {
                        // Inspect & edit aligns the court over its own video; the other views in the side-by-side one.
                        if (viewMode !== "inspect") setViewMode("side-by-side");
                        setShowCalibration((v) => !v);
                      }}
                      title={showCalibration ? "Close the court alignment" : calibration ? "Check or adjust the court alignment" : "The court is needed to find rallies and points"}
                    >
                      {showCalibration ? <IconClose size={14} /> : calibration ? <IconCheck size={14} /> : <IconTarget size={14} />}
                      {showCalibration ? "Close alignment" : calibration ? "Court aligned" : "Align court"}
                    </button>
                  )}
                </div>
              </div>

              {viewMode === "side-by-side" && (
                <div className="split-view" style={{ "--split": `${splitPct}%` } as React.CSSProperties}>
                  <div
                    className="split-pane"
                    ref={setSplitVideoEl}
                    style={{
                      overflow: "hidden",
                      cursor: correctionMode ? "crosshair" : (!showCalibration && videoZoomTransform.zoom > 1 ? "grab" : undefined),
                    }}
                    onMouseMove={handleVideoDragMove}
                    onMouseUp={handleVideoDragEnd}
                    onMouseLeave={handleVideoDragEnd}
                    onClick={handleVideoPaneClick}
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
                        onPlaybackStopped={stopPlayback}
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
                      {/* ── Manual ball correction overlay ──────────── */}
                      {correctionMode && videoElRef.current && videoZoomEl && (
                        <BallCorrectionOverlay
                          videoEl={videoElRef.current}
                          containerEl={videoZoomEl}
                          corrections={manualBallCorrections}
                          currentFrame={currentFrame}
                        />
                      )}
                      {showCalibration && splitVideoSize.width > 0 && splitVideoSize.height > 0 &&
                        renderCalibration(splitVideoSize.width, splitVideoSize.height)}
                    </div>
                    {!showCalibration && <span className="pane-label">Video</span>}
                  </div>
                  {/* Calibration maps clicks through the pane's size, so it stays put meanwhile. */}
                  {!showCalibration && (
                    <div
                      className="split-handle"
                      role="separator"
                      aria-orientation="vertical"
                      title="Drag to resize · double-click to reset"
                      onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
                      onPointerMove={(e) => {
                        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
                        const r = e.currentTarget.parentElement!.getBoundingClientRect();
                        setSplitPct(Math.max(MIN_SPLIT_PCT, Math.min(100 - MIN_SPLIT_PCT, (100 * (e.clientX - r.left)) / r.width)));
                      }}
                      onDoubleClick={() => setSplitPct(DEFAULT_SPLIT_PCT)}
                    />
                  )}
                  <div className="split-pane">
                    <SkeletonViewer
                      calibration={calibration}
                      frames={solvedFrames}
                      currentFrame={currentFrame}
                      videoAspect={videoAspect}
                      courtData={courtData}
                      isRealVideo={isRealVideo}
                      resetViewTrigger={resetViewTrigger}
                      trajectoryMode={trajectoryMode}
                      touches={displayTouches}
                    />
                    <span className="pane-label">3D reconstruction</span>
                    <button
                      className="reset-view-btn"
                      title="Back to the video's point of view"
                      onClick={() => setResetViewTrigger((v) => v + 1)}
                    >
                      <IconRefresh size={13} />Reset view
                    </button>
                  </div>
                </div>
              )}

              {viewMode === "inspect" && (
                <RallyInspector
                  frames={framesForSolver}
                  fps={result.fps}
                  videoSize={videoSize}
                  calibration={calibration}
                  positions={shownBall3dMap}
                  fittedPositions={fittedBall3dMap}
                  segments={refit.segments}
                  onUndo={() => stepEdits(-1)}
                  onRedo={() => stepEdits(1)}
                  canUndo={editHistory.past.length > 0}
                  canRedo={editHistory.future.length > 0}
                  physicsDebug={physicsDebug}
                  trace={solverTrace}
                  touchSplits={touchSplits}
                  touches={displayTouches}
                  // A hand-aligned court is checked by eye; its error against
                  // the detected keypoints says nothing.
                  courtErrPx={manualCalibrationLocked ? null : calibrationLog?.maxKpErr ?? null}
                  edits={edits}
                  onEdit={updateEdits}
                  currentFrame={currentFrame}
                  onSeek={seekToFrame}
                  solving={solving}
                  saveState={saveState}
                  saveError={saveError}
                  onAlignCourt={() => setShowCalibration(true)}
                  align={showCalibration ? { render: renderCalibration, view: calibrationViewTransform } : null}
                  video={
                    <VideoPlayer
                      videoPath={videoPath}
                      currentFrame={currentFrame}
                      fps={result.fps}
                      isPlaying={isPlaying}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onMetadataLoaded={handleMetadataLoaded}
                      onVideoElement={handleVideoElement}
                      onCanPlay={handleCanPlay}
                      onPlaybackStopped={stopPlayback}
                    />
                  }
                />
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
                      onPlaybackStopped={stopPlayback}
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
                      trajectoryMode={trajectoryMode}
                      touches={displayTouches}
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
                fps={isRealVideo ? result.fps : undefined}
                timeline={timeline}
                kinds={matchKinds}
                ralliesOnly={ralliesOnly}
                onRalliesOnlyChange={setRalliesOnly}
                onFrameChange={seekToFrame}
                onTogglePlay={togglePlay}
              />
            </div>
            {viewMode !== "inspect" && <div className="sidebar">
              <Scoreboard timeline={timeline} teams={edits.teams} onRenameTeam={renameTeam} />
              <div className="sidebar-tabs" role="tablist">
                <button
                  role="tab"
                  aria-selected={sidebarTab === "match"}
                  className={`sidebar-tab ${sidebarTab === "match" ? "active" : ""}`}
                  onClick={() => setSidebarTab("match")}
                >Rallies</button>
                <button
                  role="tab"
                  aria-selected={sidebarTab === "stats"}
                  className={`sidebar-tab ${sidebarTab === "stats" ? "active" : ""}`}
                  onClick={() => setSidebarTab("stats")}
                >Stats</button>
                <button
                  role="tab"
                  aria-selected={sidebarTab === "court"}
                  className={`sidebar-tab ${sidebarTab === "court" ? "active" : ""}`}
                  onClick={() => setSidebarTab("court")}
                >Court</button>
                <button
                  role="tab"
                  aria-selected={diagnosticsTab}
                  className={`sidebar-tab ${diagnosticsTab ? "active" : ""}`}
                  onClick={() => setSidebarTab("ball-debug")}
                >Diagnostics</button>
              </div>

              <div className="sidebar-body">
              {diagnosticsTab && (
                <div className="segmented segmented-full" role="group" aria-label="Diagnostic">
                  <button className={`view-btn ${sidebarTab === "ball-debug" ? "active" : ""}`} onClick={() => setSidebarTab("ball-debug")}>Ball tracking</button>
                  <button className={`view-btn ${sidebarTab === "depth" ? "active" : ""}`} onClick={() => setSidebarTab("depth")}>Depth filter</button>
                </div>
              )}
              {sidebarTab === "match" && (
                <MatchPanel
                  timeline={timeline}
                  fps={result.fps}
                  currentFrame={currentFrame}
                  teams={edits.teams}
                  kinds={matchKinds}
                  onKindsChange={setMatchKinds}
                  onSeek={seekToFrame}
                  onSetWinner={setPointWinner}
                  pending={
                    !isRealVideo ? "Demo data has no rallies."
                      : !calibration ? "Align the court to find rallies and points."
                      : solving ? "Solving the ball path…"
                      : null
                  }
                />
              )}
              {sidebarTab === "stats" && (
                <StatsDisplay result={result} timeline={timeline} teams={edits.teams} analysisDurationMs={analysisDurationMs} />
              )}
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
                    {/* One reconstruction path: fit a ballistic arc whose 2D
                        reprojection matches the tracked ball. The bbox-depth
                        modes are kept in the solver for diagnostics but are not
                        offered here — apparent ball size is too noisy at this
                        range to place the ball in depth, so exposing them only
                        invited picking a mode that cannot work. */}
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
                    {/* ── Manual 2D correction mode ─────────────────── */}
                    <div className="control-row">
                      <button
                        className={`btn btn-sm ${correctionMode ? "btn-danger" : "btn-secondary"}`}
                        onClick={() => setCorrectionMode((v) => !v)}
                      >
                        {correctionMode ? "Exit correction mode" : "Manual ball correction"}
                      </button>
                      {correctionMode && (
                        <span className="match-dim">
                          Click the video to set the ball centre for frame {currentFrame}
                        </span>
                      )}
                    </div>
                    {manualBallCorrections.size > 0 && (
                      <div className="control-row">
                        <span className="match-dim">
                          {manualBallCorrections.size} correction(s)
                        </span>
                        <button
                          className="btn btn-sm btn-secondary"
                          onClick={() => updateEdits((e) => {
                            const ball = { ...e.ball };
                            delete ball[currentFrame];
                            return { ...e, ball };
                          })}
                        >Clear this frame</button>
                        <button
                          className="btn btn-sm btn-secondary"
                          onClick={() => updateEdits((e) => ({ ...e, ball: {} }))}
                        >Clear all</button>
                      </div>
                    )}
                  </div>
                  {/* ── Calibration log ──────────────────────────────── */}
                  {calibrationLog && (
                    <div className="calibration-log">
                      <div className="calibration-log-title">Calibration</div>
                      <div className="calibration-log-row">
                        <span>FoV Y</span>
                        <span>{calibrationLog.fovY.toFixed(2)}°</span>
                      </div>
                      <div className="calibration-log-row">
                        <span>Camera pos</span>
                        <span>[{calibrationLog.position.map((v) => v.toFixed(2)).join(", ")}]</span>
                      </div>
                      <div className="calibration-log-row">
                        <span>LookAt</span>
                        <span>[{calibrationLog.lookAt.map((v) => v.toFixed(2)).join(", ")}]</span>
                      </div>
                      {calibrationLog.avgKpErr != null && (
                        <div className="calibration-log-row">
                          <span>Court kp err</span>
                          <span>avg {calibrationLog.avgKpErr.toFixed(1)} px / max {calibrationLog.maxKpErr?.toFixed(1)} px ({calibrationLog.courtKpErrors.length} pts)</span>
                        </div>
                      )}
                    </div>
                  )}
                  {/* A planar court calibrates to a few px when the detected
                      corners are right. Tens of px means the court itself is
                      wrong, and every 3D position downstream inherits that. */}
                  {calibrationLog?.avgKpErr != null && calibrationLog.avgKpErr > 25 && (
                    <div className="physics-failed-warning">
                      <IconAlert />
                      <span>
                        Court calibration is unreliable ({calibrationLog.avgKpErr.toFixed(0)} px error).
                        The auto-detected court does not match a real camera, so 3D ball
                        positions cannot be trusted. Fix the court corners with{" "}
                        <strong>Align court</strong> and re-check.
                      </span>
                    </div>
                  )}
                  {physicsFailedWarning && (
                    <div className="physics-failed-warning">
                      <IconAlert />
                      <span>
                        Physics reprojection failed — no arc could be fitted from the detected ball positions.
                        The 3D trajectory is empty. Check confidence threshold, camera calibration, and ensure
                        at least 4 consecutive ball detections exist.
                      </span>
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
                  {(ball3dMap.size > 0 || (physicsDebug && physicsDebug.length > 0)) && (
                    <div className="trajectory-export">
                      {/* One export. Previously five overlapping files that had
                          to be cross-referenced by hand; this is a single
                          document containing calibration, the solved
                          trajectory, per-frame records and every arc. */}
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => {
                          const report = {
                            generatedAt: new Date().toISOString(),
                            video: { path: videoPath, width: videoSize.width, height: videoSize.height },
                            calibration: calibrationLog
                              ? {
                                  fovY: calibrationLog.fovY,
                                  position: calibrationLog.position,
                                  lookAt: calibrationLog.lookAt,
                                  courtKeypointErrorPx: {
                                    avg: calibrationLog.avgKpErr,
                                    max: calibrationLog.maxKpErr,
                                    perPoint: calibrationLog.courtKpErrors,
                                  },
                                  trustworthy: (calibrationLog.avgKpErr ?? 0) <= 25,
                                }
                              : null,
                            detection: {
                              framesAnalysed: frames.length,
                              detected: frames.filter((f) => f.ball_img_pos && !f.ball_interpolated).length,
                              interpolated: frames.filter((f) => f.ball_interpolated).length,
                              solved3d: ball3dMap.size,
                            },
                            trajectory: solvedFrames.map((f) => ({
                              frame: f.frame,
                              timestamp: f.timestamp,
                              imagePos: f.ball_img_pos,
                              interpolated: f.ball_interpolated ?? false,
                              conf: f.ball_conf,
                              pos3d: f.ball_3d_pos,
                              method: f.ball_3d_method,
                            })),
                            perFrameDebug: ball3dDebug,
                            arcs: physicsDebug ?? [],
                          };
                          const json = JSON.stringify(report, null, 2);
                          void getBackend().then((b) => b.saveTextFile("ball_debug_report.json", json));
                        }}
                      >
                        Export debug report ({ball3dMap.size} solved / {ball3dDebug.length} records)
                      </button>
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
                  frameMapping={frameMapping}
                />
              )}
              </div>
            </div>}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
