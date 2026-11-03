import { useState, useRef, useCallback, useEffect } from "react";
import * as THREE from "three";
import { getBackend } from "../platform/backend";
import { type CalibrationResult, solveCameraIterative } from "../utils/cameraSolver";
import { computeHomography, applyHomography } from "../utils/courtHomography";

/** Top of the net tape (m), men's height. */
const NET_TAPE_Y_M = 2.43;
/** The antennas rise 0.8 m above the net tape. */
const ANTENNA_TOP_Y_M = NET_TAPE_Y_M + 0.8;

/**
 * Vital intersections users place manually.
 * Court centered at origin (x: left/right, z: far/near), y=0 is floor.
 */
const CALIBRATION_POINTS_3D = [
  { x: -4.5, y: 0, z: -9.0, label: "Far Left Corner", color: "#ff4444" },
  { x:  4.5, y: 0, z: -9.0, label: "Far Right Corner", color: "#44ff44" },
  { x: -4.5, y: 0, z: -3.0, label: "Far 10ft Left", color: "#44ccff" },
  { x:  4.5, y: 0, z: -3.0, label: "Far 10ft Right", color: "#00ffaa" },
  { x: -4.5, y: 0, z:  0.0, label: "Net Left", color: "#ffaa00" },
  { x:  4.5, y: 0, z:  0.0, label: "Net Right", color: "#ffcc33" },
  { x: -4.5, y: 0, z:  3.0, label: "Near 10ft Left", color: "#77aaff" },
  { x:  4.5, y: 0, z:  3.0, label: "Near 10ft Right", color: "#5588ff" },
  { x: -4.5, y: 0, z:  9.0, label: "Near Left Corner", color: "#ffff44" },
  { x:  4.5, y: 0, z:  9.0, label: "Near Right Corner", color: "#4444ff" },
  // Above the floor: the antennas stand on the sidelines, from the net tape
  // (2.43 m, men) up 0.8 m. Known heights fix the camera's focal length and
  // height scale, which floor points alone leave weak (worst from the side).
  { x: -4.5, y: ANTENNA_TOP_Y_M, z: 0, label: "Left Antenna · Top", color: "#ff66cc" },
  { x: -4.5, y: NET_TAPE_Y_M, z: 0, label: "Left Antenna · Net Tape", color: "#cc66ff" },
  { x:  4.5, y: ANTENNA_TOP_Y_M, z: 0, label: "Right Antenna · Top", color: "#ff99dd" },
  { x:  4.5, y: NET_TAPE_Y_M, z: 0, label: "Right Antenna · Net Tape", color: "#dd99ff" },
];

/**
 * The same ten court points named as seen from the side. The world frame must
 * not be mirrored relative to the picture, so a side camera is taken to stand
 * at -x: the near sideline is x = -4.5 and the picture's left end is z = -9.
 */
const SIDE_LABELS = [
  "Near sideline · left end", "Far sideline · left end",
  "Near sideline · left 3 m line", "Far sideline · left 3 m line",
  "Near sideline · centre line", "Far sideline · centre line",
  "Near sideline · right 3 m line", "Far sideline · right 3 m line",
  "Near sideline · right end", "Far sideline · right end",
  "Near antenna · top", "Near antenna · net tape",
  "Far antenna · top", "Far antenna · net tape",
];
type CourtView = "behind" | "side";
type Pt = { x: number; y: number };
/** Click order per view (indices into CALIBRATION_POINTS_3D). */
const CLICK_ORDER: Record<CourtView, number[]> = {
  behind: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
  // Along the far sideline left to right, then the near one; then the
  // antennas, far one first (the near one is often out of the picture).
  side: [1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 12, 13, 10, 11],
};
/** At least this many points must be placed (a homography needs 4). */
const MIN_POINTS = 4;

const CALIBRATION_GUIDE_LINES: [number, number][] = [
  [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
  [0, 2], [2, 4], [4, 6], [6, 8],
  [1, 3], [3, 5], [5, 7], [7, 9],
  // Antennas, and the net tape between them.
  [10, 11], [12, 13], [11, 13],
];

const AUTO_CORNER_WORLD: [number, number][] = [
  [-4.5, -9],
  [4.5, -9],
  [4.5, 9],
  [-4.5, 9],
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;

interface DebugStep {
  label: string;
  info: string;
  image: string; // base64 data URL
}

interface DebugInfo {
  steps: DebugStep[];
  log: string[];
  corners: { x: number; y: number }[] | null;
}

function expandCornersToCalibrationPoints(corners: { x: number; y: number }[]) {
  if (corners.length !== 4) return corners;
  const src = AUTO_CORNER_WORLD;
  const dst = corners.map((c) => [c.x, c.y] as [number, number]);
  const H = computeHomography(src, dst);
  if (!H) return corners;
  // Floor points only: the homography maps the floor plane, so it cannot
  // place the antennas. Those are left for the user to click.
  return CALIBRATION_POINTS_3D.filter((p) => p.y === 0).map((p) => {
    const [x, y] = applyHomography(H, p.x, p.z);
    return { x, y };
  });
}

interface CourtCalibrationProps {
  width: number;
  height: number;
  videoPath: string;
  adjustMode?: boolean;
  initialCalibration?: CalibrationResult;
  onViewTransformChange?: (transform: { zoom: number; panX: number; panY: number }) => void;
  onCalibrated: (result: CalibrationResult) => void;
  onCancel: () => void;
  /**
   * Moving camera: map a point on the frame being shown (video-normalised
   * [0,1]) into the reference frame the calibration must belong to, and back.
   * Omitted for a static camera.
   */
  toReference?: (p: Pt) => Pt;
  fromReference?: (p: Pt) => Pt;
}

export default function CourtCalibration({
  width,
  height,
  videoPath,
  adjustMode = false,
  initialCalibration,
  onViewTransformChange,
  onCalibrated,
  onCancel,
  toReference,
  fromReference,
}: CourtCalibrationProps) {
  // Indexed like CALIBRATION_POINTS_3D: a point, null = skipped (not
  // visible), undefined = not reached yet. `step` walks CLICK_ORDER[view].
  const [points, setPoints] = useState<(Pt | null | undefined)[]>([]);
  const [view, setView] = useState<CourtView>("behind");
  // Starts by adjusting the current calibration when there is one; Reset or
  // a change of view leaves it to place the points afresh.
  const [adjusting, setAdjusting] = useState(adjustMode);
  const [step, setStep] = useState(0);
  const order = CLICK_ORDER[view];
  const N = CALIBRATION_POINTS_3D.length;
  const activePoint = step < N ? order[step] : -1;
  const placedCount = points.filter(Boolean).length;
  const labelOf = (i: number) => (view === "side" ? SIDE_LABELS[i] : CALIBRATION_POINTS_3D[i].label);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [autoStatus, setAutoStatus] = useState<string>("");
  const [zoomedStep, setZoomedStep] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ mouseX: number; mouseY: number; panX: number; panY: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debug state: filled after each Apply to show step-by-step pipeline
  const [pipeline, setPipeline] = useState<{
    // Step 1 – what the user placed (container-normalized [0,1])
    containerPts: Array<Pt | null>;
    // Step 2 – letterbox geometry
    lb: { cw: number; ch: number; ox: number; oy: number; rw: number; rh: number; videoAR: number };
    // Step 3 – video-frame-normalized [0,1]
    videoFramePts: Array<Pt | null>;
    // Step 4 – NDC [-1,1] sent to solver
    ndcPts: Array<Pt | null>;
    // Step 5 – solver result
    fov: number;
    pos: [number, number, number];
    lookAt: [number, number, number];
    solverLog: string[];
    // Step 6 – reprojection (container-normalized [0,1])
    reprojected: Array<Pt | null>;
    errPx: Array<number | null>;
    avgErrPx: number;
    quality: string;
    // pending: accepted by user to actually apply
    pending: CalibrationResult;
  } | null>(null);
  const [activeStep, setActiveStep] = useState(1);

  useEffect(() => {
    onViewTransformChange?.({ zoom, panX: pan.x, panY: pan.y });
  }, [zoom, pan.x, pan.y, onViewTransformChange]);

  const getVideoAspect = useCallback(() => {
    const container = containerRef.current;
    if (!container) return width / height;
    const video = container.parentElement?.querySelector("video") as HTMLVideoElement | null;
    if (video && video.videoWidth > 0 && video.videoHeight > 0) {
      return video.videoWidth / video.videoHeight;
    }
    return width / height;
  }, [width, height]);

  const clientToNormalized = useCallback((clientX: number, clientY: number) => {
    const container = containerRef.current;
    if (!container) return null;
    const rect = container.getBoundingClientRect();
    const lx = clientX - rect.left;
    const ly = clientY - rect.top;
    const stageX = (lx - pan.x) / zoom;
    const stageY = (ly - pan.y) / zoom;
    return {
      x: Math.max(0, Math.min(1, stageX / rect.width)),
      y: Math.max(0, Math.min(1, stageY / rect.height)),
    };
  }, [pan.x, pan.y, zoom]);

  // In adjust mode, project current calibration corners to 2D on mount
  useEffect(() => {
    if (!adjusting || !initialCalibration) return;
    const aspect = getVideoAspect();
    const cam = new THREE.PerspectiveCamera(initialCalibration.fov, aspect, 0.1, 500);
    cam.position.set(...initialCalibration.position);
    cam.lookAt(new THREE.Vector3(...initialCalibration.lookAt));
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();
    // A camera along a sideline is a side view: name the points that way.
    const [px, , pz] = initialCalibration.position;
    setView(Math.abs(px) > Math.abs(pz) ? "side" : "behind");

    // Compute letterbox rect so we can map video-frame NDC → container [0,1]
    const cw2 = containerRef.current?.clientWidth || 0;
    const ch2 = containerRef.current?.clientHeight || 0;
    let ox2 = 0, oy2 = 0, rw2 = cw2 || 1, rh2 = ch2 || 1;
    if (cw2 > 0 && ch2 > 0) {
      const car = cw2 / ch2;
      if (aspect > car) {
        rw2 = cw2; rh2 = cw2 / aspect; ox2 = 0; oy2 = (ch2 - rh2) / 2;
      } else {
        rh2 = ch2; rw2 = ch2 * aspect; ox2 = (cw2 - rw2) / 2; oy2 = 0;
      }
    }

    const projected: (Pt | null)[] = [];
    for (const c of CALIBRATION_POINTS_3D) {
      const v = new THREE.Vector3(c.x, c.y, c.z).project(cam);
      // Outside the picture (or behind the camera): not placeable, leave it
      // out. Clamped to the edge it would be solved as if clicked there.
      if (Math.abs(v.x) > 1 || Math.abs(v.y) > 1 || v.z > 1) { projected.push(null); continue; }
      // Video-frame NDC [-1,1] → video-frame [0,1]
      let vx = (v.x + 1) / 2;
      let vy = (-v.y + 1) / 2;
      // The calibration belongs to the reference frame; show it on this one.
      if (fromReference) ({ x: vx, y: vy } = fromReference({ x: vx, y: vy }));
      // Video-frame [0,1] → container [0,1] (letterbox expand)
      const cx = cw2 > 0 ? (ox2 + vx * rw2) / cw2 : vx;
      const cy = ch2 > 0 ? (oy2 + vy * rh2) / ch2 : vy;
      projected.push({ x: cx, y: cy });
    }
    setPoints(projected);
    setStep(CALIBRATION_POINTS_3D.length);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [adjusting, initialCalibration, getVideoAspect, fromReference]);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (adjusting) return;
      if (dragIndex !== null || isPanning) return;
      const pos = clientToNormalized(e.clientX, e.clientY);
      if (!pos) return;

      if (activePoint >= 0) {
        setPoints((prev) => { const next = [...prev]; next[activePoint] = pos; return next; });
        setStep((prev) => prev + 1);
      }
    },
    [adjusting, activePoint, dragIndex, isPanning, clientToNormalized]
  );

  /** Switch behind/side naming; placements so far no longer mean anything. */
  const handleView = useCallback((v: CourtView) => {
    setAdjusting(false);
    setView(v);
    setPoints([]);
    setStep(0);
    setPipeline(null);
  }, []);

  /** The highlighted point is not visible in this frame: leave it out. */
  const handleSkip = useCallback(() => {
    if (activePoint < 0) return;
    setPoints((prev) => { const next = [...prev]; next[activePoint] = null; return next; });
    setStep((prev) => prev + 1);
  }, [activePoint]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent, index: number) => {
      e.stopPropagation();
      if (e.button !== 0) return;
      setDragIndex(index);
    },
    []
  );

  const handlePanStart = useCallback((e: React.MouseEvent) => {
    if (e.button !== 1 && e.button !== 2 && !e.altKey) return;
    e.preventDefault();
    panStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
    setIsPanning(true);
  }, [pan.x, pan.y]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;

    setZoom((prev) => {
      const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev * factor));
      if (next === prev) return prev;
      setPan((curr) => {
        const worldX = (px - curr.x) / prev;
        const worldY = (py - curr.y) / prev;
        return {
          x: px - worldX * next,
          y: py - worldY * next,
        };
      });
      return next;
    });
  }, []);

  useEffect(() => {
    if (dragIndex === null && !isPanning) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (dragIndex !== null) {
        const pos = clientToNormalized(e.clientX, e.clientY);
        if (!pos) return;
        setPoints((prev) => {
          const next = [...prev];
          next[dragIndex] = pos;
          return next;
        });
      } else if (isPanning && panStartRef.current) {
        const s = panStartRef.current;
        setPan({
          x: s.panX + (e.clientX - s.mouseX),
          y: s.panY + (e.clientY - s.mouseY),
        });
      }
    };

    const handleMouseUp = () => {
      setDragIndex(null);
      setIsPanning(false);
      panStartRef.current = null;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragIndex, isPanning, clientToNormalized]);

  const handleApply = useCallback(() => {
    if (placedCount < MIN_POINTS) return;

    // ── Step 1 ─ collect container-normalized [0,1] positions ────────────────
    const containerPts = CALIBRATION_POINTS_3D.map((_, i) => {
      const p = points[i];
      return p ? { x: p.x, y: p.y } : null;
    });

    // ── Step 2 ─ compute letterbox geometry ───────────────────────────────────
    const cw = containerRef.current?.clientWidth || 1;
    const ch = containerRef.current?.clientHeight || 1;
    const videoAR = getVideoAspect();
    const containerAR = cw / ch;
    let ox = 0, oy = 0, rw = cw, rh = ch;
    if (videoAR > containerAR) {
      rw = cw; rh = cw / videoAR; ox = 0; oy = (ch - rh) / 2;
    } else {
      rh = ch; rw = ch * videoAR; ox = (cw - rw) / 2; oy = 0;
    }
    const lb = { cw, ch, ox, oy, rw, rh, videoAR };

    // ── Step 3 ─ video-frame-normalized [0,1] ─────────────────────────────────
    const videoFramePts = containerPts.map(p => (p ? {
      x: (p.x * cw - ox) / rw,
      y: (p.y * ch - oy) / rh,
    } : null));

    // ── Step 4 ─ NDC [-1,1] for solver ────────────────────────────────────────
    // A moving camera's clicks are moved into the reference frame first: the
    // calibration (and the ball data) belong to that frame.
    const ndcPts = videoFramePts.map(p => {
      if (!p) return null;
      const r = toReference ? toReference(p) : p;
      return { x: r.x * 2 - 1, y: -(r.y * 2 - 1) };
    });

    // ── Step 5 ─ solver (placed points only) ──────────────────────────────────
    const used = CALIBRATION_POINTS_3D.map((_, i) => i).filter((i) => ndcPts[i]);
    const worldPoints = used.map((i) => ({ x: CALIBRATION_POINTS_3D[i].x, y: CALIBRATION_POINTS_3D[i].y, z: CALIBRATION_POINTS_3D[i].z }));
    const { result, log } = solveCameraIterative(used.map((i) => ndcPts[i]!), videoAR, worldPoints);

    if (!result) {
      setAutoStatus("ERROR: solver failed — check coordinates");
      setPipeline(null);
      return;
    }

    // ── Step 6 ─ reprojection ─────────────────────────────────────────────────
    const rCam = new THREE.PerspectiveCamera(result.fov, videoAR, 0.1, 500);
    rCam.position.set(...result.position);
    rCam.lookAt(new THREE.Vector3(...result.lookAt));
    rCam.updateMatrixWorld();
    rCam.updateProjectionMatrix();

    const reprojected: Array<Pt | null> = [];
    const errPxArr: Array<number | null> = [];
    for (let i = 0; i < CALIBRATION_POINTS_3D.length; i++) {
      const wp = CALIBRATION_POINTS_3D[i];
      const v = new THREE.Vector3(wp.x, wp.y, wp.z).project(rCam);
      // reference NDC → reference [0,1] → this frame [0,1] → container [0,1]
      let vf = { x: (v.x + 1) / 2, y: (1 - v.y) / 2 };
      if (fromReference) vf = fromReference(vf);
      reprojected.push({ x: (ox + vf.x * rw) / cw, y: (oy + vf.y * rh) / ch });
      // pixel error using video height (1080p reference), placed points only
      const placed = videoFramePts[i];
      errPxArr.push(placed ? Math.hypot(vf.x - placed.x, vf.y - placed.y) * 1080 : null);
    }
    const errs = errPxArr.filter((e): e is number => e != null);
    const avgErrPx = errs.reduce((a, b) => a + b, 0) / Math.max(1, errs.length);
    const quality = avgErrPx < 5 ? "excellent" : avgErrPx < 15 ? "good" : avgErrPx < 35 ? "fair" : "poor";

    setPipeline({
      containerPts,
      lb,
      videoFramePts,
      ndcPts,
      fov: result.fov,
      pos: result.position,
      lookAt: result.lookAt,
      solverLog: log,
      reprojected,
      errPx: errPxArr,
      avgErrPx,
      quality,
      pending: result,
    });
    setActiveStep(6); // jump straight to reprojection view
    setAutoStatus(`Solved from ${used.length} points — avg reprojection error: ${avgErrPx.toFixed(1)} px (${quality})`);
  }, [points, placedCount, getVideoAspect, toReference, fromReference]);

  const handleAccept = useCallback(() => {
    if (!pipeline) return;
    onCalibrated(pipeline.pending);
  }, [pipeline, onCalibrated]);

  const handleReset = useCallback(() => {
    setAdjusting(false);
    setPoints([]);
    setStep(0);
    setDebugInfo(null);
    setPipeline(null);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleAutoDetect = useCallback(async () => {
    if (!videoPath) {
      setAutoStatus("ERROR: No video path");
      return;
    }

    setAutoStatus("Running Python court detection...");

    try {
      // Get current timestamp from video element
      const video = containerRef.current?.parentElement?.querySelector("video") as HTMLVideoElement | null;
      const frameTime = video?.currentTime ?? 0;

      const result = await (await getBackend()).detectCourt(videoPath, frameTime);

      const steps = (result.steps ?? []) as DebugStep[];
      const log = result.log ?? [];

      if (result.action === "court_detected" && result.corners) {
        const corners = result.corners.map(([x, y]: number[]) => ({ x, y }));

        // Convert from video-frame-normalized [0,1] to container-normalized [0,1]
        // to account for object-fit:contain letterboxing
        if (video && video.videoWidth > 0 && containerRef.current) {
          const vw = video.videoWidth;
          const vh = video.videoHeight;
          const cw = containerRef.current.clientWidth;
          const ch = containerRef.current.clientHeight;
          const videoAR = vw / vh;
          const containerAR = cw / ch;

          let rw: number, rh: number, ox: number, oy: number;
          if (videoAR > containerAR) {
            rw = cw; rh = cw / videoAR;
            ox = 0; oy = (ch - rh) / 2;
          } else {
            rh = ch; rw = ch * videoAR;
            ox = (cw - rw) / 2; oy = 0;
          }

          for (const c of corners) {
            c.x = (ox + c.x * rw) / cw;
            c.y = (oy + c.y * rh) / ch;
          }
        }

        // Auto detection reports corners in behind-the-baseline order.
        setView("behind");
        const expanded = expandCornersToCalibrationPoints(corners);
        setPoints(expanded.map((p) => ({
          x: Math.max(0, Math.min(1, p.x)),
          y: Math.max(0, Math.min(1, p.y)),
        })));
        setStep(expanded.length);
        setAutoStatus(`Detected ${corners.length} corners`);
        setDebugInfo({ steps, log, corners });
      } else {
        setAutoStatus(`Detection failed: ${result.message || "Unknown error"}`);
        setDebugInfo({ steps, log, corners: null });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[AutoDetect] Error:", err);
      setAutoStatus(`ERROR: ${msg}`);
    }
  }, [videoPath]);

  return (
    <div
      ref={containerRef}
      className="calibration-overlay"
      onClick={handleClick}
      onMouseDown={handlePanStart}
      onWheel={handleWheel}
      onContextMenu={(e) => e.preventDefault()}
      style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0, cursor: adjusting ? "default" : activePoint >= 0 ? "crosshair" : "default", zIndex: 20 }}
    >
      <div className="calibration-instructions">
        {adjusting ? (
          <span>Drag points to adjust intersections. Mouse wheel zooms, middle/right drag pans. Way off? Pick Behind/Side or Re-place points to click them afresh.</span>
        ) : activePoint >= 0 ? (
          <span>
            Click point <strong>{step + 1}/{N}</strong>:{" "}
            <span style={{ color: CALIBRATION_POINTS_3D[activePoint].color }}>
              {labelOf(activePoint)}
            </span>
            {" "}<span style={{ opacity: 0.75 }}>(Skip it if it is not in the picture; {MIN_POINTS}+ needed, {placedCount} placed)</span>
          </span>
        ) : placedCount >= MIN_POINTS ? (
          <span>Drag points, then Solve ({placedCount} placed)</span>
        ) : (
          <span>Only {placedCount} placed: at least {MIN_POINTS} are needed. Reset and try another frame.</span>
        )}
      </div>

      <div
        style={{
          position: "absolute",
          inset: 0,
          transformOrigin: "0 0",
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
        }}
      >

        {points.map((p, i) => p && (
          <div
            key={i}
            onMouseDown={(e) => handleMouseDown(e, i)}
            className="calibration-point"
            style={{
              left: `${p.x * 100}%`,
              top: `${p.y * 100}%`,
              backgroundColor: CALIBRATION_POINTS_3D[i].color,
              cursor: "grab",
            }}
          >
            <span className="calibration-point-label">{labelOf(i)}</span>
          </div>
        ))}

        {(placedCount >= 2 || pipeline !== null) && (
          <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
            {/* Guide lines between placed points */}
            {CALIBRATION_GUIDE_LINES.map(([a, b], i) => {
              const pa = points[a], pb = points[b];
              if (!pa || !pb) return null;
              return (
                <line
                  key={i}
                  x1={`${pa.x * 100}%`} y1={`${pa.y * 100}%`}
                  x2={`${pb.x * 100}%`} y2={`${pb.y * 100}%`}
                  stroke="rgba(255,255,255,0.6)" strokeWidth="2" strokeDasharray="6,4"
                />
              );
            })}

            {/* Reprojection overlay — shown when on step 6 */}
            {pipeline && activeStep === 6 && points.map((p, i) => {
              const rp = pipeline.reprojected[i];
              const err = pipeline.errPx[i];
              if (!p || !rp || err == null) return null;
              const col = err < 5 ? "#00ff88" : err < 15 ? "#ffcc00" : err < 35 ? "#ff8800" : "#ff3333";
              const mx = ((p.x + rp.x) / 2) * 100;
              const my = ((p.y + rp.y) / 2) * 100;
              return (
                <g key={`rp-${i}`}>
                  <line x1={`${p.x * 100}%`} y1={`${p.y * 100}%`}
                        x2={`${rp.x * 100}%`} y2={`${rp.y * 100}%`}
                        stroke={col} strokeWidth="1.5" opacity="0.9" />
                  <circle cx={`${rp.x * 100}%`} cy={`${rp.y * 100}%`}
                          r="10" fill="none" stroke={col} strokeWidth="2" strokeDasharray="4,3" />
                  <text x={`${mx}%`} y={`${my}%`} fill={col} fontSize="10"
                        textAnchor="middle" dy="-3" style={{ fontFamily: "monospace" }}>
                    {err.toFixed(0)}px
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>

      <div className="calibration-buttons" onClick={(e) => e.stopPropagation()}>
        {(
          <span style={{ display: "inline-flex", gap: 4 }}>
            <button className={`btn ${view === "behind" ? "btn-primary" : "btn-secondary"}`} onClick={() => handleView("behind")}
              title="Camera behind a baseline">Behind</button>
            <button className={`btn ${view === "side" ? "btn-primary" : "btn-secondary"}`} onClick={() => handleView("side")}
              title="Camera along a sideline">Side</button>
          </span>
        )}
        {!adjusting && activePoint >= 0 && (
          <button className="btn btn-secondary" onClick={handleSkip} title="This point is not in the picture">Skip point</button>
        )}
        {!adjusting && <button className="btn btn-primary" onClick={handleAutoDetect}>Auto Detect</button>}
        <button className="btn btn-secondary" onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.2))}>−</button>
        <button className="btn btn-secondary" onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.2))}>+</button>
        <button className="btn btn-secondary" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Fit</button>
        {(adjusting || activePoint < 0) && placedCount >= MIN_POINTS && !pipeline && (
          <button className="btn btn-primary" onClick={handleApply}>Solve →</button>
        )}
        {pipeline && (
          <button className="btn btn-primary" onClick={handleAccept}
            style={{ background: pipeline.quality === "poor" ? "#ff4444" : "#00cc66" }}>
            ✓ Accept
          </button>
        )}
        {pipeline && (
          <button className="btn btn-secondary" onClick={() => setPipeline(null)}>Adjust Points</button>
        )}
        {adjusting && !pipeline && (
          <button className="btn btn-secondary" onClick={handleReset} title="Discard these points and click them afresh">Re-place points</button>
        )}
        {step > 0 && !adjusting && !pipeline && (
          <button className="btn btn-secondary" onClick={handleReset}>Reset</button>
        )}
        <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        {autoStatus && (
          <span style={{
            color: autoStatus.startsWith("ERROR") ? "#f55"
              : pipeline?.quality === "excellent" ? "#00ff88"
              : pipeline?.quality === "good" ? "#88dd44"
              : pipeline?.quality === "fair" ? "#ffcc00"
              : "#f88",
            fontSize: 12, alignSelf: "center", marginLeft: 8,
          }}>
            {autoStatus}
          </span>
        )}
      </div>

      {/* ── Step-by-step pipeline debug panel ── */}
      {pipeline && (
        <div className="calibration-pipeline-panel" onClick={(e) => e.stopPropagation()}>
          {/* Tab bar */}
          <div className="pipeline-tabs">
            {[
              { n: 1, label: "1 Placed" },
              { n: 2, label: "2 Letterbox" },
              { n: 3, label: "3 Video px" },
              { n: 4, label: "4 NDC" },
              { n: 5, label: "5 Solver" },
              { n: 6, label: "6 Reproject" },
            ].map(({ n, label }) => (
              <button key={n}
                className={`pipeline-tab${activeStep === n ? " active" : ""}`}
                onClick={() => setActiveStep(n)}>
                {label}
              </button>
            ))}
          </div>

          <div className="pipeline-content">
            {/* Step 1: raw container coords */}
            {activeStep === 1 && (
              <div>
                <p className="pipeline-desc">Points as placed — fraction of the <em>container</em> (including any black bars)</p>
                <table className="pipeline-table">
                  <thead><tr><th>Point</th><th>x</th><th>y</th></tr></thead>
                  <tbody>
                    {CALIBRATION_POINTS_3D.map((pt, i) => {
                      const cp = pipeline.containerPts[i];
                      return cp && (
                        <tr key={i}>
                          <td style={{ color: pt.color }}>● {labelOf(i)}</td>
                          <td>{cp.x.toFixed(4)}</td>
                          <td>{cp.y.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Step 2: letterbox geometry */}
            {activeStep === 2 && (
              <div>
                <p className="pipeline-desc">Container vs. video render rect</p>
                <div className="pipeline-kv">
                  <span>Container</span><span>{pipeline.lb.cw.toFixed(0)} × {pipeline.lb.ch.toFixed(0)} px</span>
                  <span>Video AR</span><span>{pipeline.lb.videoAR.toFixed(4)}</span>
                  <span>Render rect</span><span>{pipeline.lb.rw.toFixed(0)} × {pipeline.lb.rh.toFixed(0)} px</span>
                  <span>Offset (ox, oy)</span><span>({pipeline.lb.ox.toFixed(1)}, {pipeline.lb.oy.toFixed(1)}) px</span>
                </div>
                <p className="pipeline-desc" style={{ marginTop: 8 }}>
                  {pipeline.lb.ox > 1 ? "Pillarbox: black bars LEFT & RIGHT" :
                   pipeline.lb.oy > 1 ? "Letterbox: black bars TOP & BOTTOM" :
                   "No bars — video fills container exactly"}
                </p>
                {/* Mini diagram */}
                <svg width="100%" viewBox="0 0 200 120" style={{ marginTop: 8, border: "1px solid #333", borderRadius: 4 }}>
                  <rect x="0" y="0" width="200" height="120" fill="#111" />
                  {(() => {
                    const { cw, ch, ox, oy, rw, rh } = pipeline.lb;
                    const sx = 200 / cw, sy = 120 / ch;
                    return <rect x={ox * sx} y={oy * sy} width={rw * sx} height={rh * sy}
                                 fill="rgba(0,120,255,0.25)" stroke="#4fc3f7" strokeWidth="1" />;
                  })()}
                  <text x="4" y="12" fill="#666" fontSize="8">container</text>
                  <text x={pipeline.lb.ox * 200 / pipeline.lb.cw + 2} y={pipeline.lb.oy * 120 / pipeline.lb.ch + 14}
                        fill="#4fc3f7" fontSize="8">video</text>
                </svg>
              </div>
            )}

            {/* Step 3: video-frame normalized */}
            {activeStep === 3 && (
              <div>
                <p className="pipeline-desc">Points mapped into video frame [0,1] (black-bar offset removed)</p>
                <table className="pipeline-table">
                  <thead><tr><th>Point</th><th>x</th><th>y</th><th>in-frame?</th></tr></thead>
                  <tbody>
                    {CALIBRATION_POINTS_3D.map((pt, i) => {
                      const vp = pipeline.videoFramePts[i];
                      if (!vp) return null;
                      const ok = vp.x >= 0 && vp.x <= 1 && vp.y >= 0 && vp.y <= 1;
                      return (
                        <tr key={i}>
                          <td style={{ color: pt.color }}>● {labelOf(i)}</td>
                          <td style={{ color: ok ? "#ddd" : "#f55" }}>{vp.x.toFixed(4)}</td>
                          <td style={{ color: ok ? "#ddd" : "#f55" }}>{vp.y.toFixed(4)}</td>
                          <td style={{ color: ok ? "#5f5" : "#f55" }}>{ok ? "✓" : "✗ OOB!"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Step 4: NDC values */}
            {activeStep === 4 && (
              <div>
                <p className="pipeline-desc">NDC [-1..1] sent to solver (x right, y up)</p>
                <table className="pipeline-table">
                  <thead><tr><th>Point</th><th>x</th><th>y</th></tr></thead>
                  <tbody>
                    {CALIBRATION_POINTS_3D.map((pt, i) => {
                      const np = pipeline.ndcPts[i];
                      return np && (
                        <tr key={i}>
                          <td style={{ color: pt.color }}>● {labelOf(i)}</td>
                          <td>{np.x.toFixed(4)}</td>
                          <td>{np.y.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Step 5: solver result */}
            {activeStep === 5 && (
              <div>
                <p className="pipeline-desc">Camera parameters recovered by DLT + Nelder-Mead</p>
                <div className="pipeline-kv">
                  <span>FOV</span><span>{pipeline.fov.toFixed(2)}°</span>
                  <span>Position X</span><span>{pipeline.pos[0].toFixed(3)} m</span>
                  <span>Position Y</span><span>{pipeline.pos[1].toFixed(3)} m</span>
                  <span>Position Z</span><span>{pipeline.pos[2].toFixed(3)} m</span>
                  <span>LookAt X</span><span>{pipeline.lookAt[0].toFixed(3)}</span>
                  <span>LookAt Y</span><span>{pipeline.lookAt[1].toFixed(3)}</span>
                  <span>LookAt Z</span><span>{pipeline.lookAt[2].toFixed(3)}</span>
                </div>
                <details style={{ marginTop: 8 }}>
                  <summary style={{ fontSize: 10, color: "#888", cursor: "pointer" }}>Solver log</summary>
                  <pre style={{ fontSize: 9, color: "#999", marginTop: 4, whiteSpace: "pre-wrap" }}>
                    {pipeline.solverLog.join("\n")}
                  </pre>
                </details>
              </div>
            )}

            {/* Step 6: reprojection errors */}
            {activeStep === 6 && (
              <div>
                <div style={{
                  fontWeight: "bold", marginBottom: 6, padding: "4px 8px", borderRadius: 4,
                  background: "rgba(0,0,0,0.4)",
                  color: pipeline.quality === "excellent" ? "#00ff88"
                    : pipeline.quality === "good" ? "#88dd44"
                    : pipeline.quality === "fair" ? "#ffcc00" : "#ff4444",
                }}>
                  {pipeline.quality.toUpperCase()} — avg {pipeline.avgErrPx.toFixed(1)} px
                </div>
                <p className="pipeline-desc">● = placed &nbsp; ○ = reprojected &nbsp; — = error</p>
                <table className="pipeline-table">
                  <thead><tr><th>Point</th><th>err (px)</th></tr></thead>
                  <tbody>
                    {CALIBRATION_POINTS_3D.map((pt, i) => {
                      const err = pipeline.errPx[i];
                      if (err == null) return null;
                      const col = err < 5 ? "#00ff88" : err < 15 ? "#ffcc00" : err < 35 ? "#ff8800" : "#ff4444";
                      return (
                        <tr key={i}>
                          <td style={{ color: pt.color }}>● {labelOf(i)}</td>
                          <td style={{ color: col, fontFamily: "monospace" }}>{err.toFixed(1)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <button className="btn btn-primary" onClick={handleAccept}
              style={{ flex: 1, background: pipeline.quality === "poor" ? "#ff4444" : "#00cc66" }}>
              ✓ Accept & Close
            </button>
            <button className="btn btn-secondary" onClick={() => setPipeline(null)} style={{ flex: 1 }}>
              ✎ Adjust
            </button>
          </div>
        </div>
      )}

      {debugInfo && (
        <div
          className="calibration-debug"
          onClick={(e) => e.stopPropagation()}
          style={{
            position: "absolute", top: 0, right: 0, bottom: 0,
            width: "340px",
            background: "rgba(10,12,16,0.97)",
            borderLeft: "1px solid rgba(255,255,255,0.12)",
            zIndex: 25,
            overflow: "auto", padding: 12,
            display: "flex", flexDirection: "column", gap: 10,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong style={{ color: "#fff", fontSize: 14 }}>Detection Steps</strong>
            <button className="btn btn-secondary" onClick={() => setDebugInfo(null)} style={{ padding: "2px 10px", fontSize: 11 }}>✕ Close</button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {debugInfo.steps.map((step, i) => (
              <div key={i} style={{ background: "rgba(255,255,255,0.05)", borderRadius: 6, padding: 7, cursor: "pointer" }} onClick={() => setZoomedStep(i)}>
                <div style={{ fontSize: 12, fontWeight: "bold", color: "#fff", marginBottom: 3 }}>{step.label}</div>
                <img
                  src={step.image}
                  alt={step.label}
                  style={{ width: "100%", height: "auto", borderRadius: 4, display: "block" }}
                />
                <div style={{ fontSize: 10, color: "#aaa", marginTop: 3 }}>{step.info}</div>
              </div>
            ))}
          </div>

          <div style={{ fontSize: 11, fontFamily: "monospace", color: "#ccc", whiteSpace: "pre-wrap", lineHeight: 1.5, background: "rgba(0,0,0,0.5)", padding: 8, borderRadius: 6 }}>
            <strong style={{ color: "#fff" }}>Log:</strong>{"\n"}
            {debugInfo.log.map((line, i) => (
              <span key={i} style={{
                color: line.startsWith("FAIL") ? "#f55"
                     : line.startsWith("SUCCESS") ? "#5f5"
                     : line.startsWith("ERROR") ? "#f55"
                     : "#ccc"
              }}>
                {line}{"\n"}
              </span>
            ))}
          </div>

          {debugInfo.corners && (
            <div style={{ textAlign: "center" }}>
              <button className="btn btn-primary" style={{ width: "100%" }} onClick={() => setDebugInfo(null)}>Use Detected Corners</button>
            </div>
          )}

          {zoomedStep !== null && debugInfo.steps[zoomedStep] && (
            <div
              onClick={() => setZoomedStep(null)}
              style={{
                position: "fixed", inset: 0, zIndex: 100,
                background: "rgba(0,0,0,0.92)",
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                cursor: "zoom-out", padding: 24,
              }}
            >
              <div style={{ fontSize: 16, fontWeight: "bold", color: "#fff", marginBottom: 8 }}>
                {debugInfo.steps[zoomedStep].label}
              </div>
              <img
                src={debugInfo.steps[zoomedStep].image}
                alt={debugInfo.steps[zoomedStep].label}
                style={{
                  maxWidth: "95vw", maxHeight: "80vh",
                  width: "auto", height: "auto",
                  borderRadius: 6,
                }}
              />
              <div style={{ fontSize: 13, color: "#aaa", marginTop: 8 }}>
                {debugInfo.steps[zoomedStep].info}
              </div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 12 }}>Click anywhere to close</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

