import { useState, useRef, useCallback, useEffect } from "react";
import * as THREE from "three";
import { invoke } from "@tauri-apps/api/core";

/**
 * Known 3D court corner positions (doubles court in meters).
 * Court centered at origin, Y=0 on ground plane.
 * Order: far-left, far-right, near-right, near-left
 */
const COURT_CORNERS_3D = [
  { x: -5.485, y: 0, z: -11.885, label: "Far Left" },
  { x: 5.485, y: 0, z: -11.885, label: "Far Right" },
  { x: 5.485, y: 0, z: 11.885, label: "Near Right" },
  { x: -5.485, y: 0, z: 11.885, label: "Near Left" },
];

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

const CORNER_COLORS = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"];

export interface CalibrationResult {
  /** Camera position in world coords */
  position: [number, number, number];
  /** Camera lookAt target */
  lookAt: [number, number, number];
  /** Vertical field of view in degrees */
  fov: number;
}

interface CourtCalibrationProps {
  width: number;
  height: number;
  videoPath: string;
  adjustMode?: boolean;
  initialCalibration?: CalibrationResult;
  onCalibrated: (result: CalibrationResult) => void;
  onCancel: () => void;
}

export default function CourtCalibration({
  width,
  height,
  videoPath,
  adjustMode = false,
  initialCalibration,
  onCalibrated,
  onCancel,
}: CourtCalibrationProps) {
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [activeCorner, setActiveCorner] = useState(0);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [autoStatus, setAutoStatus] = useState<string>("");
  const [zoomedStep, setZoomedStep] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // In adjust mode, project current calibration corners to 2D on mount
  useEffect(() => {
    if (!adjustMode || !initialCalibration) return;
    const aspect = width / height;
    const cam = new THREE.PerspectiveCamera(initialCalibration.fov, aspect, 0.1, 500);
    cam.position.set(...initialCalibration.position);
    cam.lookAt(new THREE.Vector3(...initialCalibration.lookAt));
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();

    const projected: { x: number; y: number }[] = [];
    for (const c of COURT_CORNERS_3D) {
      const v = new THREE.Vector3(c.x, c.y, c.z).project(cam);
      // NDC [-1,1] → container-normalized [0,1]
      projected.push({ x: (v.x + 1) / 2, y: (-v.y + 1) / 2 });
    }
    setPoints(projected);
    setActiveCorner(4);
  }, [adjustMode, initialCalibration, width, height]);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (adjustMode) return;
      if (dragIndex !== null) return;
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;

      if (points.length < 4) {
        setPoints((prev) => [...prev, { x, y }]);
        setActiveCorner((prev) => prev + 1);
      }
    },
    [adjustMode, points.length, dragIndex]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent, index: number) => {
      e.stopPropagation();
      setDragIndex(index);
    },
    []
  );

  useEffect(() => {
    if (dragIndex === null) return;

    const handleMouseMove = (e: MouseEvent) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      setPoints((prev) => {
        const next = [...prev];
        next[dragIndex] = { x, y };
        return next;
      });
    };

    const handleMouseUp = () => setDragIndex(null);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragIndex]);

  const handleApply = useCallback(() => {
    if (points.length !== 4) return;
    // Convert from normalized [0,1] to NDC [-1,1] (y flipped)
    const ndcPoints = points.map((p) => ({
      x: p.x * 2 - 1,
      y: -(p.y * 2 - 1),
    }));
    const result = solveCameraIterative(ndcPoints, width / height);
    if (result) {
      onCalibrated(result);
    }
  }, [points, width, height, onCalibrated]);

  const handleReset = useCallback(() => {
    setPoints([]);
    setActiveCorner(0);
    setDebugInfo(null);
  }, []);

  const handleAutoDetect = useCallback(async () => {
    if (!videoPath) {
      setAutoStatus("ERROR: No video path");
      return;
    }

    setAutoStatus("Running Python court detection...");

    try {
      // Get current timestamp from video element
      const overlayView = containerRef.current?.closest(".overlay-view");
      const video = overlayView?.querySelector("video") as HTMLVideoElement | null;
      const frameTime = video?.currentTime ?? 0;

      const result = await invoke<{
        action: string;
        corners?: number[][];
        steps?: DebugStep[];
        log?: string[];
        message?: string;
      }>("detect_court", {
        videoPath,
        frameTime,
      });

      const steps = result.steps ?? [];
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

        setPoints(corners);
        setActiveCorner(4);
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
      style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0, cursor: adjustMode ? "default" : points.length < 4 ? "crosshair" : "default", zIndex: 20 }}
    >
      <div className="calibration-instructions">
        {adjustMode ? (
          <span>Drag corners to adjust court alignment</span>
        ) : points.length < 4 ? (
          <span>
            Click corner <strong>{activeCorner + 1}/4</strong>:{" "}
            <span style={{ color: CORNER_COLORS[activeCorner] }}>
              {COURT_CORNERS_3D[activeCorner].label}
            </span>
          </span>
        ) : (
          <span>Drag points to adjust, then Apply</span>
        )}
      </div>

      {points.map((p, i) => (
        <div
          key={i}
          onMouseDown={(e) => handleMouseDown(e, i)}
          className="calibration-point"
          style={{
            left: `${p.x * 100}%`,
            top: `${p.y * 100}%`,
            backgroundColor: CORNER_COLORS[i],
            cursor: "grab",
          }}
        >
          <span className="calibration-point-label">{COURT_CORNERS_3D[i].label}</span>
        </div>
      ))}

      {points.length >= 2 && (
        <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
          {points.map((p, i) => {
            const next = points[(i + 1) % points.length];
            if (i >= points.length - 1 && points.length < 4) return null;
            return (
              <line
                key={i}
                x1={`${p.x * 100}%`} y1={`${p.y * 100}%`}
                x2={`${next.x * 100}%`} y2={`${next.y * 100}%`}
                stroke="rgba(255,255,255,0.6)" strokeWidth="2" strokeDasharray="6,4"
              />
            );
          })}
        </svg>
      )}

      <div className="calibration-buttons" onClick={(e) => e.stopPropagation()}>
        {!adjustMode && <button className="btn btn-primary" onClick={handleAutoDetect}>Auto Detect</button>}
        {points.length === 4 && (
          <button className="btn btn-primary" onClick={handleApply}>Apply</button>
        )}
        {points.length > 0 && !adjustMode && (
          <button className="btn btn-secondary" onClick={handleReset}>Reset</button>
        )}
        <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        {autoStatus && (
          <span style={{ color: autoStatus.startsWith("ERROR") ? "#f55" : "#5f5", fontSize: 12, alignSelf: "center", marginLeft: 8 }}>
            {autoStatus}
          </span>
        )}
      </div>

      {debugInfo && (
        <div
          className="calibration-debug"
          onClick={(e) => e.stopPropagation()}
          style={{
            position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
            background: "rgba(0,0,0,0.95)", zIndex: 25,
            overflow: "auto", padding: 16
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <strong style={{ color: "#fff", fontSize: 16 }}>Court Detection Steps</strong>
            <button className="btn btn-secondary" onClick={() => setDebugInfo(null)} style={{ padding: "4px 12px", fontSize: 12 }}>Close</button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12, marginBottom: 16 }}>
            {debugInfo.steps.map((step, i) => (
              <div key={i} style={{ background: "rgba(255,255,255,0.05)", borderRadius: 6, padding: 8, cursor: "pointer" }} onClick={() => setZoomedStep(i)}>
                <div style={{ fontSize: 13, fontWeight: "bold", color: "#fff", marginBottom: 4 }}>{step.label}</div>
                <img
                  src={step.image}
                  alt={step.label}
                  style={{ width: "100%", height: "auto", borderRadius: 4, display: "block" }}
                />
                <div style={{ fontSize: 11, color: "#aaa", marginTop: 4 }}>{step.info}</div>
              </div>
            ))}
          </div>

          <div style={{ fontSize: 12, fontFamily: "monospace", color: "#ccc", whiteSpace: "pre-wrap", lineHeight: 1.6, background: "rgba(0,0,0,0.5)", padding: 10, borderRadius: 6 }}>
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
            <div style={{ marginTop: 12, textAlign: "center" }}>
              <button className="btn btn-primary" onClick={() => setDebugInfo(null)}>Use Detected Corners</button>
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

// ─── Iterative Camera Solver ──────────────────────────────────────────────

/**
 * Project a 3D world point to NDC using a virtual camera.
 */
function projectPoint(
  worldPt: THREE.Vector3,
  camPos: THREE.Vector3,
  lookAt: THREE.Vector3,
  fov: number,
  aspect: number
): { x: number; y: number } | null {
  const cam = new THREE.PerspectiveCamera(fov, aspect, 0.1, 500);
  cam.position.copy(camPos);
  cam.lookAt(lookAt);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();

  const v = worldPt.clone().project(cam);
  // If behind camera, invalid
  if (v.z > 1) return null;
  return { x: v.x, y: v.y };
}

/**
 * Compute total squared reprojection error for 4 correspondences.
 * params = [camX, camY, camZ, lookX, lookY, lookZ, fov]
 */
function reprojectionError(
  params: number[],
  targetNDC: { x: number; y: number }[],
  aspect: number
): number {
  const [cx, cy, cz, lx, ly, lz, fov] = params;
  if (cy < 1 || fov < 10 || fov > 150) return 1e9;

  const camPos = new THREE.Vector3(cx, cy, cz);
  const lookAt = new THREE.Vector3(lx, ly, lz);

  let totalErr = 0;
  for (let i = 0; i < 4; i++) {
    const w = COURT_CORNERS_3D[i];
    const proj = projectPoint(
      new THREE.Vector3(w.x, w.y, w.z),
      camPos, lookAt, fov, aspect
    );
    if (!proj) return 1e9;
    const dx = proj.x - targetNDC[i].x;
    const dy = proj.y - targetNDC[i].y;
    totalErr += dx * dx + dy * dy;
  }
  return totalErr;
}

/**
 * Nelder-Mead simplex optimization.
 * Minimizes f(params) over 6D parameter space.
 */
function nelderMead(
  f: (p: number[]) => number,
  start: number[],
  scales: number[],
  maxIter = 4000,
  tol = 1e-12
): number[] {
  const n = start.length;

  // Build initial simplex
  const simplex: { p: number[]; v: number }[] = [];
  simplex.push({ p: [...start], v: f(start) });
  for (let i = 0; i < n; i++) {
    const p = [...start];
    p[i] += scales[i];
    simplex.push({ p, v: f(p) });
  }

  const alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5;

  for (let iter = 0; iter < maxIter; iter++) {
    simplex.sort((a, b) => a.v - b.v);

    // Convergence check
    if (simplex[n].v - simplex[0].v < tol) break;

    // Centroid of all but worst
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        centroid[j] += simplex[i].p[j];
      }
    }
    for (let j = 0; j < n; j++) centroid[j] /= n;

    const worst = simplex[n];

    // Reflection
    const xr = centroid.map((c, j) => c + alpha * (c - worst.p[j]));
    const fr = f(xr);

    if (fr < simplex[0].v) {
      // Expansion
      const xe = centroid.map((c, j) => c + gamma * (xr[j] - c));
      const fe = f(xe);
      simplex[n] = fe < fr ? { p: xe, v: fe } : { p: xr, v: fr };
    } else if (fr < simplex[n - 1].v) {
      simplex[n] = { p: xr, v: fr };
    } else {
      // Contraction
      const xc = centroid.map((c, j) => c + rho * (worst.p[j] - c));
      const fc = f(xc);
      if (fc < worst.v) {
        simplex[n] = { p: xc, v: fc };
      } else {
        // Shrink
        for (let i = 1; i <= n; i++) {
          for (let j = 0; j < n; j++) {
            simplex[i].p[j] = simplex[0].p[j] + sigma * (simplex[i].p[j] - simplex[0].p[j]);
          }
          simplex[i].v = f(simplex[i].p);
        }
      }
    }
  }

  simplex.sort((a, b) => a.v - b.v);
  return simplex[0].p;
}

/**
 * Solve for camera pose by iteratively minimizing reprojection error.
 * Uses Nelder-Mead to find camPos, lookAt, and FOV that make the
 * 3D court corners project exactly onto the clicked 2D positions.
 */
export function solveCameraIterative(
  ndcPoints: { x: number; y: number }[],
  aspect: number
): CalibrationResult | null {
  if (ndcPoints.length !== 4) return null;

  // Initial guess: typical broadcast camera behind near baseline
  // params = [camX, camY, camZ, lookX, lookY, lookZ, fov]
  const start = [0, 10, 22, 0, 0, -3, 40];

  const scales = [2, 3, 5, 2, 1, 3, 10];

  const cost = (p: number[]) => reprojectionError(p, ndcPoints, aspect);

  // Run optimizer from multiple starting points for robustness
  let bestParams = start;
  let bestCost = Infinity;

  const starts = [
    [0, 10, 22, 0, 0, -3, 40],      // behind near baseline, center
    [0, 12, 25, 0, 0.5, -5, 35],    // further back, higher, look slightly above ground
    [0, 8, 18, 0, 0, -2, 50],       // closer, wider FOV
    [-1, 10, 20, 0, 0.3, -4, 45],   // slightly left
    [1, 10, 20, 0, -0.3, -4, 45],   // slightly right
    [0, 15, 28, 0, 1, -6, 30],      // high up, far back, narrow FOV
    [0, 7, 16, 0, 0, -1, 55],       // close, wide angle
    [0, 10, 22, 0, 0, 0, 40],       // looking at center
  ];

  for (const s of starts) {
    const params = nelderMead(cost, s, scales);
    const c = cost(params);
    if (c < bestCost) {
      bestCost = c;
      bestParams = params;
    }
  }

  console.log("[solver] Final reprojection error:", bestCost.toExponential(3));

  const [cx, cy, cz, lx, ly, lz, fov] = bestParams;

  return {
    position: [cx, Math.abs(cy), cz],
    lookAt: [lx, ly, lz],
    fov: Math.max(15, Math.min(120, fov)),
  };
}
