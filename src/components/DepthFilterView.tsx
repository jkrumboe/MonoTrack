import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { CalibrationResult } from "../utils/cameraSolver";

interface DepthFilterViewProps {
  videoEl: HTMLVideoElement | null;
  calibration?: CalibrationResult | null;
  currentFrame: number;
  videoAspect?: number;
  /**
   * Pan/zoom camera: maps between the shown frame and the reference frame the
   * calibration belongs to, in video-normalised [0,1] coords. Null = static.
   */
  frameMapping?: FrameMapping | null;
}

type Pt = { x: number; y: number };
interface FrameMapping {
  toReference: (p: Pt) => Pt;
  fromReference: (p: Pt) => Pt;
}

const COURT_HALF_W = 4.5;
const COURT_HALF_L = 9;
const GRID_RES = 8; // pixels per depth-sample cell

/** Court lines as pairs of 3-D endpoints (x, y, z) in metres. */
const COURT_LINES_3D: [[number, number, number], [number, number, number]][] = [
  // Sidelines
  [[-4.5, 0, -9], [-4.5, 0, 9]],
  [[ 4.5, 0, -9], [ 4.5, 0, 9]],
  // End lines
  [[-4.5, 0, -9], [ 4.5, 0, -9]],
  [[-4.5, 0,  9], [ 4.5, 0,  9]],
  // Attack lines
  [[-4.5, 0, -3], [ 4.5, 0, -3]],
  [[-4.5, 0,  3], [ 4.5, 0,  3]],
  // Net centre line (ground-level)
  [[-4.5, 0,  0], [ 4.5, 0,  0]],
  // Longitudinal centre line
  [[ 0,   0, -9], [ 0,   0,  9]],
];

/** Map t ∈ [0,1] (0 = far end, 1 = near end) to an RGB colour. */
function depthToRgb(t: number): [number, number, number] {
  const r = Math.round(Math.min(1, Math.max(0, t * 2)) * 255);
  const g = Math.round(Math.sin(Math.PI * t) * 220);
  const b = Math.round(Math.min(1, Math.max(0, (1 - t) * 2)) * 255);
  return [r, g, b];
}

function buildCamera(calibration: CalibrationResult, aspect: number): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(calibration.fov, aspect, 0.1, 500);
  cam.position.set(...calibration.position);
  cam.lookAt(new THREE.Vector3(...calibration.lookAt));
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  return cam;
}

function projectToCanvas(
  pt3d: THREE.Vector3,
  cam: THREE.PerspectiveCamera,
  CW: number,
  CH: number,
  mapping?: FrameMapping | null,
): [number, number] {
  const p = pt3d.clone().project(cam);
  let n = { x: (p.x + 1) / 2, y: (1 - p.y) / 2 };
  // The camera projects into the reference frame; carry that into this one.
  if (mapping) n = mapping.fromReference(n);
  return [n.x * CW, n.y * CH];
}

interface DepthOverlayOptions {
  zMin: number;
  zMax: number;
  overlayAlpha: number;
  mapping?: FrameMapping | null;
}

function drawDepthOverlay(
  ctx: CanvasRenderingContext2D,
  cam: THREE.PerspectiveCamera,
  groundPlane: THREE.Plane,
  CW: number,
  CH: number,
  opts: DepthOverlayOptions,
): void {
  const { zMin, zMax, overlayAlpha, mapping } = opts;
  const target = new THREE.Vector3();
  ctx.save();
  for (let py = 0; py < CH; py += GRID_RES) {
    for (let px = 0; px < CW; px += GRID_RES) {
      // This frame's pixel, moved to where it sits in the reference frame.
      let n = { x: (px + GRID_RES * 0.5) / CW, y: (py + GRID_RES * 0.5) / CH };
      if (mapping) n = mapping.toReference(n);
      const ndcX = n.x * 2 - 1;
      const ndcY = 1 - n.y * 2;

      const origin = cam.position.clone();
      const wp = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(cam);
      const dir = wp.sub(origin).normalize();
      const hit = new THREE.Ray(origin, dir).intersectPlane(groundPlane, target);

      if (!hit) continue;
      if (Math.abs(target.x) > COURT_HALF_W + 0.5 || Math.abs(target.z) > COURT_HALF_L + 0.5) continue;

      const t = (target.z + COURT_HALF_L) / (COURT_HALF_L * 2);
      const [r, g, b] = depthToRgb(t);
      const withinRange = target.z >= zMin && target.z <= zMax;
      ctx.fillStyle = `rgba(${r},${g},${b},${withinRange ? overlayAlpha : overlayAlpha * 0.25})`;
      ctx.fillRect(px, py, GRID_RES, GRID_RES);
    }
  }
  ctx.restore();
}

function drawCourtLines(
  ctx: CanvasRenderingContext2D,
  cam: THREE.PerspectiveCamera,
  CW: number,
  CH: number,
  mapping?: FrameMapping | null,
): void {
  ctx.save();
  for (const [a, b] of COURT_LINES_3D) {
    // A homography keeps lines straight, so mapping the endpoints is enough.
    const [ax, ay] = projectToCanvas(new THREE.Vector3(...a), cam, CW, CH, mapping);
    const [bx, by] = projectToCanvas(new THREE.Vector3(...b), cam, CW, CH, mapping);
    const isNet = a[2] === 0 && b[2] === 0;
    ctx.strokeStyle = isNet ? "rgba(200,200,255,0.9)" : "rgba(255,255,255,0.75)";
    ctx.lineWidth = isNet ? 2 : 1.5;
    ctx.setLineDash(isNet ? [4, 3] : []);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function drawDepthRangeMarkers(
  ctx: CanvasRenderingContext2D,
  cam: THREE.PerspectiveCamera,
  CW: number,
  CH: number,
  zMin: number,
  zMax: number,
  mapping?: FrameMapping | null,
): void {
  ctx.save();
  for (const [zVal, colour] of [[zMin, "#00e5ff"], [zMax, "#ff6d00"]] as [number, string][]) {
    const [lx, ly] = projectToCanvas(new THREE.Vector3(-COURT_HALF_W, 0, zVal), cam, CW, CH, mapping);
    const [rx, ry] = projectToCanvas(new THREE.Vector3( COURT_HALF_W, 0, zVal), cam, CW, CH, mapping);
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(rx, ry);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

export default function DepthFilterView({
  videoEl,
  calibration,
  currentFrame,
  videoAspect,
  frameMapping,
}: Readonly<DepthFilterViewProps>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Depth-range filter (in metres, court Z axis: -9 = far, +9 = near)
  const [zMin, setZMin] = useState(-9);
  const [zMax, setZMax] = useState( 9);
  const [overlayAlpha, setOverlayAlpha] = useState(0.4);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoEl) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const CW = canvas.width;
    const CH = canvas.height;

    ctx.clearRect(0, 0, CW, CH);
    if (videoEl.readyState >= 2) {
      ctx.drawImage(videoEl, 0, 0, CW, CH);
    }

    if (!calibration) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(0, 0, CW, CH);
      ctx.fillStyle = "#fff";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No calibration — calibrate the court first", CW / 2, CH / 2);
      return;
    }

    const aspect =
      videoAspect ??
      (videoEl.videoWidth > 0 ? videoEl.videoWidth / videoEl.videoHeight : CW / CH);

    const cam = buildCamera(calibration, aspect);
    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

    drawDepthOverlay(ctx, cam, groundPlane, CW, CH, { zMin, zMax, overlayAlpha, mapping: frameMapping });
    drawCourtLines(ctx, cam, CW, CH, frameMapping);
    drawDepthRangeMarkers(ctx, cam, CW, CH, zMin, zMax, frameMapping);
  }, [videoEl, calibration, currentFrame, videoAspect, zMin, zMax, overlayAlpha, frameMapping]);

  return (
    <div className="depth-filter-view">
      <canvas
        ref={canvasRef}
        width={480}
        height={270}
        className="depth-filter-canvas"
      />

      {/* Colour legend */}
      <div className="depth-legend">
        <span className="depth-legend-far">Far (−9 m)</span>
        <div className="depth-legend-bar" />
        <span className="depth-legend-near">Near (+9 m)</span>
      </div>

      {/* Controls */}
      <div className="depth-filter-controls">
        <label>
          Z min&nbsp;<span className="depth-val">{zMin.toFixed(1)} m</span>
          <input
            type="range"
            min={-9}
            max={9}
            step={0.5}
            value={zMin}
            onChange={e => setZMin(Math.min(Number(e.target.value), zMax - 0.5))}
          />
        </label>
        <label>
          Z max&nbsp;<span className="depth-val">{zMax.toFixed(1)} m</span>
          <input
            type="range"
            min={-9}
            max={9}
            step={0.5}
            value={zMax}
            onChange={e => setZMax(Math.max(Number(e.target.value), zMin + 0.5))}
          />
        </label>
        <label>
          Opacity&nbsp;<span className="depth-val">{Math.round(overlayAlpha * 100)}%</span>
          <input
            type="range"
            min={0.05}
            max={0.85}
            step={0.05}
            value={overlayAlpha}
            onChange={e => setOverlayAlpha(Number(e.target.value))}
          />
        </label>
      </div>

      <p className="depth-filter-hint">
        Court Z axis: <strong>−9 m</strong> = far end · <strong>0</strong> = net · <strong>+9 m</strong> = near end.<br />
        The coloured overlay shows where each image region maps onto the ground plane.<br />
        Cyan / orange dashed lines show the active depth range (Z min / Z max).
      </p>
    </div>
  );
}
