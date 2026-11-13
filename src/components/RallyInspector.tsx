/**
 * Inspect & Edit: the rally over its video with the solved path drawn on it,
 * a score, the likely problems, and the tools to correct them by hand (move,
 * add or remove the ball on a frame; add, remove or retype a touch). Every
 * correction re-runs the solver and is saved for the video (see rallyEdits.ts).
 * A trimmed-down version of the eval's rally inspector (src/eval/inspect.html).
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import type { FrameData } from "../hooks/useAnalysis";
import { aimCamera, type CalibrationResult } from "../utils/cameraSolver";
import type { PhysicsReprojectionDebug, TouchSplitDecision } from "../utils/ballReconstruction";
import { TOUCH_KINDS, touchColor, touchMatchFrames, type EditedTouch, type RallyEdits, type TouchKind } from "../utils/rallyEdits";
import { resetSegmentEdits, type RefitSegment } from "../utils/rallyRefit";
import { inspectRally, type Fix, type Problem } from "../utils/rallyInspection";
import CourtEditor3D from "./CourtEditor3D";

type V2 = [number, number];
type V3 = [number, number, number];
type Tool = "none" | "ball";
export type SaveState = "idle" | "saving" | "saved" | "error";


interface Props {
  /** Frames with the ball edits applied. */
  frames: FrameData[];
  fps: number;
  videoSize: { width: number; height: number };
  calibration: CalibrationResult | null;
  positions: Map<number, V3>;
  /** `positions` without the estimates that keep the line going: what the problem list judges. */
  fittedPositions: Map<number, V3>;
  physicsDebug?: PhysicsReprojectionDebug[];
  touchSplits?: TouchSplitDecision[];
  /** Touches with the touch edits applied. */
  touches: EditedTouch[];
  courtErrPx: number | null;
  edits: RallyEdits;
  onEdit: (change: (e: RallyEdits) => RallyEdits) => void;
  /** The flights between touches as shown, with their arc handles (see utils/rallyRefit). */
  segments: RefitSegment[];
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  currentFrame: number;
  onSeek: (frame: number) => void;
  /** The video player, drawn under the overlay. */
  video: ReactNode;
  solving: boolean;
  saveState: SaveState;
  saveError: string | null;
  onAlignCourt: () => void;
}

/** Camera-motion homography nearest a frame (moving camera), or undefined. */
function homographyNear(frames: FrameData[], idx: number): number[] | undefined {
  for (let d = 0; d <= 5; d++) {
    const H = frames[idx - d]?.ref_from_frame ?? frames[idx + d]?.ref_from_frame;
    if (H) return H;
  }
  return undefined;
}

/**
 * World point -> normalised position in the current frame's own image, or
 * null (behind the camera, or the camera motion is unknown here).
 */
function useProjector(calibration: CalibrationResult | null, aspect: number, frames: FrameData[], frame: number) {
  return useMemo(() => {
    if (!calibration || !aspect) return null;
    const cam = new THREE.PerspectiveCamera(calibration.fov, aspect, 0.1, 500);
    aimCamera(cam, calibration);
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();
    const moving = frames.some((f) => f.ref_from_frame);
    let inv: THREE.Matrix3 | null = null;
    if (moving) {
      const H = homographyNear(frames, frame);
      if (!H) return null;
      inv = new THREE.Matrix3().set(H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]).invert();
    }
    const v = new THREE.Vector3();
    return (p: V3): V2 | null => {
      v.set(...p).project(cam);
      if (v.z > 1 || !Number.isFinite(v.x)) return null;
      const q: V2 = [(v.x + 1) / 2, (1 - v.y) / 2];
      if (!inv) return q;
      const r = new THREE.Vector3(q[0], q[1], 1).applyMatrix3(inv);
      return [r.x / r.z, r.y / r.z];
    };
  }, [calibration, aspect, frames, frame]);
}

/** Court lines in world coords (x across, y up, z along; net at z = 0). */
const COURT_LINES: V3[][] = (() => {
  const seg = (a: V3, b: V3, n = 12): V3[] =>
    Array.from({ length: n + 1 }, (_, i) => a.map((v, k) => v + ((b[k] - v) * i) / n) as V3);
  return [
    seg([-4.5, 0, -9], [4.5, 0, -9]), seg([4.5, 0, -9], [4.5, 0, 9]), seg([4.5, 0, 9], [-4.5, 0, 9]), seg([-4.5, 0, 9], [-4.5, 0, -9]),
    seg([-4.5, 0, 0], [4.5, 0, 0]), seg([-4.5, 0, -3], [4.5, 0, -3]), seg([-4.5, 0, 3], [4.5, 0, 3]),
    seg([-5, 2.43, 0], [5, 2.43, 0], 2),
  ];
})();

/** Where a video of this size sits in a box, letterboxed (object-fit: contain). */
function contentBox(w: number, h: number, vw: number, vh: number) {
  const ar = vw / vh;
  return w / h > ar
    ? { x: (w - h * ar) / 2, y: 0, w: h * ar, h }
    : { x: 0, y: (h - w / ar) / 2, w, h: w / ar };
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export default function RallyInspector(p: Readonly<Props>) {
  const { frames, fps, videoSize, edits, onEdit, currentFrame: cur, onSeek, touches } = p;
  const [tool, setTool] = useState<Tool>("none");
  const [advance, setAdvance] = useState(true);
  const [addType, setAddType] = useState<TouchKind>("set");
  const [showHidden, setShowHidden] = useState(false);
  const [flash, setFlash] = useState<"ball" | "touch" | "3d" | null>(null);
  const [zoom, setZoom] = useState({ k: 1, x: 0, y: 0 });
  const stageRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [innerSize, setInnerSize] = useState({ w: 0, h: 0 });

  const insp = useMemo(() => inspectRally({
    frames, fps, positions: p.fittedPositions, physicsDebug: p.physicsDebug, touches, touchSplits: p.touchSplits,
    edits, courtErrPx: p.courtErrPx, hasCalibration: !!p.calibration,
  }), [frames, fps, p.fittedPositions, p.physicsDebug, touches, p.touchSplits, edits, p.courtErrPx, p.calibration]);
  const dismissed = useMemo(() => new Set(edits.dismissed), [edits.dismissed]);

  const aspect = videoSize.width && videoSize.height ? videoSize.width / videoSize.height : 0;
  const project = useProjector(p.calibration, aspect, frames, cur);
  const tol = touchMatchFrames(fps);
  const touchHere = useMemo(() => {
    let best: EditedTouch | null = null;
    for (const t of touches) if (Math.abs(t.frame - cur) <= tol && (!best || Math.abs(t.frame - cur) < Math.abs(best.frame - cur))) best = t;
    return best;
  }, [touches, cur, tol]);
  const removedHere = useMemo(
    () => Object.entries(edits.touches).find(([f, k]) => k === "none" && Math.abs(Number(f) - cur) <= tol)?.[0],
    [edits.touches, cur, tol],
  );

  // ── Editing ───────────────────────────────────────────────────────────────
  const setBall = useCallback((frame: number, pos: V2 | null) => onEdit((e) => ({ ...e, ball: { ...e.ball, [frame]: pos } })), [onEdit]);
  const clearBall = useCallback((frame: number) => onEdit((e) => {
    const ball = { ...e.ball }; delete ball[frame]; return { ...e, ball };
  }), [onEdit]);
  const setTouch = useCallback((frame: number, kind: TouchKind | "none") => onEdit((e) => {
    const t = { ...e.touches };
    // One edit per touch: replace edits of the same touch a frame or two away.
    for (const f of Object.keys(t)) if (Math.abs(Number(f) - frame) <= 2) delete t[Number(f)];
    t[frame] = kind;
    // Not a touch: nor anywhere in 3D.
    const pins = { ...e.pins };
    if (kind === "none") for (const f of Object.keys(pins)) if (Math.abs(Number(f) - frame) <= 2) delete pins[Number(f)];
    return { ...e, touches: t, pins };
  }), [onEdit]);
  const setPin = useCallback((frame: number, pos: V3 | null) => onEdit((e) => {
    const pins = { ...e.pins };
    if (pos) pins[frame] = pos; else delete pins[frame];
    return { ...e, pins };
  }), [onEdit]);
  const clearTouch = useCallback((frame: number) => onEdit((e) => {
    const t = { ...e.touches }; delete t[frame]; return { ...e, touches: t };
  }), [onEdit]);
  /** Height of the ball midway through the flight that starts at this touch, or null: the automatic arc. */
  const setArc = useCallback((frame: number, midY: number | null) => onEdit((e) => {
    const arcs = { ...(e.arcs ?? {}) };
    if (midY == null) delete arcs[frame]; else arcs[frame] = Math.round(midY * 100) / 100;
    return { ...e, arcs };
  }), [onEdit]);
  /** A flight back to the automatic result: its arc and the 3D placements of its two touches. */
  const resetSegment = useCallback((from: number, to: number) => onEdit((e) => resetSegmentEdits(e, from, to, 2)), [onEdit]);
  const setDismissed = useCallback((key: string, on: boolean) => onEdit((e) => ({
    ...e, dismissed: on ? [...new Set([...e.dismissed, key])] : e.dismissed.filter((k) => k !== key),
  })), [onEdit]);

  const flashPanel = (which: "ball" | "touch" | "3d") => {
    setFlash(which);
    window.setTimeout(() => setFlash((f) => (f === which ? null : f)), 1600);
  };
  const applyFix = (fix: Fix, pr: Problem) => {
    switch (fix.kind) {
      case "mark-ball": onSeek(pr.frame); setTool("ball"); flashPanel("ball"); break;
      case "remove-ball": onEdit((e) => ({ ...e, ball: { ...e.ball, ...Object.fromEntries(fix.frames.map((f) => [f, null])) } })); break;
      case "add-touch": onSeek(fix.frame); flashPanel("touch"); break;
      case "set-touch": onSeek(fix.frame); flashPanel("touch"); break;
      case "place-3d": onSeek(fix.frame); flashPanel("3d"); break;
      case "remove-touch": setTouch(fix.frame, "none"); break;
      case "align-court": p.onAlignCourt(); break;
    }
  };
  const focusProblem = (pr: Problem) => onSeek(pr.frame);
  const stepProblem = useCallback((dir: number) => {
    const list = [...insp.open].sort((a, b) => a.frame - b.frame);
    const next = dir > 0 ? list.find((x) => x.frame > cur) : [...list].reverse().find((x) => x.frame < cur);
    if (next) onSeek(next.frame);
  }, [insp.open, cur, onSeek]);

  // Keys: B ball tool, X no ball, T add touch, [ ] problems, Esc leaves the tool.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || ["TEXTAREA", "SELECT"].includes(t.tagName) || (t.tagName === "INPUT" && (t as HTMLInputElement).type !== "range"))) return;
      // Ctrl/Cmd+Z undo, Ctrl/Cmd+Shift+Z or Ctrl+Y redo.
      if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === "z" || e.key === "Z" || e.key === "y" || e.key === "Y")) {
        if (e.key === "y" || e.key === "Y" || e.shiftKey) p.onRedo(); else p.onUndo();
        e.preventDefault();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "b" || e.key === "B") setTool((x) => (x === "ball" ? "none" : "ball"));
      else if (e.key === "x" || e.key === "X") setBall(cur, null);
      else if (e.key === "t" || e.key === "T") { if (!touchHere) setTouch(cur, addType); }
      else if (e.key === "[") stepProblem(-1);
      else if (e.key === "]") stepProblem(1);
      else if (e.key === "Escape") setTool("none");
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cur, touchHere, addType, setBall, setTouch, stepProblem, p.onUndo, p.onRedo]);

  // ── Video stage: zoom, pan, click to place the ball ──────────────────────
  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setInnerSize({ w: el.offsetWidth, h: el.offsetHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      setZoom((z) => {
        const k = Math.max(1, Math.min(8, z.k * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
        if (k === 1) return { k: 1, x: 0, y: 0 };
        const s = k / z.k;
        return {
          k,
          x: Math.min(0, Math.max(r.width * (1 - k), mx - s * (mx - z.x))),
          y: Math.min(0, Math.max(r.height * (1 - k), my - s * (my - z.y))),
        };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);
  const drag = useRef<{ x: number; y: number; zx: number; zy: number; moved: boolean } | null>(null);
  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, zx: zoom.x, zy: zoom.y, moved: false };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    const d = drag.current, el = stageRef.current;
    if (!d || !el) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) < 4) return;
    d.moved = true;
    if (zoom.k <= 1) return;
    setZoom((z) => ({
      ...z,
      x: Math.min(0, Math.max(el.clientWidth * (1 - z.k), d.zx + dx)),
      y: Math.min(0, Math.max(el.clientHeight * (1 - z.k), d.zy + dy)),
    }));
  };
  const onMouseUp = (e: React.MouseEvent) => {
    const d = drag.current;
    drag.current = null;
    if (!d || d.moved || tool !== "ball" || !innerRef.current || !aspect) return;
    const el = innerRef.current, r = el.getBoundingClientRect();
    // Back to the unzoomed box, then into the letterboxed picture.
    const lx = (e.clientX - r.left) * (el.offsetWidth / r.width), ly = (e.clientY - r.top) * (el.offsetHeight / r.height);
    const b = contentBox(el.offsetWidth, el.offsetHeight, videoSize.width, videoSize.height);
    const nx = (lx - b.x) / b.w, ny = (ly - b.y) / b.h;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
    setBall(cur, [nx, ny]);
    if (advance) onSeek(cur + 1);
  };

  // ── Overlay ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !innerSize.w || !aspect) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(innerSize.w * dpr); cv.height = Math.round(innerSize.h * dpr);
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, innerSize.w, innerSize.h);
    const b = contentBox(innerSize.w, innerSize.h, videoSize.width, videoSize.height);
    // Lines stay thin when zoomed in.
    const s = 1 / zoom.k;
    const P = (q: V2): V2 => [b.x + q[0] * b.w, b.y + q[1] * b.h];
    const line = (pts: (V2 | null)[], color: string, w = 1, dash: number[] = []) => {
      ctx.beginPath(); ctx.setLineDash(dash.map((v) => v * s)); ctx.strokeStyle = color; ctx.lineWidth = w * s;
      let pen = false;
      for (const q of pts) {
        if (!q) { pen = false; continue; }
        const [x, y] = P(q);
        if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    const circle = (q: V2 | null, r: number, stroke: string | null, fill: string | null, w = 2, dash: number[] = []) => {
      if (!q) return;
      ctx.beginPath(); ctx.setLineDash(dash.map((v) => v * s)); ctx.arc(...P(q), r * s, 0, Math.PI * 2);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = w * s; ctx.stroke(); }
      ctx.setLineDash([]);
    };
    const cross = (q: V2 | null, r: number, color: string, w = 2) => {
      if (!q) return;
      const [x, y] = P(q); ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = w * s;
      ctx.moveTo(x - r * s, y); ctx.lineTo(x + r * s, y); ctx.moveTo(x, y - r * s); ctx.lineTo(x, y + r * s); ctx.stroke();
    };
    const label = (q: V2 | null, text: string, color: string, dx = 8, dy = -8) => {
      if (!q) return;
      const [x, y] = P(q);
      ctx.font = `600 ${12 * s}px ui-monospace, Consolas, monospace`;
      ctx.lineWidth = 3 * s; ctx.strokeStyle = "rgba(0,0,0,.8)"; ctx.strokeText(text, x + dx * s, y + dy * s);
      ctx.fillStyle = color; ctx.fillText(text, x + dx * s, y + dy * s);
    };

    if (project) COURT_LINES.forEach((l, i) => line(l.map(project), i === 7 ? "rgba(255,220,90,.8)" : "rgba(255,255,255,.55)", 1));
    const pos = p.positions;
    if (project) {
      const past: (V2 | null)[] = [], fut: (V2 | null)[] = [];
      for (let f = cur - 25; f <= cur; f++) { const q = pos.get(f); past.push(q ? project(q) : null); }
      for (let f = cur; f <= cur + 12; f++) { const q = pos.get(f); fut.push(q ? project(q) : null); }
      line(past, "#f2c230", 2); line(fut, "rgba(242,194,48,.4)", 2, [3, 3]);
      for (const t of touches) {
        if (Math.abs(t.frame - cur) > 4) continue;
        const sh = project([t.position[0], 0, t.position[2]]), at = project(t.position);
        circle(sh, 7, t.edited ? "#fff" : null, touchColor(t.label), 1.5);
        if (sh && at) line([sh, at], touchColor(t.label), 1.5, [2, 3]);
        label(sh, `${t.label}${t.edited ? " ✎" : ""}`, touchColor(t.label), 10, 16);
      }
    }
    const fr = frames[cur];
    const solved = pos.get(cur);
    if (project && solved) {
      const at = project(solved), sh = project([solved[0], 0, solved[2]]);
      if (at && sh) {
        line([sh, at], "rgba(255,255,255,.5)", 1, [3, 3]);
        label(sh, `h ${solved[1].toFixed(1)} m`, "#fff", 10, 4);
      }
      cross(at, 7, "#f2c230", 2.5);
    }
    const mine = edits.ball[cur];
    if (mine) {
      circle(mine, 9, "#00e5ff", null, 2.5);
      label(mine, "you", "#00e5ff", 12, -10);
    } else if (fr?.ball_img_pos) {
      const out = p.physicsDebug?.find((d) => d.frame === cur)?.isOutlier;
      circle(fr.ball_img_pos, 9, out ? "#ff5252" : fr.ball_interpolated ? "#bbbbbb" : "#69f0ae", null, 2, fr.ball_interpolated ? [3, 3] : []);
    }
    if (mine === null) label([0.01, 0.06], "ball removed on this frame", "#00e5ff", 0, 0);
    else if (!project && p.calibration) label([0.01, 0.06], "camera motion unknown on this frame: 3D hidden", "#ffab40", 0, 0);
  }, [innerSize, aspect, videoSize, zoom.k, project, p.positions, p.physicsDebug, p.calibration, touches, frames, cur, edits.ball]);

  // ── Side panel pieces ─────────────────────────────────────────────────────
  const fr = frames[cur];
  const mine = edits.ball[cur];
  let ballState: string;
  if (mine) ballState = "placed by you";
  else if (mine === null) ballState = "removed by you";
  else if (!fr?.ball_img_pos) ballState = "not detected";
  else if (fr.ball_interpolated) ballState = "gap fill (not seen)";
  else ballState = `detected · conf ${(fr.ball_conf ?? 0).toFixed(2)}`;
  const solvedHere = p.positions.get(cur);

  const corrections = useMemo(() => {
    type Row = { from: number; to: number; text: string; undo: () => void };
    const rows: Row[] = [];
    const ballFrames = Object.keys(edits.ball).map(Number).sort((a, b) => a - b);
    // Consecutive marks of one kind are one row.
    for (let i = 0; i < ballFrames.length;) {
      const removed = edits.ball[ballFrames[i]] === null;
      let j = i;
      while (j + 1 < ballFrames.length && ballFrames[j + 1] - ballFrames[j] <= 3 && (edits.ball[ballFrames[j + 1]] === null) === removed) j++;
      const group = ballFrames.slice(i, j + 1);
      rows.push({
        from: group[0], to: group[group.length - 1],
        text: `${removed ? "Ball removed" : "Ball placed"}${group.length > 1 ? ` on ${group.length} frames` : ""}`,
        undo: () => onEdit((e) => { const ball = { ...e.ball }; group.forEach((f) => delete ball[f]); return { ...e, ball }; }),
      });
      i = j + 1;
    }
    for (const [f, kind] of Object.entries(edits.touches)) {
      rows.push({ from: Number(f), to: Number(f), text: kind === "none" ? "Touch removed" : `Touch: ${kind}`, undo: () => clearTouch(Number(f)) });
    }
    for (const [f, q] of Object.entries(edits.pins)) {
      rows.push({ from: Number(f), to: Number(f), text: `Touch placed in 3D (h ${q[1].toFixed(1)} m)`, undo: () => setPin(Number(f), null) });
    }
    for (const [f, y] of Object.entries(edits.arcs ?? {})) {
      rows.push({ from: Number(f), to: Number(f), text: `Arc reshaped (${y.toFixed(1)} m high midway)`, undo: () => setArc(Number(f), null) });
    }
    return rows.sort((a, b) => a.from - b.from);
  }, [edits.ball, edits.touches, edits.pins, edits.arcs, onEdit, clearTouch, setPin, setArc]);

  const scoreColor = insp.score >= 80 ? "var(--ok)" : insp.score >= 55 ? "var(--warn)" : "var(--bad)";
  const shown = showHidden ? insp.problems : insp.open;
  const hiddenCount = insp.problems.length - insp.open.length;
  const isCurrent = (pr: Problem) => cur >= pr.frame && cur <= (pr.frameEnd ?? pr.frame);

  let saveText = "";
  if (p.saveState === "saving") saveText = "Saving…";
  else if (p.saveState === "saved") saveText = "Saved for this video";
  else if (p.saveState === "error") saveText = `Not saved: ${p.saveError ?? "error"}`;

  return (
    <div className="inspect-layout">
      <div className="inspect-main">
        <div
          className={`inspect-stage ${tool === "ball" ? "tool-ball" : zoom.k > 1 ? "zoomed" : ""}`}
          ref={stageRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={() => { drag.current = null; }}
          onDoubleClick={() => tool === "none" && setZoom({ k: 1, x: 0, y: 0 })}
        >
          <div ref={innerRef} className="inspect-stage-inner" style={{ transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.k})` }}>
            {p.video}
            <canvas ref={canvasRef} className="inspect-overlay" />
          </div>
          {tool === "ball" && (
            <div className="inspect-tool-hint">
              Click the ball's centre for frame {cur}{advance ? ", then the next frame shows" : ""} · <kbd>Esc</kbd> to stop
            </div>
          )}
          {p.solving && <div className="inspect-solving">Re-solving…</div>}
        </div>
        <div className="inspect-legend">
          <span><i style={{ background: "#69f0ae" }} />detected</span>
          <span><i style={{ background: "#ff5252" }} />off the path</span>
          <span><i style={{ background: "#00e5ff" }} />your correction</span>
          <span><i style={{ background: "#f2c230" }} />3D path</span>
          <span className="muted">wheel: zoom · drag: pan · <kbd>[</kbd> <kbd>]</kbd> problems</span>
        </div>
        <CourtEditor3D
          flash={flash === "3d"}
          frames={frames} positions={p.positions} touches={touches} calibration={p.calibration} aspect={aspect}
          cur={cur} selected={touchHere?.frame ?? null} onSelect={onSeek} onPin={(f, q) => setPin(f, q)}
          fps={fps} segments={p.segments} onArc={setArc} onResetSegment={resetSegment}
          onUndo={p.onUndo} onRedo={p.onRedo} canUndo={p.canUndo} canRedo={p.canRedo}
        />
        <Timeline
          frames={frames} fps={fps} positions={p.positions} touches={touches} edits={edits}
          problems={insp.open} rally={insp.rally} cur={cur} onSeek={onSeek}
        />
      </div>

      <aside className="inspect-side">
        <section className="inspect-card inspect-score">
          <div className="inspect-score-top">
            <div>
              <div className="inspect-h">Rally score</div>
              <div className="inspect-score-num" style={{ color: scoreColor }}>{insp.score.toFixed(0)}</div>
            </div>
            <div className="inspect-score-meta">
              <div>{insp.open.length} open problem{insp.open.length === 1 ? "" : "s"}</div>
              <div>{corrections.length} correction{corrections.length === 1 ? "" : "s"}</div>
              <div className="muted">{insp.stats.touches} touches</div>
            </div>
          </div>
          <div className="inspect-parts">
            {insp.parts.map((part) => (
              <div key={part.name} className="inspect-part" title={part.value == null ? "nothing to judge" : `${(part.value * 100).toFixed(0)}%`}>
                <span>{part.name}</span>
                <span className="inspect-bar"><i style={{ width: `${(part.value ?? 0) * 100}%`, opacity: part.value == null ? 0.25 : 1 }} /></span>
              </div>
            ))}
          </div>
        </section>

        <section className="inspect-card inspect-problems">
          <div className="inspect-h">
            Problems
            {hiddenCount > 0 && (
              <button className="inspect-link" onClick={() => setShowHidden((v) => !v)}>
                {showHidden ? "hide ignored" : `show ${hiddenCount} ignored`}
              </button>
            )}
          </div>
          {shown.length === 0 && <p className="muted inspect-empty">Nothing flagged. Play the rally through to check it by eye.</p>}
          <div className="inspect-problem-list">
            {shown.map((pr) => {
              const off = dismissed.has(pr.key);
              return (
                <div key={pr.key} className={`inspect-problem ${pr.severity} ${isCurrent(pr) ? "current" : ""} ${off ? "ignored" : ""}`}>
                  <button className="inspect-problem-main" onClick={() => focusProblem(pr)}>
                    <span className="inspect-problem-meta">{pr.severity} · {pr.frameEnd != null && pr.frameEnd !== pr.frame ? `f${pr.frame}–${pr.frameEnd}` : `f${pr.frame}`}</span>
                    <span className="inspect-problem-title">{pr.title}</span>
                    <span className="inspect-problem-detail">{pr.detail}</span>
                  </button>
                  <div className="inspect-problem-actions">
                    {!off && pr.fixes.map((fix) => (
                      <button key={fix.kind + fix.label} className="inspect-btn primary" onClick={() => applyFix(fix, pr)}>{fix.label}</button>
                    ))}
                    <button className="inspect-btn" onClick={() => setDismissed(pr.key, !off)}>{off ? "Restore" : "Not a problem"}</button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className={`inspect-card ${flash ? "flash" : ""}`}>
          <div className="inspect-h">Frame {cur} <span className="muted">· {(cur / fps).toFixed(2)} s</span></div>
          <div className={`inspect-edit-row ${flash === "ball" ? "flash" : ""}`}>
            <div className="inspect-kv"><span>Ball</span><b>{ballState}</b></div>
            {solvedHere && (
              <div className="inspect-kv"><span>3D</span><b>height {solvedHere[1].toFixed(1)} m · {solvedHere[2] >= 0 ? "near" : "far"} side {Math.abs(solvedHere[2]).toFixed(1)} m</b></div>
            )}
            <div className="inspect-buttons">
              <button className={`inspect-btn ${tool === "ball" ? "active" : "primary"}`} onClick={() => setTool((t) => (t === "ball" ? "none" : "ball"))} title="B">
                {tool === "ball" ? "Stop placing" : "Place ball"} <kbd>B</kbd>
              </button>
              <button className="inspect-btn" onClick={() => setBall(cur, null)} disabled={mine === null} title="X">No ball here <kbd>X</kbd></button>
              {cur in edits.ball && <button className="inspect-btn" onClick={() => clearBall(cur)}>Undo</button>}
            </div>
            <label className="inspect-check">
              <input type="checkbox" checked={advance} onChange={(e) => setAdvance(e.target.checked)} /> next frame after placing
            </label>
          </div>
          <div className={`inspect-edit-row ${flash === "touch" ? "flash" : ""}`}>
            {touchHere ? (
              <>
                <div className="inspect-kv">
                  <span>Touch</span>
                  <b><i className="inspect-dot" style={{ background: touchColor(touchHere.label) }} />
                    {touchHere.label} at f{touchHere.frame}{touchHere.edited ? " (yours)" : ""}</b>
                </div>
                <div className="inspect-kv">
                  <span>In 3D</span>
                  <b>
                    height {touchHere.position[1].toFixed(1)} m · {Math.abs(touchHere.position[2]).toFixed(1)} m from the net
                    {touchHere.pinned ? " · placed by you" : ""}
                  </b>
                </div>
                <div className="inspect-buttons">
                  <select
                    value={(TOUCH_KINDS as readonly string[]).includes(touchHere.label) ? touchHere.label : ""}
                    onChange={(e) => setTouch(touchHere.frame, e.target.value as TouchKind)}
                    aria-label="Touch type"
                  >
                    {!(TOUCH_KINDS as readonly string[]).includes(touchHere.label) && <option value="">{touchHere.label}…</option>}
                    {TOUCH_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                  <button className="inspect-btn" onClick={() => setTouch(touchHere.frame, "none")}>Not a touch</button>
                  {touchHere.frame in edits.touches && <button className="inspect-btn" onClick={() => clearTouch(touchHere.frame)}>Undo type</button>}
                  {touchHere.pinned && <button className="inspect-btn" onClick={() => setPin(touchHere.frame, null)}>Reset 3D position</button>}
                </div>
              </>
            ) : (
              <>
                <div className="inspect-kv"><span>Touch</span><b className="muted">{removedHere ? `removed by you (f${removedHere})` : "none here"}</b></div>
                <div className="inspect-buttons">
                  <select value={addType} onChange={(e) => setAddType(e.target.value as TouchKind)} aria-label="Type of touch to add">
                    {TOUCH_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                  <button className="inspect-btn primary" onClick={() => setTouch(cur, addType)} title="T">Add touch here <kbd>T</kbd></button>
                  {removedHere && <button className="inspect-btn" onClick={() => clearTouch(Number(removedHere))}>Undo removal</button>}
                </div>
              </>
            )}
          </div>
        </section>

        <section className="inspect-card">
          <div className="inspect-h">
            Your corrections
            {corrections.length > 0 && (
              <button className="inspect-link" onClick={() => {
                if (window.confirm("Remove every correction for this video?")) onEdit((e) => ({ ...e, ball: {}, touches: {} }));
              }}>clear all</button>
            )}
          </div>
          {corrections.length === 0
            ? <p className="muted inspect-empty">None yet. Corrections are saved for this video and applied every time it is opened.</p>
            : (
              <div className="inspect-corrections">
                {corrections.map((r) => (
                  <div key={`${r.text}${r.from}`} className="inspect-correction">
                    <button className="inspect-link" onClick={() => onSeek(r.from)}>f{r.from}{r.to !== r.from ? `–${r.to}` : ""}</button>
                    <span>{r.text}</span>
                    <button className="inspect-x" onClick={r.undo} title="Undo" aria-label="Undo">✕</button>
                  </div>
                ))}
              </div>
            )}
          <div className={`inspect-save ${p.saveState}`}>{saveText}</div>
        </section>
      </aside>
    </div>
  );
}

// ── Timeline ──────────────────────────────────────────────────────────────
const LANES = [
  { id: "problems", name: "problems", h: 12 },
  { id: "ball", name: "ball", h: 22 },
  { id: "y", name: "height", h: 80 },
  { id: "z", name: "along court", h: 60 },
] as const;
const GUTTER = 84;

function Timeline(p: Readonly<{
  frames: FrameData[];
  fps: number;
  positions: Map<number, V3>;
  touches: EditedTouch[];
  edits: RallyEdits;
  problems: Problem[];
  rally: [number, number];
  cur: number;
  onSeek: (f: number) => void;
}>) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<[number, number] | null>(null);
  const [width, setWidth] = useState(0);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const n = Math.max(1, p.frames.length - 1);
  const [f0, f1] = view ?? [0, n];
  const top = 18;
  const lanes = useMemo(() => {
    let y = top;
    return LANES.map((l) => { const o = { ...l, y }; y += l.h + 6; return o; });
  }, []);
  const height = lanes[lanes.length - 1].y + lanes[lanes.length - 1].h + 16;
  const X = useCallback((f: number) => GUTTER + ((f - f0) / Math.max(1, f1 - f0)) * (width - GUTTER - 8), [f0, f1, width]);
  const F = useCallback((x: number) => f0 + ((x - GUTTER) / (width - GUTTER - 8)) * (f1 - f0), [f0, f1, width]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !width) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr); cv.height = Math.round(height * dpr);
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const muted = css("--text-secondary") || "#a0a0b0", fg = css("--text-primary") || "#e0e0e0";
    const faint = "rgba(255,255,255,.05)", grid = "rgba(255,255,255,.14)";
    const colW = Math.max(1, (width - GUTTER - 8) / Math.max(1, f1 - f0));
    const L = Object.fromEntries(lanes.map((l) => [l.id, l]));
    ctx.font = "11px 'Inter Variable', -apple-system, 'Segoe UI', sans-serif";
    for (const l of lanes) {
      ctx.fillStyle = faint; ctx.fillRect(GUTTER, l.y, width - GUTTER - 8, l.h);
      ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(l.name, GUTTER - 8, l.y + l.h / 2);
    }
    ctx.textAlign = "left";
    const from = Math.max(0, Math.floor(f0) - 1), to = Math.min(p.frames.length - 1, Math.ceil(f1) + 1);

    // Problems
    for (const pr of p.problems) {
      const a = X(pr.frame) - colW / 2, b = X(pr.frameEnd ?? pr.frame) + colW / 2;
      ctx.fillStyle = pr.severity === "high" ? "#ef5350" : pr.severity === "medium" ? "#ffb74d" : "#8a8aa0";
      ctx.fillRect(a, L.problems.y + 2, Math.max(3, b - a), L.problems.h - 4);
    }
    // Ball
    { const l = L.ball;
      for (let f = from; f <= to; f++) {
        const fr = p.frames[f], x = X(f) - colW / 2;
        const mine = p.edits.ball[f];
        if (mine !== undefined) {
          ctx.fillStyle = "#00e5ff"; ctx.globalAlpha = mine ? 0.9 : 0.5;
          ctx.fillRect(x, l.y + 1, colW + 0.5, l.h - 2);
        } else if (fr?.ball_img_pos && !fr.ball_interpolated) {
          ctx.fillStyle = "#4caf50"; ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, (fr.ball_conf ?? 0.5) / 0.6);
          ctx.fillRect(x, l.y + 3, colW + 0.5, l.h - 6);
        } else if (fr?.ball_img_pos) {
          ctx.fillStyle = muted; ctx.globalAlpha = 0.45; ctx.fillRect(x, l.y + l.h / 2 - 3, colW + 0.5, 6);
        } else if (f >= p.rally[0] && f <= p.rally[1]) {
          ctx.fillStyle = "#ef5350"; ctx.globalAlpha = 0.7; ctx.fillRect(x, l.y + l.h - 3, colW + 0.5, 3);
        }
      }
      ctx.globalAlpha = 1;
    }
    const series = (l: typeof lanes[number], get: (v: V3) => number, lo: number, hi: number) => {
      const Y = (v: number) => Math.max(l.y, Math.min(l.y + l.h, l.y + l.h - ((v - lo) / (hi - lo)) * l.h));
      ctx.strokeStyle = "#f2c230"; ctx.lineWidth = 2; ctx.beginPath();
      let pen = false;
      for (let f = from; f <= to; f++) {
        const q = p.positions.get(f);
        if (!q) { pen = false; continue; }
        if (pen) ctx.lineTo(X(f), Y(get(q))); else ctx.moveTo(X(f), Y(get(q)));
        pen = true;
      }
      ctx.stroke();
      return Y;
    };
    const refLine = (Y: (v: number) => number, v: number, text: string) => {
      ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(GUTTER, Y(v)); ctx.lineTo(width - 8, Y(v)); ctx.stroke();
      if (text) { ctx.fillStyle = muted; ctx.textBaseline = "alphabetic"; ctx.fillText(text, GUTTER + 3, Y(v) - 3); }
    };
    const Yy = series(L.y, (q) => q[1], 0, 8);
    refLine(Yy, 2.43, "net 2.43 m"); refLine(Yy, 0, "");
    const Yz = series(L.z, (q) => q[2], -12, 12);
    refLine(Yz, 0, "net"); refLine(Yz, 9, "end"); refLine(Yz, -9, "end");

    // Touches across the lanes
    const t0 = lanes[0].y, t1 = lanes[lanes.length - 1].y + lanes[lanes.length - 1].h;
    ctx.font = "600 11px 'Inter Variable', -apple-system, 'Segoe UI', sans-serif"; ctx.textBaseline = "alphabetic";
    for (const t of p.touches) {
      if (t.frame < from || t.frame > to) continue;
      const x = X(t.frame), c = touchColor(t.label);
      ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.setLineDash(t.edited ? [] : [4, 3]);
      ctx.beginPath(); ctx.moveTo(x, t0 - 2); ctx.lineTo(x, t1); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c; ctx.fillText(t.label + (t.edited ? " ✎" : ""), x + 3, t0 - 5);
      ctx.beginPath(); ctx.arc(x, Yy(t.position[1]), 4.5, 0, Math.PI * 2); ctx.fill();
    }
    for (const [f, k] of Object.entries(p.edits.touches)) {
      if (k !== "none") continue;
      const x = X(Number(f));
      ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(x, t0); ctx.lineTo(x, t1); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#00e5ff"; ctx.fillText("✕", x - 4, t0 - 5);
    }
    // Playhead and ticks
    const px = X(p.cur);
    ctx.strokeStyle = fg; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px, t0 - 4); ctx.lineTo(px, t1 + 2); ctx.stroke();
    ctx.fillStyle = muted; ctx.font = "11px 'Inter Variable', -apple-system, 'Segoe UI', sans-serif";
    const step = [1, 5, 10, 25, 50, 100, 250, 500].find((s) => (f1 - f0) / s < 14) ?? 1000;
    for (let f = Math.ceil(f0 / step) * step; f <= f1; f += step) ctx.fillText(String(f), X(f) - 6, height - 2);
  }, [width, height, lanes, f0, f1, X, p.frames, p.positions, p.touches, p.edits, p.problems, p.rally, p.cur]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const at = F(e.clientX - cv.getBoundingClientRect().left);
      const k = e.deltaY > 0 ? 1.25 : 0.8;
      let a = at - (at - f0) * k, b = at + (f1 - at) * k;
      if (b - a < 12) return;
      if (b - a >= n) { setView(null); return; }
      if (a < 0) { b -= a; a = 0; }
      if (b > n) { a -= b - n; b = n; }
      setView([a, b]);
    };
    cv.addEventListener("wheel", onWheel, { passive: false });
    return () => cv.removeEventListener("wheel", onWheel);
  }, [F, f0, f1, n]);

  const frameAt = (e: React.MouseEvent) => {
    const x = e.clientX - e.currentTarget.getBoundingClientRect().left;
    return x < GUTTER ? null : Math.max(0, Math.min(n, Math.round(F(x))));
  };
  const tipText = (f: number) => {
    const fr = p.frames[f], q = p.positions.get(f), mine = p.edits.ball[f];
    const lines = [`frame ${f} · ${(f / p.fps).toFixed(2)} s`];
    if (mine !== undefined) lines.push(mine ? "ball placed by you" : "ball removed by you");
    else lines.push(!fr?.ball_img_pos ? "ball not detected" : fr.ball_interpolated ? "ball not seen (gap fill)" : `ball detected, conf ${(fr.ball_conf ?? 0).toFixed(2)}`);
    if (q) lines.push(`height ${q[1].toFixed(1)} m · along court ${q[2].toFixed(1)} m`);
    for (const t of p.touches) if (Math.abs(t.frame - f) <= 2) lines.push(`${t.label} at f${t.frame}${t.edited ? " (yours)" : ""}`);
    for (const pr of p.problems) if (f >= pr.frame - 1 && f <= (pr.frameEnd ?? pr.frame) + 1) lines.push(`⚠ ${pr.title}`);
    return lines.join("\n");
  };

  return (
    <div className="inspect-timeline">
      <canvas
        ref={ref}
        style={{ height }}
        onClick={(e) => { const f = frameAt(e); if (f != null) p.onSeek(f); }}
        onDoubleClick={() => setView(null)}
        onMouseMove={(e) => { const f = frameAt(e); setTip(f == null ? null : { x: e.clientX, y: e.clientY, text: tipText(f) }); }}
        onMouseLeave={() => setTip(null)}
      />
      <div className="inspect-tl-help">Click to jump · wheel to zoom · double-click to reset · lines: touches (solid = yours) · cyan: your corrections</div>
      {tip && <div className="inspect-tip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 320), top: tip.y + 14 }}>{tip.text}</div>}
    </div>
  );
}
