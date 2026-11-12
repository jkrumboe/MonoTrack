/**
 * Edit the rally in 3D: the court from above and from the side, on one
 * shared along-the-court axis, with the solved path and its touches. Drag a
 * touch to where it really was; the solver holds it there and refits the
 * flights either side (see the pins in rallyEdits.ts).
 *
 * From one camera the depth along each line of sight is what the solver
 * cannot see, so the selected touch shows its line of sight (the ball is
 * somewhere on it) and a drag snaps onto it. Hold Shift to place it freely,
 * e.g. when the ball was hidden at the touch.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { FrameData } from "../hooks/useAnalysis";
import type { CalibrationResult } from "../utils/cameraSolver";
import { ballRay, touchColor, type EditedTouch } from "../utils/rallyEdits";

type V3 = [number, number, number];

/** View extents (m): along the court (z), across it (x), height (y). */
const Z_MAX = 13, X_MAX = 7, Y_MAX = 8.5;
/** A drag this close (m) to the line of sight snaps onto it. */
const SNAP_M = 1.2;
const HIT_PX = 11;

const css = (name: string, fallback: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

interface Props {
  frames: FrameData[];
  positions: Map<number, V3>;
  touches: EditedTouch[];
  calibration: CalibrationResult | null;
  aspect: number;
  cur: number;
  /** Frame of the touch being edited, if any. */
  selected: number | null;
  onSelect: (frame: number) => void;
  onPin: (frame: number, pos: V3) => void;
  /** Draw attention to the editor (a problem's fix points here). */
  flash?: boolean;
}

type View = "top" | "side";

export default function CourtEditor3D(p: Readonly<Props>) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const topRef = useRef<HTMLCanvasElement>(null);
  const sideRef = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(0);
  /** Touch being dragged and where it is now (not yet committed). */
  const [drag, setDrag] = useState<{ frame: number; pos: V3; snapped: boolean } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.floor((el.clientWidth - 20) / 2)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const s = w / (2 * Z_MAX); // px per metre, both views
  const topH = Math.round(2 * X_MAX * s), sideH = Math.round(Y_MAX * s);
  // Court z runs left to right in both views; x down the top view; y up the side view.
  const mapper = (view: View) => view === "top"
    ? { px: (q: V3): [number, number] => [w / 2 + q[2] * s, topH / 2 + q[0] * s], h: topH }
    : { px: (q: V3): [number, number] => [w / 2 + q[2] * s, sideH - q[1] * s], h: sideH };

  const selTouch = p.touches.find((t) => t.frame === p.selected) ?? null;
  const ray = useMemo(
    () => (p.calibration && p.selected != null && p.aspect ? ballRay(p.frames, p.selected, p.calibration, p.aspect) : null),
    [p.calibration, p.selected, p.aspect, p.frames],
  );
  const shown = (t: EditedTouch): V3 => (drag && drag.frame === t.frame ? drag.pos : t.position);

  useEffect(() => {
    if (!w) return;
    for (const view of ["top", "side"] as View[]) {
      const cv = view === "top" ? topRef.current : sideRef.current;
      if (!cv) continue;
      const { px, h } = mapper(view);
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.width = `${w}px`; cv.style.height = `${h}px`;
      const ctx = cv.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = css("--bg-stage", "#0d1014"); ctx.fillRect(0, 0, w, h);
      const line = (pts: V3[], color: string, width = 1, dash: number[] = []) => {
        ctx.beginPath(); ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = width;
        pts.forEach((q, i) => (i ? ctx.lineTo(...px(q)) : ctx.moveTo(...px(q))));
        ctx.stroke(); ctx.setLineDash([]);
      };
      const courtLine = "rgba(255,255,255,.45)";
      if (view === "top") {
        ctx.fillStyle = "rgba(61,94,128,.32)";
        const [x0, y0] = px([-4.5, 0, -9]), [x1, y1] = px([4.5, 0, 9]);
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        line([[-4.5, 0, -9], [-4.5, 0, 9], [4.5, 0, 9], [4.5, 0, -9], [-4.5, 0, -9]], courtLine);
        line([[-4.5, 0, -3], [4.5, 0, -3]], courtLine); line([[-4.5, 0, 3], [4.5, 0, 3]], courtLine);
        line([[-5, 0, 0], [5, 0, 0]], "rgba(255,220,90,.9)", 2);
      } else {
        line([[0, 0, -Z_MAX], [0, 0, Z_MAX]], courtLine);
        line([[0, 0, -9], [0, 0, 9]], "rgba(93,113,134,.9)", 3);
        line([[0, 0, 0], [0, 2.43, 0]], "rgba(255,220,90,.9)", 2);
        ctx.fillStyle = "rgba(255,255,255,.4)"; ctx.font = "10px 'Inter Variable', -apple-system, 'Segoe UI', sans-serif";
        ctx.fillText("net 2.43 m", px([0, 2.43, 0])[0] + 4, px([0, 2.43, 0])[1] - 2);
        line([[0, 2.43, -Z_MAX], [0, 2.43, Z_MAX]], "rgba(255,255,255,.08)", 1, [3, 4]);
      }
      // The camera, or an arrow towards it at the edge.
      if (p.calibration) {
        const c = p.calibration.position;
        const [cx, cy] = px(c);
        const inView = cx >= 0 && cx <= w && cy >= 0 && cy <= h;
        const [ex, ey] = [Math.max(8, Math.min(w - 8, cx)), Math.max(8, Math.min(h - 8, cy))];
        ctx.fillStyle = "rgba(255,255,255,.6)"; ctx.font = "11px 'Inter Variable', -apple-system, 'Segoe UI', sans-serif";
        ctx.fillText(inView ? "◉ camera" : "▸ camera", ex - (ex > w / 2 ? 56 : 0), ey + 4);
      }
      // Line of sight through the ball at the selected touch.
      if (ray) {
        const a = ray.origin, d = ray.dir;
        line([a, [a[0] + d[0] * 90, a[1] + d[1] * 90, a[2] + d[2] * 90]], "rgba(0,229,255,.55)", 1.5, [5, 4]);
      }
      // Solved path: the whole rally faint, the last second bright.
      const runsOf = (from: number, to: number) => {
        const out: V3[][] = [];
        let run: V3[] = [];
        for (let f = from; f <= to; f++) {
          const q = p.positions.get(f);
          if (q) run.push(q); else if (run.length) { out.push(run); run = []; }
        }
        if (run.length) out.push(run);
        return out;
      };
      for (const r of runsOf(0, p.frames.length - 1)) line(r, "rgba(242,194,48,.28)", 1.5);
      for (const r of runsOf(p.cur - 30, p.cur)) line(r, "#f2c230", 2);
      const now = p.positions.get(p.cur);
      if (now) {
        ctx.beginPath(); ctx.arc(...px(now), 4, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
      }
      // Touches
      ctx.font = "600 11px -apple-system, Segoe UI, sans-serif";
      for (const t of p.touches) {
        const q = shown(t), [x, y] = px(q), sel = t.frame === p.selected;
        ctx.beginPath(); ctx.arc(x, y, sel ? 7 : 5.5, 0, Math.PI * 2);
        ctx.fillStyle = touchColor(t.label); ctx.fill();
        if (t.pinned || sel) { ctx.strokeStyle = sel ? "#00e5ff" : "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
        if (sel || Math.abs(t.frame - p.cur) <= 30) {
          ctx.fillStyle = touchColor(t.label);
          ctx.fillText(`${t.label}${t.pinned ? " 📌" : ""}`, x + 9, y - 6);
        }
        // Drop line to the floor in the side view.
        if (view === "side") line([q, [q[0], 0, q[2]]], "rgba(255,255,255,.25)", 1, [2, 3]);
      }
      if (drag) {
        const q = drag.pos;
        ctx.fillStyle = "#00e5ff"; ctx.font = "11px ui-monospace, Consolas, monospace";
        ctx.fillText(`x ${q[0].toFixed(1)}  h ${q[1].toFixed(1)}  z ${q[2].toFixed(1)} m${drag.snapped ? "  on line of sight" : ""}`, 6, 13);
      }
    }
  }, [w, s, topH, sideH, p.positions, p.touches, p.frames.length, p.cur, p.selected, p.calibration, ray, drag]);

  /** Mouse position -> court point in a view, keeping the coordinate the view does not show. */
  const toWorld = (view: View, mx: number, my: number, keep: V3, free: boolean): { pos: V3; snapped: boolean } => {
    const mz = (mx - w / 2) / s;
    const other = view === "top" ? (my - topH / 2) / s : (sideH - my) / s;
    let pos: V3 = view === "top" ? [other, keep[1], mz] : [keep[0], other, mz];
    let snapped = false;
    if (ray && !free) {
      // Point of the line of sight nearest the mouse, as seen in this view.
      const [o, d] = [ray.origin, ray.dir];
      const [i, j] = view === "top" ? [0, 2] : [1, 2];
      const tgt = [other, mz];
      const dd = d[i] * d[i] + d[j] * d[j];
      if (dd > 1e-9) {
        const t = Math.max(0, ((tgt[0] - o[i]) * d[i] + (tgt[1] - o[j]) * d[j]) / dd);
        const q: V3 = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
        if (Math.hypot(q[i] - tgt[0], q[j] - tgt[1]) <= SNAP_M && q[1] >= 0) { pos = q; snapped = true; }
      }
    }
    pos[1] = Math.max(0, pos[1]);
    return { pos, snapped };
  };

  const hit = (view: View, mx: number, my: number): EditedTouch | null => {
    const { px } = mapper(view);
    let best: EditedTouch | null = null, bestD = HIT_PX;
    for (const t of p.touches) {
      const [x, y] = px(t.position);
      const d = Math.hypot(x - mx, y - my);
      if (d < bestD) { bestD = d; best = t; }
    }
    return best;
  };

  const handlers = (view: View) => ({
    onPointerDown: (e: React.PointerEvent<HTMLCanvasElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      const t = hit(view, e.clientX - r.left, e.clientY - r.top);
      if (!t) return;
      p.onSelect(t.frame);
      e.currentTarget.setPointerCapture(e.pointerId);
      setDrag({ frame: t.frame, pos: t.position, snapped: false });
    },
    onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      if (!drag) {
        e.currentTarget.style.cursor = hit(view, mx, my) ? "grab" : "default";
        return;
      }
      // The line of sight is the selected touch's, and the dragged touch is selected.
      setDrag({ frame: drag.frame, ...toWorld(view, mx, my, drag.pos, e.shiftKey) });
    },
    onPointerUp: () => {
      if (!drag) return;
      const t = p.touches.find((x) => x.frame === drag.frame);
      const moved = t && Math.hypot(t.position[0] - drag.pos[0], t.position[1] - drag.pos[1], t.position[2] - drag.pos[2]) > 0.05;
      if (moved) p.onPin(drag.frame, drag.pos.map((v) => Math.round(v * 100) / 100) as V3);
      setDrag(null);
    },
  });

  return (
    <div className={`court3d ${p.flash ? "flash" : ""}`} ref={wrapRef}>
      <div className="court3d-view">
        <div className="court3d-title">From above</div>
        <canvas ref={topRef} {...handlers("top")} />
      </div>
      <div className="court3d-view">
        <div className="court3d-title">From the side · height</div>
        <canvas ref={sideRef} {...handlers("side")} />
      </div>
      <div className="court3d-help">
        {selTouch
          ? <>Drag the <b style={{ color: touchColor(selTouch.label) }}>{selTouch.label}</b> to where the ball was played.
              {ray ? " It snaps to the camera's line of sight (cyan): the ball is somewhere on it. Shift: place freely." : " The ball is hidden at this touch: place it freely."}</>
          : "Click a touch to select it, then drag it to where the ball was played. The flights either side follow."}
      </div>
    </div>
  );
}
