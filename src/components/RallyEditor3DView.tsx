/**
 * The rally editor in perspective: the court, the solved path and its
 * touches, to orbit around and drag in (the view in CourtEditor3D).
 *
 * Each touch is a ball with a ring on the floor below it. Dragging the ball
 * moves it up and down only (in a wall facing the view); dragging the ring
 * moves it along the floor at its height. Alt flips the two. With "snap to
 * line of sight" on
 * (Shift flips it), the touch slides along the camera's line of sight
 * through the ball instead. Grabbing the path anywhere else puts a new touch
 * there. The diamonds midway through each flight drag up and down to make
 * the flight higher or flatter. Dragging empty space orbits; right-drag pans;
 * the wheel zooms.
 *
 * While a touch is dragged the flights either side are drawn as they will
 * be (cyan); letting go pins the touch and the path is rebuilt from it.
 */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { CalibrationResult } from "../utils/cameraSolver";
import { touchColor, type EditedTouch } from "../utils/rallyEdits";
import { ARC_MAX_G, ARC_MIN_G, arcAt, gravityForMidHeight, type RefitSegment } from "../utils/rallyRefit";
import { createCourt } from "./SkeletonViewer";

type V3 = [number, number, number];

const G = 9.81;
/** Grab radius (px) of a touch or a diamond, and of the path for a new touch. */
const HIT_PX = 12, PATH_HIT_PX = 8;
/** Furthest (m) a drag may put a touch from the centre of the court, and highest. */
const REACH_M = 60, TOP_M = 30;
const ORBIT_TARGET = new THREE.Vector3(0, 1, 0);
const CYAN = 0x00e5ff;

interface Props {
  w: number;
  h: number;
  positions: Map<number, V3>;
  touches: EditedTouch[];
  segments: RefitSegment[];
  calibration: CalibrationResult | null;
  /** The camera's line of sight through the ball at the selected touch. */
  ray: { origin: V3; dir: V3 } | null;
  selected: number | null;
  cur: number;
  fps: number;
  snap: boolean;
  onSelect: (frame: number) => void;
  onPin: (frame: number, pos: V3) => void;
  onArc: (fromFrame: number, midY: number) => void;
}

type Drag =
  /** A touch, or a new one on the path (`fresh`), and where it is now. */
  | {
    kind: "touch"; frame: number; pos: V3; fresh: boolean; snapped: boolean; mode: "height" | "floor";
    /** Screen offset (px) from the pointer to the ball and to its ring at the grab, so neither jumps to the cursor. */
    offBall: [number, number]; offRing: [number, number];
  }
  /** An arc handle: the flight's first touch and the height midway now. */
  | { kind: "arc"; from: number; midY: number };

interface Three {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  dyn: THREE.Group;
  render: () => void;
  reset: () => void;
}

const vec = (q: V3) => new THREE.Vector3(q[0], q[1], q[2]);

function line(pts: V3[], color: number, opacity = 1, dashed = false): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints(pts.map(vec));
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 0.3, gapSize: 0.2, depthWrite: false })
    : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
  const l = new THREE.Line(geo, mat);
  if (dashed) l.computeLineDistances();
  return l;
}

function disposeAll(group: THREE.Group) {
  group.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose();
  });
  group.clear();
}

/** Point of the line o + t·d (t ≥ 0) nearest the line r0 + s·rd. */
function nearestOnLine(o: V3, d: V3, r0: THREE.Vector3, rd: THREE.Vector3): V3 {
  const D = vec(d), w0 = vec(o).sub(r0);
  const a = D.dot(D), b = D.dot(rd), c = rd.dot(rd), dd = D.dot(w0), e = rd.dot(w0);
  const den = a * c - b * b;
  const t = Math.max(0, den > 1e-9 ? (b * e - c * dd) / den : 0);
  return [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
}

export default function RallyEditor3DView(p: Readonly<Props>) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLCanvasElement>(null);
  const three = useRef<Three | null>(null);
  const props = useRef(p);
  props.current = p;
  const dragRef = useRef<Drag | null>(null);
  const [drag, setDragState] = useState<Drag | null>(null);
  const setDrag = (d: Drag | null) => { dragRef.current = d; setDragState(d); };

  // ── Scene, camera, controls and pointer handling: set up once ────────────
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setClearColor(0x0d1014);
    host.prepend(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x20242e, 0.75));
    scene.add(new THREE.AmbientLight(0xffffff, 0.25));
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(8, 18, 12);
    scene.add(key);
    scene.add(createCourt());
    const dyn = new THREE.Group();
    scene.add(dyn);

    const mouse = (e: PointerEvent): [number, number] => {
      const r = renderer.domElement.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const toScreen = (q: V3): [number, number] | null => {
      const v = vec(q).project(camera);
      if (v.z > 1) return null;
      const { w, h } = props.current;
      return [((v.x + 1) / 2) * w, ((1 - v.y) / 2) * h];
    };
    const mouseRay = (mx: number, my: number) => {
      const { w, h } = props.current;
      const rc = new THREE.Raycaster();
      rc.setFromCamera(new THREE.Vector2((mx / w) * 2 - 1, 1 - (my / h) * 2), camera);
      return rc.ray;
    };
    /** Horizontal direction the view looks along: the normal of a wall facing it. */
    const wallNormal = () => {
      const d = camera.getWorldDirection(new THREE.Vector3()).setY(0);
      return d.lengthSq() < 1e-6 ? new THREE.Vector3(0, 0, 1) : d.normalize();
    };

    type Hit = { kind: "touch"; t: EditedTouch; mode: "height" | "floor" } | { kind: "arc"; sg: RefitSegment } | { kind: "path"; frame: number };
    const pick = (mx: number, my: number): Hit | null => {
      const cur = props.current;
      let best: Hit | null = null, bestD = HIT_PX;
      // The ball (height) first, then its ring on the floor (along the floor).
      for (const mode of ["height", "floor"] as const) {
        for (const t of cur.touches) {
          const q = t.position;
          const s = toScreen(mode === "height" ? q : [q[0], 0, q[2]]);
          if (!s) continue;
          // Touches on top of each other: the selected one is the one grabbed.
          const d = Math.hypot(s[0] - mx, s[1] - my) - (t.frame === cur.selected ? HIT_PX : 0);
          if (d < bestD) { bestD = d; best = { kind: "touch", t, mode }; }
        }
        if (best) return best;
      }
      for (const sg of cur.segments) {
        const s = toScreen(sg.mid);
        if (!s) continue;
        const d = Math.hypot(s[0] - mx, s[1] - my);
        if (d < bestD) { bestD = d; best = { kind: "arc", sg }; }
      }
      if (best) return best;
      bestD = PATH_HIT_PX;
      cur.positions.forEach((q, frame) => {
        const s = toScreen(q);
        if (!s) return;
        const d = Math.hypot(s[0] - mx, s[1] - my);
        if (d < bestD) { bestD = d; best = { kind: "path", frame }; }
      });
      return best;
    };

    // Registered before the controls' own, so a grab can switch them off first.
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const [mx, my] = mouse(e);
      const hit = pick(mx, my);
      if (!hit) return;
      controls.enabled = false;
      renderer.domElement.setPointerCapture(e.pointerId);
      const cur = props.current;
      const offsets = (q: V3) => {
        const b = toScreen(q), g = toScreen([q[0], 0, q[2]]);
        return {
          offBall: (b ? [b[0] - mx, b[1] - my] : [0, 0]) as [number, number],
          offRing: (g ? [g[0] - mx, g[1] - my] : [0, 0]) as [number, number],
        };
      };
      if (hit.kind === "touch") {
        cur.onSelect(hit.t.frame);
        setDrag({ kind: "touch", frame: hit.t.frame, pos: hit.t.position, fresh: false, snapped: false, mode: hit.mode, ...offsets(hit.t.position) });
      } else if (hit.kind === "arc") {
        setDrag({ kind: "arc", from: hit.sg.from, midY: hit.sg.mid[1] });
      } else {
        const q = cur.positions.get(hit.frame)!;
        setDrag({ kind: "touch", frame: hit.frame, pos: q, fresh: true, snapped: false, mode: "floor", ...offsets(q) });
      }
    };
    const onMove = (e: PointerEvent) => {
      const [mx, my] = mouse(e);
      const d = dragRef.current;
      if (!d) {
        const hit = controls.enabled ? pick(mx, my) : null;
        renderer.domElement.style.cursor = !hit ? "" : hit.kind === "touch" ? (hit.mode === "height" ? "ns-resize" : "move") : hit.kind === "arc" ? "ns-resize" : "copy";
        return;
      }
      const r = mouseRay(mx, my);
      if (d.kind === "arc") {
        const sg = props.current.segments.find((x) => x.from === d.from);
        if (!sg) return;
        const at = r.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(wallNormal(), vec(sg.mid)), new THREE.Vector3());
        if (at) setDrag({ ...d, midY: Math.max(0.2, Math.min(TOP_M, at.y)) });
        return;
      }
      const { ray, snap } = props.current;
      const byHeight = (d.mode === "height") !== e.altKey;
      // The ball follows the pointer by height; the ring on the floor follows it
      // along the floor. A new touch dragged off the path is held at its height.
      const onFloor = !byHeight && !d.fresh;
      const off = onFloor ? d.offRing : d.offBall;
      const pr = mouseRay(mx + off[0], my + off[1]);
      let pos: V3 = d.pos, snapped = false;
      if (ray && !d.fresh && snap !== e.shiftKey) {
        snapped = true;
        if (onFloor) {
          // The point on the line of sight whose ring is under the pointer.
          const g = pr.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), new THREE.Vector3());
          const [ox, , oz] = ray.origin, [dx, , dz] = ray.dir;
          const dd = dx * dx + dz * dz;
          if (g && dd > 1e-9) {
            const t = Math.max(0, (dx * (g.x - ox) + dz * (g.z - oz)) / dd);
            pos = [ox + ray.dir[0] * t, ray.origin[1] + ray.dir[1] * t, oz + ray.dir[2] * t];
          }
        } else {
          pos = nearestOnLine(ray.origin, ray.dir, pr.origin, pr.direction);
        }
      } else if (byHeight) {
        // Height only, in a wall through the touch facing the view.
        const at = pr.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(wallNormal(), vec(d.pos)), new THREE.Vector3());
        if (at) pos = [d.pos[0], at.y, d.pos[2]];
      } else {
        // Along the floor: the ring under the pointer, the ball keeping its height.
        const at = pr.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), onFloor ? 0 : -d.pos[1]), new THREE.Vector3());
        if (at) pos = [at.x, d.pos[1], at.z];
      }
      pos = [
        Math.max(-REACH_M, Math.min(REACH_M, pos[0])),
        Math.max(0, Math.min(TOP_M, pos[1])),
        Math.max(-REACH_M, Math.min(REACH_M, pos[2])),
      ];
      setDrag({ ...d, pos, snapped });
    };
    const onUp = (e: PointerEvent) => {
      const d = dragRef.current;
      if (renderer.domElement.hasPointerCapture(e.pointerId)) renderer.domElement.releasePointerCapture(e.pointerId);
      controls.enabled = true;
      if (!d) return;
      const cur = props.current;
      if (d.kind === "arc") {
        const sg = cur.segments.find((x) => x.from === d.from);
        if (sg && Math.abs(sg.mid[1] - d.midY) > 0.03) cur.onArc(d.from, d.midY);
      } else {
        const from = d.fresh ? cur.positions.get(d.frame) : cur.touches.find((t) => t.frame === d.frame)?.position;
        const moved = from && Math.hypot(from[0] - d.pos[0], from[1] - d.pos[1], from[2] - d.pos[2]) > 0.05;
        if (moved) cur.onPin(d.frame, d.pos.map((v) => Math.round(v * 100) / 100) as V3);
        // A click on the path goes to that moment; a drag off it puts a touch there.
        if (d.fresh) cur.onSelect(d.frame);
      }
      setDrag(null);
    };
    const el = renderer.domElement;
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);

    const controls = new OrbitControls(camera, el);
    controls.minDistance = 2;
    controls.maxDistance = 200;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;

    const render = () => {
      renderer.render(scene, camera);
      drawLabels();
    };
    const drawLabels = () => {
      const cv = labelsRef.current;
      if (!cv) return;
      const { w, h, touches, selected, cur } = props.current;
      const dpr = window.devicePixelRatio || 1;
      if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      cv.style.width = `${w}px`; cv.style.height = `${h}px`;
      const ctx = cv.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.font = "600 11px -apple-system, Segoe UI, sans-serif";
      const dg = dragRef.current;
      for (const t of touches) {
        if (t.frame !== selected && Math.abs(t.frame - cur) > 30) continue;
        const q = dg?.kind === "touch" && dg.frame === t.frame ? dg.pos : t.position;
        const s = toScreen(q);
        if (!s) continue;
        ctx.fillStyle = touchColor(t.label);
        ctx.fillText(`${t.label}${t.pinned ? " 📌" : ""}`, s[0] + 10, s[1] - 8);
      }
      ctx.fillStyle = "#00e5ff"; ctx.font = "11px ui-monospace, Consolas, monospace";
      if (dg?.kind === "touch") {
        const q = dg.pos;
        ctx.fillText(`x ${q[0].toFixed(1)}  h ${q[1].toFixed(1)}  z ${q[2].toFixed(1)} m${dg.snapped ? "  on line of sight" : ""}`, 8, 16);
      } else if (dg?.kind === "arc") {
        ctx.fillText(`arc ${dg.midY.toFixed(1)} m high midway`, 8, 16);
      }
    };
    /** Look at the court from the camera's side, above it. */
    const reset = () => {
      const side = (props.current.calibration?.position[0] ?? 1) < 0 ? -1 : 1;
      controls.target.copy(ORBIT_TARGET);
      camera.position.copy(ORBIT_TARGET).add(new THREE.Vector3(side * 20, 15, 6));
      camera.lookAt(ORBIT_TARGET);
      controls.update();
      render();
    };
    controls.addEventListener("change", render);
    three.current = { scene, camera, renderer, controls, dyn, render, reset };
    reset();

    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose();
      });
      renderer.dispose();
      el.remove();
      three.current = null;
    };
  }, []);

  // ── Size ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const t = three.current;
    if (!t || !p.w || !p.h) return;
    t.renderer.setSize(p.w, p.h);
    t.camera.aspect = p.w / p.h;
    t.camera.updateProjectionMatrix();
    t.render();
  }, [p.w, p.h]);

  // ── The path, touches and handles: rebuilt when any of them change ────────
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    const { dyn } = t;
    disposeAll(dyn);
    const shown = (x: EditedTouch): V3 => (drag?.kind === "touch" && drag.frame === x.frame ? drag.pos : x.position);

    // The path, faint, with the last second bright, and its shadow on the floor.
    const runs: { pts: V3[]; recent: boolean }[] = [];
    let run: V3[] = [], recent = false;
    const frames = [...p.positions.keys()].sort((a, b) => a - b);
    let prev = -2;
    for (const f of frames) {
      const r = f > p.cur - 30 && f <= p.cur;
      if (f !== prev + 1 || r !== recent) {
        if (run.length > 1) runs.push({ pts: run, recent });
        run = f === prev + 1 && run.length ? [run[run.length - 1]] : [];
        recent = r;
      }
      run.push(p.positions.get(f)!);
      prev = f;
    }
    if (run.length > 1) runs.push({ pts: run, recent });
    for (const r of runs) {
      dyn.add(line(r.pts, 0xf2c230, r.recent ? 1 : 0.4));
      dyn.add(line(r.pts.map(([x, , z]): V3 => [x, 0.01, z]), 0x000000, 0.35));
    }
    // Flights rebuilt from an edit: dashed, as estimates.
    for (const sg of p.segments) {
      if (!sg.rebuilt) continue;
      const pts: V3[] = [];
      for (let f = sg.from; f <= sg.to; f++) { const q = p.positions.get(f); if (q) pts.push(q); }
      if (pts.length > 1) dyn.add(line(pts, sg.warning ? 0xff7043 : 0xfff0a0, 0.9, true));
    }

    // Line of sight through the ball at the selected touch.
    if (p.ray) {
      const { origin: o, dir: d } = p.ray;
      dyn.add(line([o, [o[0] + d[0] * 90, o[1] + d[1] * 90, o[2] + d[2] * 90]], CYAN, 0.55, true));
    }

    // While dragging: the flights either side of the touch, as they will be.
    if (drag?.kind === "touch") {
      const sorted = p.touches.filter((x) => x.frame !== drag.frame).sort((a, b) => a.frame - b.frame);
      const before = [...sorted].reverse().find((x) => x.frame < drag.frame);
      const after = sorted.find((x) => x.frame > drag.frame);
      const gOf = (from: number) => p.segments.find((s) => s.from === from)?.g ?? G;
      const arc = (a: V3, b: V3, fa: number, fb: number, g: number) => {
        const pts: V3[] = [];
        for (let f = fa; f <= fb; f++) pts.push(arcAt(a, b, fa, fb, g, p.fps, f));
        dyn.add(line(pts, CYAN));
      };
      if (before) arc(before.position, drag.pos, before.frame, drag.frame, gOf(before.frame));
      if (after) arc(drag.pos, after.position, drag.frame, after.frame, gOf(drag.frame));
    }
    if (drag?.kind === "arc") {
      const sg = p.segments.find((x) => x.from === drag.from);
      const a = p.touches.find((x) => x.frame === sg?.from), b = p.touches.find((x) => x.frame === sg?.to);
      if (sg && a && b) {
        const g = Math.min(ARC_MAX_G * G, Math.max(ARC_MIN_G * G, gravityForMidHeight(a.position, b.position, sg.from, sg.to, p.fps, drag.midY)));
        const pts: V3[] = [];
        for (let f = sg.from; f <= sg.to; f++) pts.push(arcAt(a.position, b.position, sg.from, sg.to, g, p.fps, f));
        dyn.add(line(pts, CYAN));
      }
    }

    // Touches: a ball with a drop line and a ring on the floor below it.
    const touchMarker = (q: V3, color: string | number, sel: boolean) => {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(sel ? 0.24 : 0.18, 20, 14), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 }));
      ball.position.set(...q);
      dyn.add(ball);
      dyn.add(line([q, [q[0], 0.01, q[2]]], 0xffffff, 0.35, true));
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(sel ? 0.3 : 0.22, sel ? 0.4 : 0.3, 32),
        new THREE.MeshBasicMaterial({ color: sel ? CYAN : color, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(q[0], 0.015, q[2]);
      dyn.add(ring);
      if (sel) {
        const halo = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.03, 8, 40), new THREE.MeshBasicMaterial({ color: CYAN }));
        halo.position.set(...q);
        halo.rotation.x = -Math.PI / 2;
        dyn.add(halo);
      }
    };
    for (const x of p.touches) touchMarker(shown(x), touchColor(x.label), x.frame === p.selected);
    if (drag?.kind === "touch" && drag.fresh) touchMarker(drag.pos, touchColor("touch"), true);

    // Arc handles: a diamond midway through each flight.
    for (const sg of p.segments) {
      const q: V3 = drag?.kind === "arc" && drag.from === sg.from ? [sg.mid[0], drag.midY, sg.mid[2]] : sg.mid;
      const color = sg.warning ? 0xff7043 : sg.manual ? CYAN : 0xf2c230;
      const dia = new THREE.Mesh(new THREE.OctahedronGeometry(0.16), new THREE.MeshBasicMaterial({ color, wireframe: !sg.manual }));
      dia.position.set(...q);
      dyn.add(dia);
    }

    // The ball now.
    const now = p.positions.get(p.cur);
    if (now) {
      const b = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      b.position.set(...now);
      dyn.add(b);
    }
    t.render();
  }, [p.positions, p.touches, p.segments, p.ray, p.selected, p.cur, p.fps, drag]);

  return (
    <div className="court3d-orbit" ref={hostRef} style={{ width: p.w, height: p.h }}>
      <canvas ref={labelsRef} className="court3d-labels" />
      <button className="inspect-btn court3d-reset" onClick={() => three.current?.reset()}>Reset view</button>
    </div>
  );
}
