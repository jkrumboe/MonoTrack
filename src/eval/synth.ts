/**
 * Synthetic rallies with exact 3D ground truth.
 *
 * A rally is scripted as touches (type, time, 3D ball position) with a
 * ballistic flight between consecutive touches. Everything the analyzer would
 * report is generated from that truth through a real calibrated camera and
 * then corrupted the way broadcast footage is: pixel noise, detections lost
 * at touches, a run of frames where the tracker sits off the ball, noisy and
 * crowd-biased ball sizes, player boxes with lifted feet while jumping,
 * hidden touchers, distractor players at the net, and optionally a camera
 * that pans and zooms to follow play.
 */
import * as THREE from "three";
import type { FrameData } from "../hooks/useAnalysis";
import type { CalibrationResult } from "../utils/cameraSolver";
import { cameraFor, projectNorm } from "./pipeline";
import type { Fixture, TouchLabel, TouchType, Truth } from "./types";

const G = 9.81;
const BALL_D = 0.21;
const PLAYER_H = 1.9;

export interface SynthTouch {
  type: TouchType | "end";
  /** Seconds from the start of the clip. */
  t: number;
  /** True ball position at the touch (m): x across, y up, z along the court. */
  pos: [number, number, number];
  /** Effective downward acceleration (× g) of the flight starting here. */
  g?: number;
  /** How high the toucher's feet are off the floor at contact (m). */
  jump?: number;
  /** Toucher not detected (e.g. attacker hidden behind the block). */
  hidden?: boolean;
  /** Where the toucher stands, if not right under the ball. */
  feet?: [number, number];
}

export interface Scenario {
  name: string;
  /** Short note on what the scenario stresses. */
  note: string;
  camera: CalibrationResult;
  video?: { width: number; height: number; fps: number };
  touches: SynthTouch[];
  /** Detection noise, px (σ). */
  noisePx?: number;
  /** Frames lost right after each touch (the player hides the ball). */
  dropAtTouch?: number;
  /** A run of frames where the tracker sits this many px off the ball. */
  glitch?: { start: number; frames: number; px: number };
  /** Ball diameter noise (σ, fraction) and extra px when in front of the crowd. */
  sizeNoise?: number;
  sizeCrowdBiasPx?: number;
  /** No ball-size measurements at all. */
  noSize?: boolean;
  /** Distractor players (court x, z) standing all rally. */
  distractors?: [number, number][];
  seed?: number;
  /** Excluded from the regression gate: documents a known limit. */
  knownLimit?: boolean;
  /**
   * Camera that pans and zooms from a fixed spot to follow the ball. Raw
   * positions come from the moving camera; with `stabilize` (default) the
   * frames also carry the reference-camera versions the analyzer's motion
   * estimate produces (ball_ref_pos etc.).
   */
  motion?: {
    /** Seconds the camera lags behind the ball. */
    followLagS: number;
    /** FoV multiplier range (zoom out/in) over a slow cycle. */
    zoom?: [number, number];
    zoomPeriodS?: number;
    /** Error of the motion estimate, px (σ, per frame). */
    hNoisePx?: number;
    /** Emit the stabilised fields (false = what the old pipeline sees). */
    stabilize?: boolean;
  };
}

/** Deterministic RNG (mulberry32) with a normal sampler. */
function rng(seed: number) {
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(1e-12, uni()), v = uni();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { uni, normal };
}

/** Ball position at time t, or null outside the rally. Same arc model as the solver. */
function ballAt(touches: SynthTouch[], t: number): [number, number, number] | null {
  for (let k = 0; k + 1 < touches.length; k++) {
    const a = touches[k], b = touches[k + 1];
    if (t < a.t || t > b.t) continue;
    const T = b.t - a.t, s = t - a.t, w = s / T;
    const g = (a.g ?? 1) * G;
    return [
      a.pos[0] + (b.pos[0] - a.pos[0]) * w,
      a.pos[1] + (b.pos[1] - a.pos[1]) * w + 0.5 * g * s * (T - s),
      a.pos[2] + (b.pos[2] - a.pos[2]) * w,
    ];
  }
  return null;
}

/** Default distractors: three front-row and three back-row players per side. */
export const DEFAULT_DISTRACTORS: [number, number][] = [
  [-3, 1.2], [0, 1.0], [3, 1.2], [-3, 6], [0, 7], [3, 6],
  [-3, -1.2], [0, -1.0], [3, -1.2], [-3, -6], [0, -7], [3, -6],
];

export function generate(sc: Scenario): { fixture: Fixture; truth: Truth } {
  const video = sc.video ?? { width: 1920, height: 1080, fps: 30 };
  const { width: W, height: H, fps } = video;
  const aspect = W / H;
  const refCam = cameraFor(sc.camera, aspect);
  const r = rng(sc.seed ?? 1);
  const noise = sc.noisePx ?? 2;
  const touches = sc.touches;
  const tEnd = touches[touches.length - 1].t;
  const nFrames = Math.ceil((tEnd + 0.7) * fps);
  const motion = sc.motion;
  const stabilize = motion ? motion.stabilize !== false : false;
  const fyOf = (fovDeg: number) => H / 2 / Math.tan((fovDeg * Math.PI) / 360);

  // Per-frame camera. Static: the reference camera. Panning: same position,
  // turning towards a lagged copy of the ball, with a slow zoom in and out.
  const cams: THREE.PerspectiveCamera[] = [];
  const base = new THREE.Vector3(...sc.camera.lookAt);
  let aim = base.clone();
  for (let f = 0; f < nFrames; f++) {
    if (!motion) { cams.push(refCam); continue; }
    const p = ballAt(touches, f / fps);
    const want = p ? new THREE.Vector3(p[0], Math.min(p[1], 3), p[2]) : base;
    aim = aim.clone().lerp(want, Math.min(1, 1 / (fps * motion.followLagS)));
    const [z0, z1] = motion.zoom ?? [1, 1];
    const zf = z0 + (z1 - z0) * 0.5 * (1 - Math.cos((2 * Math.PI * f) / (fps * (motion.zoomPeriodS ?? 4))));
    cams.push(cameraFor({ position: sc.camera.position, lookAt: [aim.x, aim.y, aim.z], fov: sc.camera.fov * zf }, aspect));
  }

  // This frame's pixels -> reference pixels. Exact for a camera that only
  // turns and zooms: every point on a pixel's ray lands on one reference pixel.
  // A per-frame offset stands in for motion-estimation error.
  // (Drawn only for moving cameras, so static scenarios keep their exact data.)
  const hNoise: [number, number][] = motion
    ? cams.map(() => [r.normal() * (motion.hNoisePx ?? 0), r.normal() * (motion.hNoisePx ?? 0)])
    : [];
  const toRef = (f: number, img: [number, number]): [number, number] => {
    const cam = cams[f];
    if (cam === refCam) return img;
    const d = new THREE.Vector3(img[0] * 2 - 1, 1 - img[1] * 2, 0.5).unproject(cam).sub(cam.position).normalize();
    const pt = cam.position.clone().addScaledVector(d, 40);
    const [u, v] = projectNorm(refCam, [pt.x, pt.y, pt.z]);
    return [u + hNoise[f][0] / W, v + hNoise[f][1] / H];
  };

  // Frames where the detection is missing: right after each touch.
  const dropped = new Set<number>();
  for (const tc of touches.slice(1, -1)) {
    const f0 = Math.round(tc.t * fps);
    for (let d = 0; d < (sc.dropAtTouch ?? 2); d++) dropped.add(f0 + d);
  }

  // Player boxes per frame: touchers around their touch, distractors always.
  type Person = { x: number; z: number; lift: (f: number) => number; from: number; to: number };
  const people: Person[] = [];
  for (const tc of touches) {
    if (tc.type === "end" || tc.hidden) continue;
    const f0 = Math.round(tc.t * fps);
    const jump = tc.jump ?? 0;
    const [fx, fz] = tc.feet ?? [tc.pos[0], tc.pos[2]];
    // A jump lasts ~0.6 s with the peak at contact.
    const half = Math.round(0.3 * fps);
    people.push({
      x: fx, z: fz, from: f0 - 20, to: f0 + 20,
      lift: (f) => {
        const d = Math.abs(f - f0);
        return d >= half ? 0 : jump * (1 - (d / half) ** 2);
      },
    });
  }
  for (const [x, z] of sc.distractors ?? DEFAULT_DISTRACTORS) {
    people.push({ x, z, from: 0, to: nFrames, lift: () => 0 });
  }

  const truePath: Record<number, [number, number, number]> = {};
  const frames: FrameData[] = [];
  const visible: number[] = [];
  for (let f = 0; f < nFrames; f++) {
    const t = f / fps;
    const cam = cams[f];
    const fd: FrameData = { frame: f, timestamp: Math.round(t * 1e4) / 1e4 } as FrameData;
    const p = t >= touches[0].t ? ballAt(touches, t) : null;
    if (p) {
      truePath[f] = p;
      if (!dropped.has(f)) {
        let [u, v] = projectNorm(cam, p);
        u += (r.normal() * noise) / W;
        v += (r.normal() * noise) / H;
        if (sc.glitch && f >= sc.glitch.start && f < sc.glitch.start + sc.glitch.frames) {
          u += sc.glitch.px / W;
          v -= (0.5 * sc.glitch.px) / H;
        }
        fd.ball_img_pos = [u, v];
        fd.ball_interpolated = false;
        fd.ball_conf = 0.8;
        if (!sc.noSize) {
          const fwd = cam.getWorldDirection(new THREE.Vector3());
          const depth = new THREE.Vector3(...p).sub(cam.position).dot(fwd);
          let d = (fyOf(cam.fov) * BALL_D) / depth * (1 + r.normal() * (sc.sizeNoise ?? 0.1));
          // In front of the crowd (upper part of the frame) the colour mask bleeds.
          if (v < 0.35) d += sc.sizeCrowdBiasPx ?? 2;
          fd.ball_diameter_px = Math.round(d * 100) / 100;
        }
        // Players near the ball, filtered like the analyzer (0.16 frame widths).
        const boxes: number[][] = [];
        for (const pp of people) {
          if (f < pp.from || f > pp.to) continue;
          const lift = pp.lift(f);
          const [fxN, fyN] = projectNorm(cam, [pp.x, lift, pp.z]);
          const [, hyN] = projectNorm(cam, [pp.x, lift + PLAYER_H, pp.z]);
          const h = fyN - hyN;
          if (h <= 0) continue;
          const w = (0.38 * h * H) / W;
          const box = [fxN - w / 2, hyN, fxN + w / 2, fyN, 0.9];
          const dx = Math.max(box[0] - u, 0, u - box[2]);
          const dy = (Math.max(box[1] - v, 0, v - box[3]) * H) / W;
          if (Math.hypot(dx, dy) <= 0.16) boxes.push(box.map((n) => Math.round(n * 1e5) / 1e5));
        }
        fd.players_near_ball = boxes;
        if (stabilize && cam !== refCam) {
          fd.ball_ref_pos = toRef(f, [u, v]);
          if (fd.ball_diameter_px != null) fd.ball_ref_diameter_px = fd.ball_diameter_px * (fyOf(refCam.fov) / fyOf(cam.fov));
          fd.players_ref = boxes.map((b) => {
            const c = [toRef(f, [b[0], b[1]]), toRef(f, [b[2], b[1]]), toRef(f, [b[0], b[3]]), toRef(f, [b[2], b[3]])];
            return [Math.min(...c.map((q) => q[0])), Math.min(...c.map((q) => q[1])),
                    Math.max(...c.map((q) => q[0])), Math.max(...c.map((q) => q[1])), b[4]];
          });
        }
        visible.push(f);
      }
    }
    frames.push(fd);
  }

  // The analyzer fills short gaps with straight lines (ball_interpolated).
  for (let i = 0; i + 1 < visible.length; i++) {
    const a = visible[i], b = visible[i + 1];
    if (b - a <= 1 || b - a > 18) continue;
    for (const key of ["ball_img_pos", "ball_ref_pos"] as const) {
      const pa = frames[a][key], pb = frames[b][key];
      if (!pa || !pb) continue;
      for (let f = a + 1; f < b; f++) {
        const w = (f - a) / (b - a);
        frames[f][key] = [pa[0] + (pb[0] - pa[0]) * w, pa[1] + (pb[1] - pa[1]) * w];
        frames[f].ball_interpolated = true;
        frames[f].ball_conf = 0;
      }
    }
  }

  // Court corners as the analyzer reports them: in reference-image coords.
  const courtCorners = ([[-4.5, 0, -9], [4.5, 0, -9], [4.5, 0, 9], [-4.5, 0, 9]] as [number, number, number][])
    .map((c) => {
      const [u, v] = projectNorm(refCam, c);
      return [u + (r.normal() * 0.5) / W, v + (r.normal() * 0.5) / H];
    });

  const truthTouches: TouchLabel[] = touches
    .filter((tc) => tc.type !== "end")
    .map((tc) => ({ frame: Math.round(tc.t * fps), type: tc.type as TouchType, position: tc.pos }));

  return {
    fixture: { clip: sc.name, video, courtCorners, frames },
    truth: { clip: sc.name, touches: truthTouches, path: truePath },
  };
}
