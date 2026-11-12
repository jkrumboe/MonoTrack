import type { CalibrationResult } from "../utils/cameraSolver";
import type { Scenario, SynthTouch } from "./synth";

/** Broadcast camera solved from France vs Japan (~47 m behind, 7.4 m up, 9.3° FoV). */
export const BROADCAST_CAM: CalibrationResult = {
  position: [-1.449, 7.447, 46.743],
  lookAt: [0.555, 0.117, -7.338],
  fov: 9.295,
};

/**
 * Phone on a tripod 7 m behind the end line, 3.5 m up, main lens (46° vertical
 * in landscape): both baselines and balls up to ~8 m over the net are in view.
 */
export const TRIPOD_CAM: CalibrationResult = { position: [0, 3.5, 16], lookAt: [0, 1.5, 0], fov: 46 };

const cam = (position: [number, number, number], fov: number): CalibrationResult => ({
  position, lookAt: [0, 0.5, -6], fov,
});

/**
 * Standard side-out: serve from behind one baseline, reception deep on the
 * other side, set at the net, front-row attack into a block, ball deflects
 * out. `S` = +1 when the near (camera) side serves.
 */
function sideOut(S: 1 | -1, opts: { attackZ?: number; hide?: Partial<Record<string, boolean>>; t0?: number } = {}): SynthTouch[] {
  const t0 = opts.t0 ?? 0.5;
  const aZ = opts.attackZ ?? 0.8;
  const hide = opts.hide ?? {};
  return [
    { type: "toss",      t: t0,        pos: [-2.6, 1.6, S * 12.5], g: 1.0, feet: [-2.6, S * 12.8] },
    { type: "serve",     t: t0 + 0.85, pos: [-2.5, 3.1, S * 10.2], g: 1.6, jump: 0.6, feet: [-2.5, S * 10.8] },
    { type: "reception", t: t0 + 1.7,  pos: [-2.0, 1.0, -S * 5.5], g: 1.0, hidden: hide.reception },
    { type: "set",       t: t0 + 2.75, pos: [0.4, 2.6, -S * 0.8],  g: 1.0, hidden: hide.set },
    { type: "attack",    t: t0 + 3.6,  pos: [3.6, 3.2, -S * aZ],   g: 1.6, jump: 0.8, hidden: hide.attack },
    { type: "block",     t: t0 + 3.72, pos: [3.7, 3.3, S * 0.2],   g: 1.0, jump: 0.6, feet: [3.7, S * 0.6] },
    { type: "end",       t: t0 + 4.4,  pos: [6.5, 4.5, -S * 3.0] },
  ];
}

/**
 * Side-out that continues: the attack gets past the block, the serving team
 * digs it, sets and attacks back, and the ball lands in.
 */
function longRally(S: 1 | -1): SynthTouch[] {
  const upToAttack = sideOut(S).slice(0, 5); // toss, serve, reception, set, attack
  const t = upToAttack[upToAttack.length - 1].t;
  return [
    ...upToAttack,
    { type: "dig",    t: t + 0.6,  pos: [2.0, 0.8, S * 5.0],   g: 1.0 },
    { type: "set",    t: t + 1.6,  pos: [0.0, 2.7, S * 0.8],   g: 1.0 },
    { type: "attack", t: t + 2.4,  pos: [-3.2, 3.2, S * 0.8],  g: 1.6, jump: 0.8 },
    { type: "end",    t: t + 2.9,  pos: [-2.0, 0.11, -S * 6.0] },
  ];
}

export const SCENARIOS: Scenario[] = [
  { name: "near-serve", note: "Baseline: near side serves, standard side-out.", camera: BROADCAST_CAM, touches: sideOut(1), seed: 1 },
  { name: "far-serve", note: "Far side serves toward the camera.", camera: BROADCAST_CAM, touches: sideOut(-1), seed: 2 },
  { name: "attacker-hidden", note: "Attacker hidden behind the block (as in France vs Japan).", camera: BROADCAST_CAM, touches: sideOut(1, { hide: { attack: true } }), seed: 3 },
  { name: "receiver-hidden", note: "Receiver not detected; reception rests on rules and size.", camera: BROADCAST_CAM, touches: sideOut(1, { hide: { reception: true } }), seed: 4 },
  { name: "no-size", note: "No ball-size measurements (other ball colour).", camera: BROADCAST_CAM, touches: sideOut(1), noSize: true, seed: 5 },
  { name: "noisy-glitch", note: "4 px noise, 3 frames lost per touch, 7-frame 15 px tracker glitch mid-set.", camera: BROADCAST_CAM, touches: sideOut(1), noisePx: 4, dropAtTouch: 3, glitch: { start: Math.round((0.5 + 3.0) * 30), frames: 7, px: 15 }, seed: 6 },
  { name: "biased-size", note: "Ball size biased +5 px in front of the crowd, 20 % noise.", camera: BROADCAST_CAM, touches: sideOut(1), sizeNoise: 0.2, sizeCrowdBiasPx: 5, seed: 7 },
  { name: "high-camera", note: "Higher, closer camera (12 m up, 38 m back, 12° FoV).", camera: cam([0, 12, 38], 12), touches: sideOut(1), seed: 8 },
  { name: "low-camera", note: "Lower camera (4.5 m up, 50 m back, 9° FoV).", camera: cam([0, 4.5, 50], 9), touches: sideOut(1), seed: 9 },
  { name: "wide-camera", note: "Wide lens from 28 m (18° FoV): more perspective.", camera: cam([0, 8, 28], 18), touches: sideOut(1), seed: 10 },
  { name: "offset-camera", note: "Camera 6 m off-centre.", camera: { position: [6, 7.5, 46], lookAt: [0, 0.5, -6], fov: 9.5 }, touches: sideOut(-1), seed: 11 },
  { name: "60fps", note: "60 fps footage.", camera: BROADCAST_CAM, video: { width: 1920, height: 1080, fps: 60 }, touches: sideOut(1), seed: 12 },
  // Side views: the court's length runs across the image; depth is its width.
  { name: "side-static", note: "Sideline camera 24 m out, 9 m up, whole court in view.", camera: { position: [24, 9, 0], lookAt: [0, 0.5, 0], fov: 30 }, touches: sideOut(1), seed: 15 },
  { name: "side-static-far", note: "Sideline camera 40 m out, 13 m up, slightly off-centre.", camera: { position: [40, 13, -3], lookAt: [0, 0.5, 0], fov: 19 }, touches: sideOut(-1), seed: 16 },
  // Pan/zoom from a fixed spot, with the motion estimate's output (stabilised).
  { name: "side-pan", note: "Sideline camera that follows the ball (0.35 s lag) and zooms ×0.85–1.15; 1.5 px motion-estimate error.", camera: { position: [20, 8, 0], lookAt: [0, 0.5, 0], fov: 22 }, touches: sideOut(1), motion: { followLagS: 0.35, zoom: [0.85, 1.15], zoomPeriodS: 3, hNoisePx: 1.5 }, seed: 17 },
  { name: "behind-pan", note: "Broadcast camera behind the baseline that follows the ball and zooms ×0.9–1.2; 1 px motion error.", camera: BROADCAST_CAM, touches: sideOut(-1), motion: { followLagS: 0.4, zoom: [0.9, 1.2], zoomPeriodS: 4, hNoisePx: 1 }, seed: 18 },
  // The product target: a phone on a tripod behind the end line. Static, wide
  // lens (the whole court in view), 1080p. The ball is ~10 px at the far
  // baseline and ~17 px at the net, so its size says little about depth
  // (20 % size noise); perspective is strong instead.
  { name: "tripod-30", note: "Phone on a tripod 7 m behind the end line, 3.5 m up, 46° lens, 30 fps; near side serves.", camera: TRIPOD_CAM, touches: sideOut(1), sizeNoise: 0.2, sizeCrowdBiasPx: 0, seed: 21 },
  { name: "tripod-60", note: "Same tripod at 60 fps; far side serves.", camera: TRIPOD_CAM, video: { width: 1920, height: 1080, fps: 60 }, touches: sideOut(-1), sizeNoise: 0.2, sizeCrowdBiasPx: 0, seed: 22 },
  { name: "tripod-close-30", note: "Tripod 5 m behind the end line, 3 m up, 56° lens, 30 fps; far side serves.", camera: { position: [0, 3, 14], lookAt: [0, 1.5, 0], fov: 56 }, touches: sideOut(-1), sizeNoise: 0.2, sizeCrowdBiasPx: 0, seed: 23 },
  { name: "tripod-far-60", note: "Tripod 8 m behind the end line, 4 m up, 1.5 m off-centre, 42° lens, 60 fps; near side serves.", camera: { position: [1.5, 4, 17], lookAt: [0, 1.5, 0], fov: 42 }, video: { width: 1920, height: 1080, fps: 60 }, touches: sideOut(1), sizeNoise: 0.2, sizeCrowdBiasPx: 0, seed: 24 },
  { name: "tripod-long-30", note: "Tripod, 30 fps: the attack is dug, set and attacked back, and the ball lands in.", camera: TRIPOD_CAM, touches: longRally(-1), sizeNoise: 0.2, sizeCrowdBiasPx: 0, seed: 25 },
  { name: "side-pan-raw", note: "Known limit: the panning side camera without motion compensation (what the old pipeline saw).", camera: { position: [20, 8, 0], lookAt: [0, 0.5, 0], fov: 22 }, touches: sideOut(1), motion: { followLagS: 0.35, zoom: [0.85, 1.15], zoomPeriodS: 3, stabilize: false }, seed: 17, knownLimit: true },
  { name: "back-row-attack", note: "Known limit: back-row attack contacted 3 m from the net.", camera: BROADCAST_CAM, touches: sideOut(1, { attackZ: 3.0 }), seed: 13, knownLimit: true },
  { name: "long-rally", note: "Known limit: dig, set and second attack after the block (no rules yet).", camera: BROADCAST_CAM, touches: longRally(1), seed: 14, knownLimit: true },
];
