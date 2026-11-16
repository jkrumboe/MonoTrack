import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { FrameData, CourtDataEntry } from "../hooks/useAnalysis";
import { estimateBallWorldPosition } from "../utils/ballReconstruction";
import type { BallTouch, TrajectoryMode } from "../utils/ballReconstruction";
import { aimCamera, type CalibrationResult } from "../utils/cameraSolver";
import { touchColor } from "../utils/rallyEdits";

// FIVB volleyball court dimensions (metres)
const COURT_LENGTH = 18.0;
const COURT_WIDTH  = 9.0;
const NET_HEIGHT   = 2.43;
const ATTACK_DIST  = 3.0;
const COURT_HALF_L = COURT_LENGTH / 2;
const COURT_HALF_W = COURT_WIDTH / 2;

// Matches the rally solver's bounds: court plus free zone, and room behind
// each baseline for a jump server's run-up.
function inRenderableCourtVolume(x: number, y: number, z: number): boolean {
  return (
    Math.abs(x) <= COURT_HALF_W + 6 &&
    Math.abs(z) <= COURT_HALF_L + 9 &&
    y >= -0.3 &&
    y <= 10
  );
}

// Orbit pivot for the free (non-overlay) view.
const COURT_VIEW_TARGET = new THREE.Vector3(0, 1, 0);
// Box that must stay fully visible after a reset: the court plus a small
// margin, from the floor up to the top of the net posts.
const COURT_FIT_MARGIN = 1.0;
const COURT_FIT_TOP = NET_HEIGHT + 0.6;

/**
 * Place `camera` along `viewDir` (pointing from the target towards the camera)
 * at the closest distance where the whole court fits the current window,
 * respecting both the vertical FOV and the horizontal FOV implied by `aspect`.
 */
function fitCameraToCourt(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  viewDir: THREE.Vector3,
  aspect: number,
) {
  const target = COURT_VIEW_TARGET;
  const dir = viewDir.clone().normalize();

  camera.aspect = aspect;
  camera.updateProjectionMatrix();

  // Orient the camera first so we can read its right/up/forward axes.
  camera.position.copy(target).add(dir);
  camera.lookAt(target);
  camera.updateMatrixWorld();
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const forward = dir.clone().negate();

  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const tanH = tanV * aspect;
  const hx = COURT_HALF_W + COURT_FIT_MARGIN;
  const hz = COURT_HALF_L + COURT_FIT_MARGIN;

  // For a corner at offset r from the target and camera at target - d*forward,
  // its depth is d + r·forward; it fits when |r·right| <= tanH*depth and
  // |r·up| <= tanV*depth. Take the smallest d satisfying every corner.
  let dist = camera.near * 2;
  const r = new THREE.Vector3();
  for (const x of [-hx, hx]) {
    for (const y of [0, COURT_FIT_TOP]) {
      for (const z of [-hz, hz]) {
        r.set(x, y, z).sub(target);
        const depthOffset = r.dot(forward);
        dist = Math.max(
          dist,
          Math.abs(r.dot(right)) / tanH - depthOffset,
          Math.abs(r.dot(up)) / tanV - depthOffset,
        );
      }
    }
  }
  dist *= 1.05; // small breathing room around the edges

  controls.maxDistance = Math.max(controls.maxDistance, dist * 1.5);
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dir, dist);
  camera.lookAt(target);
  controls.enableDamping = false;
  controls.update();
  controls.enableDamping = true;
}

/** Over the video, the ball trail covers this many frames back from now. */
const OVERLAY_TRAIL_FRAMES = 30;
/** Over the video, touches this many frames either side of now are marked. */
const OVERLAY_TOUCH_FRAMES = 30;

interface SkeletonViewerProps {
  overlay?: boolean;
  calibration?: CalibrationResult | null;
  frames?: FrameData[];
  currentFrame?: number;
  videoAspect?: number;
  courtData?: CourtDataEntry[];
  isRealVideo?: boolean;
  overlayVideoAspect?: number;
  resetViewTrigger?: number;
  /** Active trajectory mode — controls fallback behaviour in the 3D viewer. */
  trajectoryMode?: TrajectoryMode;
  /** Player touches from the rally solver, drawn as coloured markers. */
  touches?: BallTouch[];
}

/** Behind the 3D court: the stage colour of the app (--bg-stage in App.css). */
const SCENE_BACKGROUND = 0x0d1014;

/**
 * Clip-space matrix that takes the reference-frame camera's projection to
 * this video frame's, for a camera that pans and zooms from a fixed spot.
 *
 * The analyzer stores per frame `ref_from_frame`: a homography from this
 * frame's normalised image coords [0,1] (y down) to the reference frame's.
 * Its inverse, rewritten in NDC (x, y in [-1,1], y up), acts on clip coords
 * (x, y, w); z passes through, which keeps depth ordering (w changes only
 * slightly for a rotation/zoom). Frames without a trusted estimate borrow the
 * nearest one within a few frames. Null for a static camera.
 */
function frameWarpMatrix(
  frames: FrameData[] | undefined,
  currentFrame: number,
): THREE.Matrix4 | null {
  if (!frames || frames.length === 0 || !frames.some((f) => f.ref_from_frame)) return null;
  const idx = Math.min(Math.max(0, currentFrame), frames.length - 1);
  let H: number[] | undefined;
  for (let d = 0; d <= 5 && !H; d++) {
    H = frames[idx - d]?.ref_from_frame ?? frames[idx + d]?.ref_from_frame;
  }
  if (!H) return null;
  // ref -> frame in normalised coords.
  const G = new THREE.Matrix3().set(H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]).invert();
  // normalised -> NDC and back.
  const N = new THREE.Matrix3().set(2, 0, -1, 0, -2, 1, 0, 0, 1);
  const Ninv = new THREE.Matrix3().set(0.5, 0, 0.5, 0, -0.5, 0.5, 0, 0, 1);
  const h = new THREE.Matrix3().multiplyMatrices(N, G).multiply(Ninv).elements; // column-major
  // Matrix4.set takes row-major; h[col * 3 + row].
  return new THREE.Matrix4().set(
    h[0], h[3], 0, h[6],
    h[1], h[4], 0, h[7],
    0, 0, 1, 0,
    h[2], h[5], 0, h[8],
  );
}

export default function SkeletonViewer({
  overlay = false,
  calibration,
  frames,
  currentFrame = 0,
  videoAspect,
  courtData,
  overlayVideoAspect,
  resetViewTrigger,
  trajectoryMode,
  touches,
}: SkeletonViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayModeRef = useRef(overlay);
  const overlayAspectRef = useRef<number | undefined>(overlayVideoAspect);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    courtGroup: THREE.Group;
    ballMesh: THREE.Mesh;
    ballShadow: THREE.Mesh;
    ballStem: THREE.Line;
    ballTrail: THREE.LineSegments;
    estimatedTrail: THREE.LineSegments;
    groundTrail: THREE.LineSegments;
    dropLines: THREE.LineSegments;
    touchMarkers: THREE.Group;
  } | null>(null);
  const lastRenderSizeRef = useRef({ w: 0, h: 0, aspect: 0 });
  // Overlay on a panning/zooming camera: the calibration belongs to the
  // reference frame, so each video frame's view is that camera followed by
  // the frame's homography. Held as a clip-space matrix; null = static.
  const frameWarpRef = useRef<THREE.Matrix4 | null>(null);

  useEffect(() => {
    overlayModeRef.current = overlay;
  }, [overlay]);

  useEffect(() => {
    overlayAspectRef.current = overlayVideoAspect;
  }, [overlayVideoAspect]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SCENE_BACKGROUND);

    // Camera
    // far=200 clipped the court once the user orbited further out than 200 m.
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 5000);
    camera.position.set(0, 15, 20);
    camera.lookAt(0, 0, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    // Keep zoom within a range where the court stays meaningful on screen.
    controls.minDistance = 2;
    controls.maxDistance = 400;
    controls.target.set(0, 1, 0);

    // Lighting: a hemisphere fill keeps the floor from reading flat, with a
    // key light for shading on the poles and net tape.
    const hemiLight = new THREE.HemisphereLight(0xdfe9ff, 0x20242e, 0.75);
    scene.add(hemiLight);
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(8, 18, 12);
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0xaecbff, 0.35);
    fillLight.position.set(-10, 10, -12);
    scene.add(fillLight);

    // Court
    const courtGroup = createCourt();
    scene.add(courtGroup);

    // Ball visualisation (3D reconstructed position)
    const ballMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 24, 24),
      new THREE.MeshStandardMaterial({
        color: 0xffc107,
        emissive: 0x332200,
        emissiveIntensity: 0.55,
        roughness: 0.35,
        metalness: 0.08,
      }),
    );
    ballMesh.visible = false;
    scene.add(ballMesh);

    const ballShadow = new THREE.Mesh(
      new THREE.CircleGeometry(0.13, 24),
      new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.28, depthWrite: false }),
    );
    ballShadow.rotation.x = -Math.PI / 2;
    ballShadow.position.y = 0.005;
    ballShadow.visible = false;
    scene.add(ballShadow);

    const ballStem = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0xffe082, transparent: true, opacity: 0.7 }),
    );
    ballStem.visible = false;
    scene.add(ballStem);

    const ballTrail = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([]),
      new THREE.LineBasicMaterial({ color: 0xffca28, transparent: true, opacity: 0.65 }),
    );
    scene.add(ballTrail);

    // Stretches of the trail the solver could not fit, estimated to keep the
    // line going through the rally. Dashed: they are not measurements.
    const estimatedTrail = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([]),
      new THREE.LineDashedMaterial({ color: 0xffca28, transparent: true, opacity: 0.65, dashSize: 0.3, gapSize: 0.2 }),
    );
    scene.add(estimatedTrail);

    // The trail's footprint on the floor, plus faint verticals joining the
    // two. A lone line in 3D gives no depth cue on screen; tying it to the
    // court shows where over the court the ball actually travelled.
    const groundTrail = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([]),
      new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }),
    );
    scene.add(groundTrail);

    const dropLines = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([]),
      new THREE.LineBasicMaterial({ color: 0xffe082, transparent: true, opacity: 0.18, depthWrite: false }),
    );
    scene.add(dropLines);

    const touchMarkers = new THREE.Group();
    scene.add(touchMarkers);

    sceneRef.current = {
      scene, camera, renderer, controls, courtGroup,
      ballMesh, ballShadow, ballStem, ballTrail, estimatedTrail, groundTrail, dropLines, touchMarkers,
    };

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0) {
        if (lastRenderSizeRef.current.w !== w || lastRenderSizeRef.current.h !== h) {
          renderer.setSize(w, h, false);
          lastRenderSizeRef.current.w = w;
          lastRenderSizeRef.current.h = h;
        }
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, w, h);
        renderer.clear();

        const isOverlay = overlayModeRef.current;
        const videoAspect = overlayAspectRef.current;
        if (isOverlay && videoAspect && videoAspect > 0) {
          const containerAR = w / h;
          let renderW: number, renderH: number, offsetX: number, offsetYTop: number;
          if (videoAspect > containerAR) {
            renderW = w;
            renderH = w / videoAspect;
            offsetX = 0;
            offsetYTop = (h - renderH) / 2;
          } else {
            renderH = h;
            renderW = h * videoAspect;
            offsetX = (w - renderW) / 2;
            offsetYTop = 0;
          }
          const offsetYGl = h - offsetYTop - renderH;
          if (lastRenderSizeRef.current.aspect !== videoAspect) {
            camera.aspect = videoAspect;
            camera.updateProjectionMatrix();
            lastRenderSizeRef.current.aspect = videoAspect;
          }
          renderer.setScissorTest(true);
          renderer.setViewport(offsetX, offsetYGl, renderW, renderH);
          renderer.setScissor(offsetX, offsetYGl, renderW, renderH);
        } else {
          const nextAspect = w / h;
          if (lastRenderSizeRef.current.aspect !== nextAspect) {
            camera.aspect = nextAspect;
            camera.updateProjectionMatrix();
            lastRenderSizeRef.current.aspect = nextAspect;
          }
        }
      }

      // Don't update controls in overlay mode — it would fight camera.lookAt() set by calibration.
      if (!overlayModeRef.current) controls.update();
      if (overlayModeRef.current) {
        // Rebuild from the plain calibrated projection, then warp it into
        // the current video frame (no-op for a static camera).
        camera.updateProjectionMatrix();
        const warp = frameWarpRef.current;
        if (warp) {
          camera.projectionMatrix.premultiply(warp);
          camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        }
      }
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0) {
        renderer.setSize(w, h);
        lastRenderSizeRef.current.w = w;
        lastRenderSizeRef.current.h = h;
      }
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  // Per-frame warp for the overlay when the camera pans and zooms.
  useEffect(() => {
    frameWarpRef.current = overlay ? frameWarpMatrix(frames, currentFrame) : null;
  }, [overlay, frames, currentFrame]);

  // Toggle overlay mode: transparent background + lock/unlock orbit controls
  useEffect(() => {
    if (!sceneRef.current) return;
    const { scene, controls, courtGroup } = sceneRef.current;
    if (overlay) {
      scene.background = null;
      controls.enabled = false;
    } else {
      scene.background = new THREE.Color(SCENE_BACKGROUND);
      controls.enabled = true;
    }

    // Over the video only the court lines are drawn: the painted floor and
    // the free zone are scenery, and would tint the footage.
    courtGroup.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      if (obj.name === "court-free-zone" || obj.name === "court-surface" || obj.name === "court-front-zone") {
        obj.visible = !overlay;
      }
      if (obj.name === "court-free-zone") return;
      if (obj.name !== "court-surface" && obj.name !== "court-front-zone") return;
      const mat = obj.material;
      if (mat instanceof THREE.MeshStandardMaterial) {
        mat.transparent = overlay;
        mat.opacity = overlay ? 0.28 : 1.0;
        mat.depthWrite = !overlay;
        mat.needsUpdate = true;
      }
    });
  }, [overlay]);

  // Apply calibration to camera
  useEffect(() => {
    if (!sceneRef.current || !calibration) return;
    const { camera, controls, renderer } = sceneRef.current;
    const container = containerRef.current;

    // Re-sync camera aspect & renderer size (container may have toggled display:none)
    if (container) {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0) {
        const targetAspect = overlay && overlayVideoAspect && overlayVideoAspect > 0
          ? overlayVideoAspect
          : (w / h);
        camera.aspect = targetAspect;
        renderer.setSize(w, h);
      }
    }

    camera.position.set(...calibration.position);
    camera.fov = calibration.fov;
    camera.updateProjectionMatrix();

    if (overlay) {
      // In overlay mode the camera must match the video exactly.
      // Also sync controls.target so damping can't drift the camera back.
      controls.target.set(...calibration.lookAt);
      aimCamera(camera, calibration);
    } else if (container && container.clientWidth > 0 && container.clientHeight > 0) {
      // In side-by-side mode, orbit around the court centre from the
      // calibrated direction, pulled back so the whole court fits the window.
      const dir = new THREE.Vector3(...calibration.position).sub(COURT_VIEW_TARGET);
      fitCameraToCourt(camera, controls, dir, container.clientWidth / container.clientHeight);
    }
  }, [calibration, overlay, overlayVideoAspect]);

  // Reset camera to calibrated (or default) view when resetViewTrigger changes
  useEffect(() => {
    if (resetViewTrigger === undefined || resetViewTrigger === 0) return;
    if (!sceneRef.current) return;
    const { camera, controls } = sceneRef.current;
    const container = containerRef.current;

    if (overlay && calibration) {
      // Overlay must match the video exactly — restore the calibrated pose.
      camera.position.set(...calibration.position);
      camera.fov = calibration.fov;
      camera.updateProjectionMatrix();
      controls.target.set(...calibration.lookAt);
      aimCamera(camera, calibration);
      return;
    }

    if (!container || container.clientWidth <= 0 || container.clientHeight <= 0) return;
    const aspect = container.clientWidth / container.clientHeight;
    const dir = calibration
      ? new THREE.Vector3(...calibration.position).sub(COURT_VIEW_TARGET)
      : new THREE.Vector3(0, 15, 20).sub(COURT_VIEW_TARGET);
    camera.fov = calibration ? calibration.fov : 60;
    fitCameraToCourt(camera, controls, dir, aspect);
    // Make the render loop's aspect cache agree with what we just set.
    lastRenderSizeRef.current.aspect = aspect;
  }, [resetViewTrigger]); // eslint-disable-line react-hooks/exhaustive-deps

  const ballTrajectory = useMemo(() => {
    if (!calibration || !frames || frames.length === 0) return [];
    const aspect = (videoAspect && videoAspect > 0)
      ? videoAspect
      : (overlayVideoAspect && overlayVideoAspect > 0)
        ? overlayVideoAspect
        : 16 / 9;
    // Whole clip, not a trailing window: the full path stays on screen so it
    // can be inspected while scrubbing. The ball marks the current frame.
    const start = 0;
    const end = frames.length - 1;
    const nearestCorners = (frameIdx: number): number[][] | undefined => {
      if (!courtData || courtData.length === 0) return undefined;
      let best = courtData[0];
      for (const entry of courtData) {
        if (entry.frame <= frameIdx) best = entry;
        else break;
      }
      return best.corners;
    };
    const pts: Array<{ frame: number; x: number; y: number; z: number; gx: number; gz: number; method: string }> = [];
    for (let i = start; i <= end; i++) {
      const f = frames[i];
      if (!f) continue;
      // Prefer physics-solver 3D position when available
      if (f.ball_3d_pos) {
        const [bx, by, bz] = f.ball_3d_pos;
        if (inRenderableCourtVolume(bx, by, bz)) {
          pts.push({ frame: i, x: bx, y: by, z: bz, gx: bx, gz: bz, method: f.ball_3d_method ?? "unknown" });
          continue;
        }
      }
      // In physics-reprojection mode, never fall back to per-frame estimation.
      // Frames not covered by the arc solver are intentionally absent from the trail.
      if (trajectoryMode === "physics-reprojection") continue;
      const estimate = estimateBallWorldPosition(f, calibration, aspect, nearestCorners(i));
      if (!estimate) continue;
      if (!inRenderableCourtVolume(estimate.position[0], estimate.position[1], estimate.position[2])) continue;
      // Skip ground-intersection results — they produce impossible z values for airborne balls.
      if (estimate.method === "ground-intersection") continue;
      pts.push({
        frame: i,
        x: estimate.position[0],
        y: estimate.position[1],
        z: estimate.position[2],
        gx: estimate.ground[0],
        gz: estimate.ground[2],
        method: estimate.method,
      });
    }
    return pts;
  }, [calibration, frames, videoAspect, overlayVideoAspect, courtData, trajectoryMode]);

  useEffect(() => {
    const sceneState = sceneRef.current;
    if (!sceneState) return;
    const { ballMesh, ballShadow, ballStem, ballTrail, estimatedTrail, groundTrail, dropLines } = sceneState;

    if (ballTrajectory.length === 0) {
      ballMesh.visible = false;
      ballShadow.visible = false;
      ballStem.visible = false;
      for (const line of [ballTrail, estimatedTrail, groundTrail, dropLines]) {
        line.geometry.dispose();
        line.geometry = new THREE.BufferGeometry();
      }
      return;
    }

    const cur = ballTrajectory.find((p) => p.frame === currentFrame);
    ballMesh.visible = !!cur;
    ballShadow.visible = !!cur;
    // Over the video the picture already gives depth: no stem, no floor footprint.
    ballStem.visible = !!cur && !overlay;
    if (cur) {
      ballMesh.position.set(cur.x, cur.y, cur.z);
      ballShadow.position.set(cur.gx, 0.005, cur.gz);
      ballStem.geometry.dispose();
      ballStem.geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(cur.gx, 0.01, cur.gz),
        new THREE.Vector3(cur.x, cur.y, cur.z),
      ]);
    }

    // Build segment pairs — skip across frame gaps > 3, 3D jumps > 2 m, or method
    // changes. Estimated stretches join whatever is either side of them.
    const segPoints: THREE.Vector3[] = [];
    const estimatedPoints: THREE.Vector3[] = [];
    const groundPoints: THREE.Vector3[] = [];
    const dropPoints: THREE.Vector3[] = [];
    const FLOOR_Y = 0.01;
    for (let i = 1; i < ballTrajectory.length; i++) {
      const prev = ballTrajectory[i - 1];
      const curr = ballTrajectory[i];
      // Over the video only the ball's recent past is drawn.
      if (overlay && (curr.frame > currentFrame || prev.frame < currentFrame - OVERLAY_TRAIL_FRAMES)) continue;
      const frameDiff = curr.frame - prev.frame;
      const dist = Math.hypot(curr.x - prev.x, curr.y - prev.y, curr.z - prev.z);
      const estimated = prev.method === "estimated" || curr.method === "estimated";
      if (frameDiff <= 3 && (estimated || (dist <= 2 && prev.method === curr.method))) {
        const trail = estimated ? estimatedPoints : segPoints;
        trail.push(new THREE.Vector3(prev.x, prev.y, prev.z));
        trail.push(new THREE.Vector3(curr.x, curr.y, curr.z));
        if (overlay) continue;
        groundPoints.push(new THREE.Vector3(prev.x, FLOOR_Y, prev.z));
        groundPoints.push(new THREE.Vector3(curr.x, FLOOR_Y, curr.z));
      }
    }
    for (const p of ballTrajectory) {
      if (overlay || p.frame % 4 !== 0) continue;
      dropPoints.push(new THREE.Vector3(p.x, FLOOR_Y, p.z), new THREE.Vector3(p.x, p.y, p.z));
    }
    for (const [line, pts] of [[ballTrail, segPoints], [estimatedTrail, estimatedPoints], [groundTrail, groundPoints], [dropLines, dropPoints]] as const) {
      line.geometry.dispose();
      line.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    }
    estimatedTrail.computeLineDistances();
  }, [ballTrajectory, currentFrame, overlay]);

  // Touch markers: a small sphere where the ball was played, and a ring on the
  // floor below it at the player's spot.
  useEffect(() => {
    if (!sceneRef.current) return;
    const { touchMarkers } = sceneRef.current;
    for (const child of [...touchMarkers.children]) {
      touchMarkers.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }
    for (const t of touches ?? []) {
      // Over the video: only the touches around now, and no floor ring.
      if (overlay && Math.abs(t.frame - (currentFrame ?? 0)) > OVERLAY_TOUCH_FRAMES) continue;
      const color = touchColor(t.label);
      const [x, y, z] = t.position;
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 16, 16),
        new THREE.MeshBasicMaterial({ color }),
      );
      dot.position.set(x, y, z);
      touchMarkers.add(dot);
      if (overlay) continue;
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.22, 0.32, 24),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, 0.015, z);
      touchMarkers.add(ring);
    }
  }, [touches, overlay, overlay ? currentFrame : 0]); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: 400 }} />;
}

// Court surface + surroundings. Colours follow the FIVB indoor convention:
// a warm ochre playing court inside a blue free zone, which also makes the
// court boundary readable from any camera angle.
const COLOR_FREE_ZONE = 0x151d27;
const COLOR_COURT     = 0x22384d;
const COLOR_FRONT_ZONE= 0x2b4560;
const COLOR_LINE      = 0xf2f2f2;
const FREE_ZONE       = 3.0;   // FIVB minimum free zone (m)
const LINE_W          = 0.05;  // official line width (m)

export function createCourt(): THREE.Group {
  const group = new THREE.Group();

  const halfL = COURT_HALF_L;  // 9 m
  const halfW = COURT_HALF_W;  // 4.5 m

  // Free zone apron — gives the court a floor to sit on instead of leaving it
  // floating in the void.
  const freeZone = new THREE.Mesh(
    new THREE.PlaneGeometry(COURT_WIDTH + 2 * FREE_ZONE, COURT_LENGTH + 2 * FREE_ZONE),
    new THREE.MeshStandardMaterial({ color: COLOR_FREE_ZONE, roughness: 0.95, metalness: 0.0 })
  );
  freeZone.name = "court-free-zone";
  freeZone.rotation.x = -Math.PI / 2;
  freeZone.position.y = -0.03;
  group.add(freeZone);

  // Playing court
  const surface = new THREE.Mesh(
    new THREE.PlaneGeometry(COURT_WIDTH, COURT_LENGTH),
    new THREE.MeshStandardMaterial({ color: COLOR_COURT, roughness: 0.9, metalness: 0.0 })
  );
  surface.name = "court-surface";
  surface.rotation.x = -Math.PI / 2;
  surface.position.y = -0.015;
  group.add(surface);

  // Front zones (net to attack line) in a lighter shade, as on a real court.
  const frontZoneMat = new THREE.MeshStandardMaterial({
    color: COLOR_FRONT_ZONE, roughness: 0.9, metalness: 0.0,
  });
  for (const sign of [-1, 1]) {
    const fz = new THREE.Mesh(new THREE.PlaneGeometry(COURT_WIDTH, ATTACK_DIST), frontZoneMat.clone());
    fz.name = "court-front-zone";
    fz.rotation.x = -Math.PI / 2;
    fz.position.set(0, -0.01, (sign * ATTACK_DIST) / 2);
    group.add(fz);
  }

  // ── Court lines ────────────────────────────────────────────────────────
  // Drawn as thin quads at true 5 cm width. THREE.Line would render a
  // 1-pixel hairline that aliases badly and ignores perspective.
  const lineMaterial = new THREE.MeshBasicMaterial({
    color: COLOR_LINE,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });

  // Boundary lines are inside the court dimensions, so lay them centred on
  // the nominal boundary.
  addCourtLine(group, lineMaterial, -halfW, -halfL,  halfW, -halfL);  // far baseline
  addCourtLine(group, lineMaterial, -halfW,  halfL,  halfW,  halfL);  // near baseline
  addCourtLine(group, lineMaterial, -halfW, -halfL, -halfW,  halfL);  // left sideline
  addCourtLine(group, lineMaterial,  halfW, -halfL,  halfW,  halfL);  // right sideline
  addCourtLine(group, lineMaterial, -halfW, 0, halfW, 0);             // centre line
  addCourtLine(group, lineMaterial, -halfW, -ATTACK_DIST, halfW, -ATTACK_DIST);
  addCourtLine(group, lineMaterial, -halfW,  ATTACK_DIST, halfW,  ATTACK_DIST);

  // ── Net assembly ───────────────────────────────────────────────────────
  // A real net is 9.5–10 m wide (wider than the 9 m court), hung on a steel
  // cable strung between the poles, with a bottom rope and end lacing tying
  // each side to its pole. Without those the net reads as floating.
  const netDrop = 1.0;
  const netTopY = NET_HEIGHT;
  const netBottomY = NET_HEIGHT - netDrop;
  const netCenterY = (netTopY + netBottomY) / 2;
  const netHalfW = 5.0;                 // 10 m net
  const poleOffset = halfW + 0.9;       // poles 0.9 m outside the sidelines
  const poleHeight = NET_HEIGHT + 0.25;
  const topCableY = netTopY - 0.035;
  const bottomRopeY = netBottomY + 0.025;

  // Netting: a tiled 10 cm square-mesh texture. Mipmapping fades it to a
  // soft grey at distance instead of the shimmering hairlines a
  // LineSegments grid gives.
  const meshTex = makeNetTexture();
  meshTex.repeat.set((2 * netHalfW) / 0.1, netDrop / 0.1);
  const netPanel = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * netHalfW, netDrop - 0.06),
    new THREE.MeshStandardMaterial({
      map: meshTex,
      color: 0xe8e8e8,
      transparent: true,
      alphaTest: 0.05,
      side: THREE.DoubleSide,
      depthWrite: false,
      roughness: 0.9,
    })
  );
  netPanel.name = "net-mesh";
  netPanel.position.set(0, netCenterY - 0.005, 0);
  group.add(netPanel);

  // Top and bottom tape span the full net width.
  const tapeMaterial = new THREE.MeshStandardMaterial({ color: 0xf7f7f7, roughness: 0.7 });
  const topTape = new THREE.Mesh(new THREE.BoxGeometry(2 * netHalfW, 0.07, 0.05), tapeMaterial);
  topTape.position.set(0, topCableY, 0);
  group.add(topTape);

  const bottomTape = new THREE.Mesh(new THREE.BoxGeometry(2 * netHalfW, 0.05, 0.04), tapeMaterial);
  bottomTape.position.set(0, bottomRopeY, 0);
  group.add(bottomTape);

  // Steel top cable running pole to pole through the top tape, and the
  // bottom rope from the net's lower corners to the poles.
  const cableMaterial = new THREE.MeshStandardMaterial({ color: 0x3a3f46, metalness: 0.7, roughness: 0.35 });
  const ropeMaterial = new THREE.MeshStandardMaterial({ color: 0xcfcfcf, roughness: 0.85 });
  const addSpan = (x0: number, x1: number, y: number, r: number, mat: THREE.Material) => {
    const len = Math.abs(x1 - x0);
    const span = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 8), mat);
    span.rotation.z = Math.PI / 2;
    span.position.set((x0 + x1) / 2, y, 0);
    group.add(span);
  };
  addSpan(-poleOffset, poleOffset, topCableY, 0.007, cableMaterial);
  for (const sign of [-1, 1]) {
    addSpan(sign * netHalfW, sign * poleOffset, bottomRopeY, 0.006, ropeMaterial);
  }

  // End rods through the net's side edges, laced to the pole.
  const rodMaterial = new THREE.MeshStandardMaterial({ color: 0x8a6a45, roughness: 0.7 });
  const lacePoints: THREE.Vector3[] = [];
  for (const sign of [-1, 1]) {
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, netDrop, 10), rodMaterial);
    rod.position.set(sign * netHalfW, netCenterY, 0);
    group.add(rod);

    // Zig-zag lacing between rod and pole.
    const laces = 6;
    for (let i = 0; i < laces; i++) {
      const y0 = netBottomY + 0.1 + (i * (netDrop - 0.2)) / laces;
      const y1 = netBottomY + 0.1 + ((i + 1) * (netDrop - 0.2)) / laces;
      lacePoints.push(
        new THREE.Vector3(sign * netHalfW, y0, 0),
        new THREE.Vector3(sign * (poleOffset - 0.05), (y0 + y1) / 2, 0),
        new THREE.Vector3(sign * (poleOffset - 0.05), (y0 + y1) / 2, 0),
        new THREE.Vector3(sign * netHalfW, y1, 0),
      );
    }
  }
  group.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(lacePoints),
    new THREE.LineBasicMaterial({ color: 0xd8d8d8, transparent: true, opacity: 0.8 })
  ));

  // Vertical side bands sit directly above the sidelines and mark the
  // in-bounds width of the net.
  for (const sign of [-1, 1]) {
    const band = new THREE.Mesh(new THREE.BoxGeometry(0.05, netDrop, 0.045), tapeMaterial);
    band.position.set(sign * halfW, netCenterY, 0);
    group.add(band);
  }

  // Antennae: 1.8 m rods on the outer edge of each side band, extending
  // 80 cm above the tape, in the usual 20 cm red/white stripes.
  const antennaRed = new THREE.MeshStandardMaterial({ color: 0xd93025, roughness: 0.6 });
  const antennaWhite = new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.6 });
  for (const sign of [-1, 1]) {
    const baseY = netTopY - 1.0;  // 1.8 m rod, 0.8 m of it above the tape
    for (let i = 0; i < 9; i++) {
      const seg = new THREE.Mesh(
        new THREE.CylinderGeometry(0.005, 0.005, 0.2, 8),
        i % 2 === 0 ? antennaWhite : antennaRed
      );
      seg.position.set(sign * (halfW + 0.03), baseY + 0.1 + i * 0.2, 0);
      group.add(seg);
    }
  }

  // Poles, set back from the sidelines as the rules require, with the
  // padded sleeve used in competition, a cap, a winch for the top cable and
  // a floor plate.
  const poleMaterial = new THREE.MeshStandardMaterial({ color: 0xa3a9b0, metalness: 0.6, roughness: 0.35 });
  const padMaterial = new THREE.MeshStandardMaterial({ color: 0x2f5f9e, roughness: 0.8 });
  const padHeight = 1.9;
  for (const sign of [-1, 1]) {
    const x = sign * poleOffset;

    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, poleHeight, 20), poleMaterial);
    pole.position.set(x, poleHeight / 2, 0);
    group.add(pole);

    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), poleMaterial);
    cap.position.set(x, poleHeight, 0);
    group.add(cap);

    const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, padHeight, 24), padMaterial);
    pad.position.set(x, padHeight / 2 + 0.02, 0);
    group.add(pad);

    // Winch housing on the outside face, where the top cable is tensioned.
    const winch = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.14, 0.1), cableMaterial);
    winch.position.set(x + sign * 0.07, topCableY - 0.2, 0);
    group.add(winch);

    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.22, 0.03, 24), poleMaterial);
    base.position.set(x, 0.0, 0);
    group.add(base);
  }

  return group;
}

/** One 10 cm net cell: a transparent square bordered by a cord, tiled across the net. */
function makeNetTexture(): THREE.Texture {
  const size = 32;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 3;
  ctx.strokeRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Draw a court line of official width as a flat quad on the playing surface. */
function addCourtLine(
  group: THREE.Group,
  material: THREE.Material,
  x1: number, z1: number,
  x2: number, z2: number,
  width: number = LINE_W
) {
  const dx = x2 - x1;
  const dz = z2 - z1;
  const length = Math.hypot(dx, dz);
  if (length === 0) return;

  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(length, width), material);
  mesh.rotation.x = -Math.PI / 2;
  // PlaneGeometry's local +x runs along the segment; spin it flat about Y.
  mesh.rotation.z = -Math.atan2(dz, dx);
  mesh.position.set((x1 + x2) / 2, 0.001, (z1 + z2) / 2);
  group.add(mesh);
}
