import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { FrameData } from "../hooks/useAnalysis";
import type { CalibrationResult } from "./CourtCalibration";

// MediaPipe Pose landmark connections for skeleton rendering
const POSE_CONNECTIONS: [number, number][] = [
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left leg
  [23, 25], [25, 27],
  // Right leg
  [24, 26], [26, 28],
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  // Shoulders to ears
  [9, 10],
  // Hands
  [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22],
  // Feet
  [27, 29], [27, 31],
  [28, 30], [28, 32],
];

interface SkeletonViewerProps {
  frames: FrameData[];
  currentFrame: number;
  overlay?: boolean;
  calibration?: CalibrationResult | null;
}

export default function SkeletonViewer({ frames, currentFrame, overlay = false, calibration }: SkeletonViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    player1Points: THREE.Points;
    player2Points: THREE.Points;
    player1Lines: THREE.LineSegments;
    player2Lines: THREE.LineSegments;
    ballMesh: THREE.Mesh;
    ballTrail: THREE.Line;
    courtGroup: THREE.Group;
  } | null>(null);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 200);
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
    controls.target.set(0, 1, 0);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    // Court
    const courtGroup = createCourt();
    scene.add(courtGroup);

    // Player 1 skeleton (blue)
    const p1Material = new THREE.PointsMaterial({ color: 0x4fc3f7, size: 0.15 });
    const p1Geometry = new THREE.BufferGeometry();
    p1Geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(33 * 3), 3));
    const player1Points = new THREE.Points(p1Geometry, p1Material);
    scene.add(player1Points);

    const p1LineMaterial = new THREE.LineBasicMaterial({ color: 0x4fc3f7, linewidth: 2 });
    const p1LineGeometry = new THREE.BufferGeometry();
    p1LineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(POSE_CONNECTIONS.length * 2 * 3), 3));
    const player1Lines = new THREE.LineSegments(p1LineGeometry, p1LineMaterial);
    scene.add(player1Lines);

    // Player 2 skeleton (orange)
    const p2Material = new THREE.PointsMaterial({ color: 0xff8a65, size: 0.15 });
    const p2Geometry = new THREE.BufferGeometry();
    p2Geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(33 * 3), 3));
    const player2Points = new THREE.Points(p2Geometry, p2Material);
    scene.add(player2Points);

    const p2LineMaterial = new THREE.LineBasicMaterial({ color: 0xff8a65, linewidth: 2 });
    const p2LineGeometry = new THREE.BufferGeometry();
    p2LineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(POSE_CONNECTIONS.length * 2 * 3), 3));
    const player2Lines = new THREE.LineSegments(p2LineGeometry, p2LineMaterial);
    scene.add(player2Lines);

    // Ball
    const ballGeometry = new THREE.SphereGeometry(0.1, 16, 16);
    const ballMaterial = new THREE.MeshStandardMaterial({ color: 0xcddc39, emissive: 0xcddc39, emissiveIntensity: 0.3 });
    const ballMesh = new THREE.Mesh(ballGeometry, ballMaterial);
    scene.add(ballMesh);

    // Ball trail
    const trailMaterial = new THREE.LineBasicMaterial({ color: 0xcddc39, transparent: true, opacity: 0.4 });
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(300 * 3), 3));
    trailGeometry.setDrawRange(0, 0);
    const ballTrail = new THREE.Line(trailGeometry, trailMaterial);
    scene.add(ballTrail);

    sceneRef.current = {
      scene, camera, renderer, controls,
      player1Points, player2Points,
      player1Lines, player2Lines,
      ballMesh, ballTrail, courtGroup,
    };

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  // Toggle overlay mode (transparent background, keep court visible)
  useEffect(() => {
    if (!sceneRef.current) return;
    const { scene } = sceneRef.current;
    if (overlay) {
      scene.background = null;
    } else {
      scene.background = new THREE.Color(0x1a1a2e);
    }
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
        camera.aspect = w / h;
        renderer.setSize(w, h);
      }
    }

    camera.position.set(...calibration.position);
    camera.fov = calibration.fov;
    camera.updateProjectionMatrix();

    controls.target.set(...calibration.lookAt);

    // Disable damping temporarily to prevent residual drift from
    // prior interactions fighting the calibration position
    controls.enableDamping = false;
    controls.update();
    controls.enableDamping = true;
  }, [calibration]);

  // Update skeleton data when frame changes
  useEffect(() => {
    if (!sceneRef.current || frames.length === 0) return;

    const frameIndex = Math.min(currentFrame, frames.length - 1);
    const frame = frames[frameIndex];
    if (!frame) return;

    const { player1Points, player2Points, player1Lines, player2Lines, ballMesh, ballTrail } = sceneRef.current;

    // Update player poses
    updateSkeleton(player1Points, player1Lines, frame.pose_player1);
    updateSkeleton(player2Points, player2Lines, frame.pose_player2);

    // Update ball position
    if (frame.ball_detected) {
      ballMesh.visible = true;
      ballMesh.position.set(frame.ball.x, frame.ball.y, frame.ball.z);
    } else {
      ballMesh.visible = false;
    }

    // Update ball trail (show last 50 frames)
    const trailStart = Math.max(0, frameIndex - 50);
    const trailPositions: number[] = [];
    for (let i = trailStart; i <= frameIndex; i++) {
      const f = frames[i];
      if (f && f.ball_detected) {
        trailPositions.push(f.ball.x, f.ball.y, f.ball.z);
      }
    }
    const trailGeometry = ballTrail.geometry;
    const trailAttr = trailGeometry.getAttribute("position") as THREE.BufferAttribute;
    const trailArray = trailAttr.array as Float32Array;
    trailArray.fill(0);
    for (let i = 0; i < trailPositions.length; i++) {
      trailArray[i] = trailPositions[i];
    }
    trailAttr.needsUpdate = true;
    trailGeometry.setDrawRange(0, trailPositions.length / 3);
  }, [frames, currentFrame]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: 400 }} />;
}

function updateSkeleton(
  points: THREE.Points,
  lines: THREE.LineSegments,
  landmarks: [number, number, number][]
) {
  const posAttr = points.geometry.getAttribute("position") as THREE.BufferAttribute;
  const posArray = posAttr.array as Float32Array;

  for (let i = 0; i < landmarks.length; i++) {
    const [x, y, z] = landmarks[i];
    posArray[i * 3] = x;
    posArray[i * 3 + 1] = y;
    posArray[i * 3 + 2] = z;
  }
  posAttr.needsUpdate = true;

  // Update line segments
  const lineAttr = lines.geometry.getAttribute("position") as THREE.BufferAttribute;
  const lineArray = lineAttr.array as Float32Array;

  for (let i = 0; i < POSE_CONNECTIONS.length; i++) {
    const [a, b] = POSE_CONNECTIONS[i];
    if (a < landmarks.length && b < landmarks.length) {
      lineArray[i * 6] = landmarks[a][0];
      lineArray[i * 6 + 1] = landmarks[a][1];
      lineArray[i * 6 + 2] = landmarks[a][2];
      lineArray[i * 6 + 3] = landmarks[b][0];
      lineArray[i * 6 + 4] = landmarks[b][1];
      lineArray[i * 6 + 5] = landmarks[b][2];
    }
  }
  lineAttr.needsUpdate = true;
}

function createCourt(): THREE.Group {
  const group = new THREE.Group();

  // Tennis court dimensions (meters): 23.77 x 10.97 (doubles), 23.77 x 8.23 (singles)
  const courtLength = 23.77;
  const doublesWidth = 10.97;
  const singlesWidth = 8.23;
  const serviceBoxDepth = 6.4;

  // Court surface
  const surfaceGeometry = new THREE.PlaneGeometry(doublesWidth, courtLength);
  const surfaceMaterial = new THREE.MeshStandardMaterial({ color: 0x2d5a27, side: THREE.DoubleSide });
  const surface = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
  surface.rotation.x = -Math.PI / 2;
  surface.position.y = -0.01;
  group.add(surface);

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  const halfL = courtLength / 2;
  const halfDW = doublesWidth / 2;
  const halfSW = singlesWidth / 2;

  // Outer doubles boundary
  addLine(group, lineMaterial, -halfDW, 0, -halfL, -halfDW, 0, halfL);
  addLine(group, lineMaterial, halfDW, 0, -halfL, halfDW, 0, halfL);
  addLine(group, lineMaterial, -halfDW, 0, -halfL, halfDW, 0, -halfL);
  addLine(group, lineMaterial, -halfDW, 0, halfL, halfDW, 0, halfL);

  // Singles sidelines
  addLine(group, lineMaterial, -halfSW, 0, -halfL, -halfSW, 0, halfL);
  addLine(group, lineMaterial, halfSW, 0, -halfL, halfSW, 0, halfL);

  // Net (center line)
  addLine(group, lineMaterial, -halfDW, 0, 0, halfDW, 0, 0);

  // Service lines
  addLine(group, lineMaterial, -halfSW, 0, -serviceBoxDepth, halfSW, 0, -serviceBoxDepth);
  addLine(group, lineMaterial, -halfSW, 0, serviceBoxDepth, halfSW, 0, serviceBoxDepth);

  // Center service line
  addLine(group, lineMaterial, 0, 0, -serviceBoxDepth, 0, 0, serviceBoxDepth);

  // Net mesh (visual)
  const netGeometry = new THREE.PlaneGeometry(doublesWidth, 1.07);
  const netMaterial = new THREE.MeshStandardMaterial({
    color: 0xcccccc,
    transparent: true,
    opacity: 0.3,
    side: THREE.DoubleSide,
  });
  const net = new THREE.Mesh(netGeometry, netMaterial);
  net.position.set(0, 1.07 / 2, 0);
  group.add(net);

  return group;
}

function addLine(
  group: THREE.Group,
  material: THREE.LineBasicMaterial,
  x1: number, y1: number, z1: number,
  x2: number, y2: number, z2: number
) {
  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(x1, y1, z1),
    new THREE.Vector3(x2, y2, z2),
  ]);
  const line = new THREE.Line(geometry, material);
  group.add(line);
}
