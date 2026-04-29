import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { CourtDataEntry } from "../hooks/useAnalysis";
import type { CalibrationResult } from "../utils/cameraSolver";

// FIVB volleyball court dimensions (metres)
const COURT_LENGTH = 18.0;
const COURT_WIDTH  = 9.0;
const NET_HEIGHT   = 2.43;
const ATTACK_DIST  = 3.0;

interface SkeletonViewerProps {
  overlay?: boolean;
  calibration?: CalibrationResult | null;
  courtData?: CourtDataEntry[];
  isRealVideo?: boolean;
  overlayVideoAspect?: number;
}

export default function SkeletonViewer({ overlay = false, calibration, overlayVideoAspect }: SkeletonViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayModeRef = useRef(overlay);
  const overlayAspectRef = useRef<number | undefined>(overlayVideoAspect);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    courtGroup: THREE.Group;
  } | null>(null);
  const lastRenderSizeRef = useRef({ w: 0, h: 0, aspect: 0 });

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

    sceneRef.current = {
      scene, camera, renderer, controls, courtGroup,
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

  // Toggle overlay mode: transparent background + lock/unlock orbit controls
  useEffect(() => {
    if (!sceneRef.current) return;
    const { scene, controls, courtGroup } = sceneRef.current;
    if (overlay) {
      scene.background = null;
      controls.enabled = false;
    } else {
      scene.background = new THREE.Color(0x1a1a2e);
      controls.enabled = true;
    }

    // Make the court floor more transparent for video overlays.
    const surface = courtGroup.getObjectByName("court-surface") as THREE.Mesh | undefined;
    if (surface && surface.material instanceof THREE.MeshStandardMaterial) {
      surface.material.transparent = overlay;
      surface.material.opacity = overlay ? 0.28 : 1.0;
      surface.material.depthWrite = !overlay;
      surface.material.needsUpdate = true;
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
      camera.lookAt(new THREE.Vector3(...calibration.lookAt));
    } else {
      // In side-by-side mode, keep the calibrated viewing direction while
      // still allowing manual orbit around that target.
      controls.target.set(...calibration.lookAt);
      controls.enableDamping = false;
      controls.update();
      controls.enableDamping = true;
    }
  }, [calibration, overlay, overlayVideoAspect]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: 400 }} />;
}

function createCourt(): THREE.Group {
  const group = new THREE.Group();

  // Volleyball court: 18 m long × 9 m wide (FIVB)
  const halfL = COURT_LENGTH / 2;  // 9 m
  const halfW = COURT_WIDTH  / 2;  // 4.5 m

  // Court surface
  const surfaceGeometry = new THREE.PlaneGeometry(COURT_WIDTH, COURT_LENGTH);
  const surfaceMaterial = new THREE.MeshStandardMaterial({ color: 0x1a3d6e, side: THREE.DoubleSide });
  const surface = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
  surface.name = "court-surface";
  surface.rotation.x = -Math.PI / 2;
  surface.position.y = -0.01;
  group.add(surface);

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });

  // Outer boundary
  addLine(group, lineMaterial, -halfW, 0, -halfL,  halfW, 0, -halfL);  // far baseline
  addLine(group, lineMaterial, -halfW, 0,  halfL,  halfW, 0,  halfL);  // near baseline
  addLine(group, lineMaterial, -halfW, 0, -halfL, -halfW, 0,  halfL);  // left sideline
  addLine(group, lineMaterial,  halfW, 0, -halfL,  halfW, 0,  halfL);  // right sideline

  // Centre line (under net)
  addLine(group, lineMaterial, -halfW, 0, 0, halfW, 0, 0);

  // Attack lines (3 m from centre on each side)
  addLine(group, lineMaterial, -halfW, 0, -ATTACK_DIST, halfW, 0, -ATTACK_DIST);  // far
  addLine(group, lineMaterial, -halfW, 0,  ATTACK_DIST, halfW, 0,  ATTACK_DIST);  // near

  // Suspended volleyball-style net and poles.
  const netDrop = 1.0;
  const netTopY = NET_HEIGHT;
  const netBottomY = NET_HEIGHT - netDrop;
  const netCenterY = (netTopY + netBottomY) / 2;

  const netPanelGeometry = new THREE.PlaneGeometry(COURT_WIDTH, netDrop);
  const netPanelMaterial = new THREE.MeshStandardMaterial({
    color: 0xd9d9d9,
    transparent: true,
    opacity: 0.12,
    side: THREE.DoubleSide,
  });
  const netPanel = new THREE.Mesh(netPanelGeometry, netPanelMaterial);
  netPanel.position.set(0, netCenterY, 0);
  group.add(netPanel);

  const tapeMaterial = new THREE.MeshStandardMaterial({ color: 0xf5f5f5 });
  const topTape = new THREE.Mesh(new THREE.BoxGeometry(COURT_WIDTH + 0.16, 0.055, 0.04), tapeMaterial);
  topTape.position.set(0, netTopY, 0);
  group.add(topTape);

  const bottomTape = new THREE.Mesh(new THREE.BoxGeometry(COURT_WIDTH + 0.05, 0.03, 0.03), tapeMaterial);
  bottomTape.position.set(0, netBottomY, 0);
  group.add(bottomTape);

  const gridPoints: THREE.Vector3[] = [];
  const cellW = 0.22;
  const cellH = 0.16;

  for (let x = -halfW; x <= halfW + 1e-6; x += cellW) {
    gridPoints.push(new THREE.Vector3(x, netBottomY, 0), new THREE.Vector3(x, netTopY, 0));
  }
  for (let y = netBottomY; y <= netTopY + 1e-6; y += cellH) {
    gridPoints.push(new THREE.Vector3(-halfW, y, 0), new THREE.Vector3(halfW, y, 0));
  }

  const netGridGeometry = new THREE.BufferGeometry().setFromPoints(gridPoints);
  const netGridMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 });
  const netGrid = new THREE.LineSegments(netGridGeometry, netGridMaterial);
  group.add(netGrid);

  const poleHeight = NET_HEIGHT + 0.5;
  const poleRadius = 0.05;
  const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x2c3e50, metalness: 0.25, roughness: 0.55 });
  const leftPole = new THREE.Mesh(new THREE.CylinderGeometry(poleRadius, poleRadius, poleHeight, 16), poleMaterial);
  leftPole.position.set(-(halfW + 0.18), poleHeight / 2, 0);
  group.add(leftPole);

  const rightPole = new THREE.Mesh(new THREE.CylinderGeometry(poleRadius, poleRadius, poleHeight, 16), poleMaterial);
  rightPole.position.set(halfW + 0.18, poleHeight / 2, 0);
  group.add(rightPole);

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
