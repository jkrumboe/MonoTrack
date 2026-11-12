import { useEffect, useReducer, useRef } from "react";

/**
 * Volleyball court keypoints (10 total):
 *   0: far-left corner    1: far-right corner
 *   2: near-left corner   3: near-right corner
 *   4: far attack line left    5: far attack line right
 *   6: near attack line left   7: near attack line right
 *   8: net left                9: net right
 */

/** Lines connecting keypoints (pairs of indices) */
const COURT_LINES: [number, number][] = [
  [0, 1],  // far baseline
  [2, 3],  // near baseline
  [0, 2],  // left sideline
  [1, 3],  // right sideline
  [4, 5],  // far attack line
  [6, 7],  // near attack line
  [8, 9],  // net
];

const DOT_RADIUS = 3;
const DOT_COLOR = "#c6f135";
const LINE_COLOR = "rgba(198, 241, 53, 0.55)";
const NET_COLOR  = "rgba(255, 255, 255, 0.8)";
const LINE_WIDTH = 1.25;

interface CourtKeypointOverlayProps {
  keypoints: (number[] | null)[];
  videoEl: HTMLVideoElement | null;
  containerEl: HTMLDivElement | null;
}

export default function CourtKeypointOverlay({ keypoints, videoEl, containerEl }: CourtKeypointOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [resized, redraw] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    if (!containerEl) return;
    const ro = new ResizeObserver(redraw);
    ro.observe(containerEl);
    return () => ro.disconnect();
  }, [containerEl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoEl || !containerEl) return;

    const cw = containerEl.clientWidth;
    const ch = containerEl.clientHeight;
    canvas.width = cw;
    canvas.height = ch;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, cw, ch);

    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh) return;

    // Compute object-fit:contain mapping
    const videoAR = vw / vh;
    const containerAR = cw / ch;
    let renderW: number, renderH: number, offsetX: number, offsetY: number;
    if (videoAR > containerAR) {
      renderW = cw;
      renderH = cw / videoAR;
      offsetX = 0;
      offsetY = (ch - renderH) / 2;
    } else {
      renderH = ch;
      renderW = ch * videoAR;
      offsetX = (cw - renderW) / 2;
      offsetY = 0;
    }

    // Convert normalized [0,1] keypoint to canvas pixel coords
    const toPixel = (kp: number[]): [number, number] => {
      return [
        offsetX + kp[0] * renderW,
        offsetY + kp[1] * renderH,
      ];
    };

    // Draw filled court polygon using the 4 outer corners (kp 0,1,3,2)
    const outerCorners = [keypoints[0], keypoints[1], keypoints[3], keypoints[2]];
    if (outerCorners.every(Boolean)) {
      ctx.fillStyle = "rgba(198, 241, 53, 0.05)";
      ctx.beginPath();
      const [fx, fy] = toPixel(outerCorners[0]!);
      ctx.moveTo(fx, fy);
      for (const kp of outerCorners.slice(1)) {
        const [px, py] = toPixel(kp!);
        ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
    }

    // Draw lines first (behind dots)
    ctx.lineWidth = LINE_WIDTH;
    for (const [a, b] of COURT_LINES) {
      const kpA = keypoints[a];
      const kpB = keypoints[b];
      if (!kpA || !kpB) continue;
      const [x1, y1] = toPixel(kpA);
      const [x2, y2] = toPixel(kpB);
      // Net line gets a distinct gold colour
      ctx.strokeStyle = (a === 8 && b === 9) ? NET_COLOR : LINE_COLOR;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Draw dots
    for (const kp of keypoints) {
      if (!kp) continue;
      const [px, py] = toPixel(kp);
      ctx.beginPath();
      ctx.arc(px, py, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = DOT_COLOR;
      ctx.fill();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [keypoints, videoEl, containerEl, resized]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 2,
      }}
    />
  );
}
