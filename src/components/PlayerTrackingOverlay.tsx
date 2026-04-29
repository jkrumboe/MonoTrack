import { useEffect, useRef } from "react";
import type { FrameData } from "../hooks/useAnalysis";

const TRAIL_LENGTH = 60; // frames to trail

const PLAYER_COLORS = {
  p1: { fill: "#4fc3f7", stroke: "#0288d1" },
  p2: { fill: "#ff8a65", stroke: "#e64a19" },
};

interface PlayerTrackingOverlayProps {
  frames: FrameData[];
  currentFrame: number;
  videoEl: HTMLVideoElement | null;
  containerEl: HTMLDivElement | null;
}

export default function PlayerTrackingOverlay({
  frames,
  currentFrame,
  videoEl,
  containerEl,
}: PlayerTrackingOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoEl || !containerEl || frames.length === 0) return;

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

    const toPixel = (pos: [number, number]): [number, number] => [
      offsetX + pos[0] * renderW,
      offsetY + pos[1] * renderH,
    ];

    const drawPlayer = (
      pos: [number, number],
      trail: [number, number][],
      colors: { fill: string; stroke: string },
      label: string,
    ) => {
      // Draw trail
      for (let i = 0; i < trail.length; i++) {
        const alpha = ((i + 1) / trail.length) * 0.55;
        const radius = 4 + (i / trail.length) * 3;
        const [tx, ty] = toPixel(trail[i]);
        ctx.beginPath();
        ctx.arc(tx, ty, radius, 0, Math.PI * 2);
        ctx.fillStyle = `${colors.fill}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`;
        ctx.fill();
      }

      // Draw current position marker
      const [px, py] = toPixel(pos);
      // Outer ring
      ctx.beginPath();
      ctx.arc(px, py, 13, 0, Math.PI * 2);
      ctx.strokeStyle = colors.stroke;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      // Filled dot
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.fillStyle = colors.fill;
      ctx.fill();
      // Label
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#000";
      ctx.fillText(label, px, py + 4);
    };

    const frameIdx = Math.min(currentFrame, frames.length - 1);
    const trailStart = Math.max(0, frameIdx - TRAIL_LENGTH);

    // Collect trail positions for each player
    const trail1: [number, number][] = [];
    const trail2: [number, number][] = [];
    for (let i = trailStart; i < frameIdx; i++) {
      const f = frames[i];
      if (f.player1_img_pos) trail1.push(f.player1_img_pos);
      if (f.player2_img_pos) trail2.push(f.player2_img_pos);
    }

    const frame = frames[frameIdx];
    if (frame.player1_img_pos) {
      drawPlayer(frame.player1_img_pos, trail1, PLAYER_COLORS.p1, "P1");
    }
    if (frame.player2_img_pos) {
      drawPlayer(frame.player2_img_pos, trail2, PLAYER_COLORS.p2, "P2");
    }
  }, [frames, currentFrame, videoEl, containerEl]);

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
        zIndex: 3,
      }}
    />
  );
}
