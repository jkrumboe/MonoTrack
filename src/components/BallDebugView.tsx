import { useEffect, useRef, useMemo } from "react";
import type { FrameData, CourtDataEntry } from "../hooks/useAnalysis";
import type { CalibrationResult } from "../utils/cameraSolver";
import { estimateBallWorldPosition } from "../utils/ballReconstruction";

interface BallDebugViewProps {
  frames: FrameData[];
  currentFrame: number;
  videoEl: HTMLVideoElement | null;
  calibration?: CalibrationResult | null;
  courtData?: CourtDataEntry[];
  videoAspect?: number;
}

const HISTORY = 300; // how many frames back to summarise

export default function BallDebugView({
  frames,
  currentFrame,
  videoEl,
  calibration,
  courtData,
  videoAspect,
}: Readonly<BallDebugViewProps>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── Draw current frame snapshot with detection box ──────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoEl) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const CW = canvas.width;
    const CH = canvas.height;
    ctx.clearRect(0, 0, CW, CH);

    // Draw the video frame
    ctx.drawImage(videoEl, 0, 0, CW, CH);

    const frame = frames[Math.min(currentFrame, frames.length - 1)];
    if (!frame) return;

    const vw = videoEl.videoWidth || 1;
    const vh = videoEl.videoHeight || 1;

    // Scale from video pixels → canvas pixels
    const scaleX = CW / vw;
    const scaleY = CH / vh;

    if (frame.ball_bbox) {
      const [nx1, ny1, nx2, ny2] = frame.ball_bbox;
      const bx1 = nx1 * vw * scaleX;
      const by1 = ny1 * vh * scaleY;
      const bx2 = nx2 * vw * scaleX;
      const by2 = ny2 * vh * scaleY;

      // Detection box
      ctx.strokeStyle = "#ffb300";
      ctx.lineWidth = 2;
      ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);

      // Corner ticks
      const tick = 8;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      const corners: [number, number, number, number][] = [
        [bx1, by1, 1, 1], [bx2, by1, -1, 1], [bx2, by2, -1, -1], [bx1, by2, 1, -1],
      ];
      for (const [cx, cy, dx, dy] of corners) {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + dx * tick, cy);
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx, cy + dy * tick);
        ctx.stroke();
      }

      // Label
      const label = `${Math.round((frame.ball_conf ?? 0) * 100)}%`;
      ctx.font = "bold 13px monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "#ffb300";
      ctx.fillRect(bx1, by1 - 20, tw + 8, 18);
      ctx.fillStyle = "#000";
      ctx.fillText(label, bx1 + 4, by1 - 5);

      // Ground-shadow indicator: dot at bbox bottom-centre (the ray used by bbox-bottom-ray)
      const shadowX = (bx1 + bx2) * 0.5;
      const shadowY = by2;
      ctx.beginPath();
      ctx.arc(shadowX, shadowY, 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0,229,255,0.9)";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    } else if (frame.ball_img_pos) {
      // Interpolated — draw crosshair only
      const [ncx, ncy] = frame.ball_img_pos;
      const px = ncx * vw * scaleX;
      const py = ncy * vh * scaleY;
      const R = 14;

      ctx.strokeStyle = "#aaaaaa";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(px, py, R, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = "#cccccc";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px - R - 5, py);
      ctx.lineTo(px + R + 5, py);
      ctx.moveTo(px, py - R - 5);
      ctx.lineTo(px, py + R + 5);
      ctx.stroke();

      ctx.font = "11px monospace";
      ctx.fillStyle = "#ccc";
      ctx.fillText("interp", px + R + 4, py + 4);
    } else {
      // No ball at all
      ctx.font = "13px monospace";
      ctx.fillStyle = "#f44";
      ctx.fillText("no detection", 8, 20);
    }
  }, [frames, currentFrame, videoEl]);

  // ── Stats over the last HISTORY frames ──────────────────────────────
  const stats = useMemo(() => {
    const end = Math.min(currentFrame, frames.length - 1);
    const start = Math.max(0, end - HISTORY + 1);
    let detected = 0;
    let interpolated = 0;
    let missing = 0;
    let confSum = 0;
    let confCount = 0;
    for (let i = start; i <= end; i++) {
      const f = frames[i];
      if (!f) continue;
      if (f.ball_img_pos) {
        if (f.ball_interpolated) {
          interpolated++;
        } else {
          detected++;
          confSum += f.ball_conf ?? 0;
          confCount++;
        }
      } else {
        missing++;
      }
    }
    const total = detected + interpolated + missing;
    return { detected, interpolated, missing, total, avgConf: confCount > 0 ? confSum / confCount : 0 };
  }, [frames, currentFrame]);

  const cur = frames[Math.min(currentFrame, frames.length - 1)];
  const ball3D = useMemo(() => {
    if (!cur?.ball_img_pos || !calibration) return null;
    const aspect = videoAspect ?? (videoEl?.videoWidth && videoEl.videoHeight
      ? videoEl.videoWidth / videoEl.videoHeight
      : undefined);
    if (!aspect) return null;

    let nearestCorners: number[][] | undefined;
    if (courtData && courtData.length > 0) {
      let best = courtData[0];
      for (const entry of courtData) {
        if (entry.frame <= currentFrame) best = entry;
        else break;
      }
      nearestCorners = best.corners;
    }

    return estimateBallWorldPosition(cur, calibration, aspect, nearestCorners);
  }, [cur, calibration, courtData, currentFrame, videoAspect, videoEl]);

  useEffect(() => {
    if (!cur) return;
    if (!ball3D) {
      if (cur.ball_img_pos) {
        console.debug("[ball-3d] frame=%d no 3D estimate img=(%.4f, %.4f)", currentFrame, cur.ball_img_pos[0], cur.ball_img_pos[1]);
      }
      return;
    }
    console.debug(
      "[ball-3d] frame=%d method=%s pos=(%.2f, %.2f, %.2f) ground=(%.2f, %.2f, %.2f)",
      currentFrame,
      ball3D.method,
      ball3D.position[0],
      ball3D.position[1],
      ball3D.position[2],
      ball3D.ground[0],
      ball3D.ground[1],
      ball3D.ground[2],
    );
  }, [ball3D, cur, currentFrame]);

  let statusLabel: string;
  let statusClass: string;
  if (cur?.ball_bbox) {
    statusLabel = `DETECTED  ${Math.round((cur.ball_conf ?? 0) * 100)}%`;
    statusClass = "ball-debug-status--detected";
  } else if (cur?.ball_img_pos) {
    statusLabel = "INTERPOLATED";
    statusClass = "ball-debug-status--interp";
  } else {
    statusLabel = "NOT FOUND";
    statusClass = "ball-debug-status--missing";
  }

  return (
    <div className="ball-debug-panel">
      <div className={`ball-debug-status ${statusClass}`}>
        ● {statusLabel}
      </div>

      <canvas
        ref={canvasRef}
        width={320}
        height={180}
        className="ball-debug-canvas"
      />

      {cur?.ball_img_pos && (
        <div className="ball-debug-pos">
          pos ({(cur.ball_img_pos[0] * 100).toFixed(1)}%,{" "}
          {(cur.ball_img_pos[1] * 100).toFixed(1)}%)
          {cur.ball_bbox && (
            <>
              {" "}— size{" "}
              {Math.round((cur.ball_bbox[2] - cur.ball_bbox[0]) * 100)}×
              {Math.round((cur.ball_bbox[3] - cur.ball_bbox[1]) * 100)}px%
            </>
          )}
        </div>
      )}

      <div className="ball-debug-pos">
        {ball3D ? (
          <>
            3D{" "}
            <span className={`ball-method ball-method--${ball3D.method}`}>
              {ball3D.method}
            </span>
            {" "}({ball3D.position[0].toFixed(2)},{" "}
            {ball3D.position[1].toFixed(2)},{" "}
            {ball3D.position[2].toFixed(2)}){" "}
            h={ball3D.height.toFixed(2)}m
          </>
        ) : (
          <>
            {cur?.ball_img_pos
              ? "3D estimate unavailable"
              : "3D estimate unavailable: no ball detection"}
          </>
        )}
      </div>

      <div className="ball-debug-stats">
        <span>Last {stats.total} frames:</span>
        <span className="det-detected">✓ {stats.detected} detected</span>
        <span className="det-interp">~ {stats.interpolated} interpolated</span>
        <span className="det-missing">✗ {stats.missing} missing</span>
        {stats.detected > 0 && (
          <span className="det-conf">avg conf {Math.round(stats.avgConf * 100)}%</span>
        )}
      </div>
    </div>
  );
}
