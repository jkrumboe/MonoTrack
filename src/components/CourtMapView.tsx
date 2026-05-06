import { useMemo } from "react";
import type { FrameData, CourtDataEntry } from "../hooks/useAnalysis";
import { imageToCourtHomography, applyHomography } from "../utils/courtHomography";

// FIVB volleyball court dimensions (metres)
const COURT_LENGTH = 18.0;
const COURT_WIDTH  = 9.0;
const ATTACK_DIST  = 3.0;
const NET_Y_NORM   = 0.5;                          // net at midpoint
const ATTACK_NORM  = ATTACK_DIST / COURT_LENGTH;   // ≈0.1667
const HALF_L = COURT_LENGTH / 2;
const HALF_W = COURT_WIDTH  / 2;

const TRAIL_FRAMES = 120;

/** Convert dummy world coords (Three.js space) to [0,1] court-normalised. */
function worldToNorm(x: number, z: number): [number, number] {
  // nx: 0 = left sideline, 1 = right sideline
  const nx = (x + HALF_W) / COURT_WIDTH;
  // ny: 0 = far baseline (z=−HALF_L), 1 = near baseline (z=+HALF_L)
  const ny = (z + HALF_L) / COURT_LENGTH;
  return [nx, ny];
}

interface PlayerPosSource {
  /** Normalised [0,1] image position (from MediaPipe, real video) */
  imgPos?: [number, number];
  /** World-space hip landmark position from skeleton (dummy mode) */
  worldLms?: [number, number, number][];
}

function resolveCourtPos(
  src: PlayerPosSource,
  H: number[][] | null,
): [number, number] | null {
  if (H && src.imgPos) {
    const [cx, cy] = applyHomography(H, src.imgPos[0], src.imgPos[1]);
    if (cx >= -0.1 && cx <= 1.1 && cy >= -0.1 && cy <= 1.1) return [cx, cy];
  }
  if (src.worldLms && src.worldLms.length > 24) {
    // Use hip midpoint X, Z (world metres)
    const lhip = src.worldLms[23];
    const rhip = src.worldLms[24];
    const wx = (lhip[0] + rhip[0]) / 2;
    const wz = (lhip[2] + rhip[2]) / 2;
    return worldToNorm(wx, wz);
  }
  return null;
}

interface CourtMapViewProps {
  frames: FrameData[];
  currentFrame: number;
  courtData: CourtDataEntry[];
  isRealVideo: boolean;
}

// Court lines in normalised [0,1] coordinates [x1,y1,x2,y2]
// y=0 = far baseline, y=1 = near baseline, x=0 = left sideline
const COURT_SVG_LINES = [
  // Baselines
  [0, 0, 1, 0],
  [0, 1, 1, 1],
  // Sidelines
  [0, 0, 0, 1],
  [1, 0, 1, 1],
  // Net (centre line)
  [0, NET_Y_NORM, 1, NET_Y_NORM],
  // Far attack line
  [0, NET_Y_NORM - ATTACK_NORM, 1, NET_Y_NORM - ATTACK_NORM],
  // Near attack line
  [0, NET_Y_NORM + ATTACK_NORM, 1, NET_Y_NORM + ATTACK_NORM],
];

export default function CourtMapView({
  frames,
  currentFrame,
  courtData,
  isRealVideo,
}: CourtMapViewProps) {
  // Build image→court homography from the nearest court detection
  const H = useMemo<number[][] | null>(() => {
    if (!isRealVideo || courtData.length === 0) return null;
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= currentFrame) best = entry;
      else break;
    }
    return imageToCourtHomography(best.corners);
  }, [courtData, currentFrame, isRealVideo]);

  const frameIdx = Math.min(currentFrame, frames.length - 1);
  const trailStart = Math.max(0, frameIdx - TRAIL_FRAMES);

  // Current positions
  const frame = frames[frameIdx];
  const p1Pos = frame
    ? resolveCourtPos(
        { imgPos: frame.player1_img_pos, worldLms: frame.pose_player1 as [number, number, number][] },
        H,
      )
    : null;
  const p2Pos = frame
    ? resolveCourtPos(
        { imgPos: frame.player2_img_pos, worldLms: frame.pose_player2 as [number, number, number][] },
        H,
      )
    : null;

  // Trail positions
  const trail1: [number, number][] = [];
  const trail2: [number, number][] = [];
  for (let i = trailStart; i < frameIdx; i++) {
    const f = frames[i];
    if (!f) continue;
    const pos1 = resolveCourtPos(
      { imgPos: f.player1_img_pos, worldLms: f.pose_player1 as [number, number, number][] },
      H,
    );
    const pos2 = resolveCourtPos(
      { imgPos: f.player2_img_pos, worldLms: f.pose_player2 as [number, number, number][] },
      H,
    );
    if (pos1) trail1.push(pos1);
    if (pos2) trail2.push(pos2);
  }

  // SVG dimensions (portrait court)
  const svgW = 180;
  const svgH = 300;
  const pad = 12;
  const cW = svgW - pad * 2;
  const cH = svgH - pad * 2;

  const toSvg = (nx: number, ny: number): [number, number] => [
    pad + nx * cW,
    pad + ny * cH,
  ];

  const trailPath = (pts: [number, number][], color: string) =>
    pts.map(([nx, ny], i) => {
      const [sx, sy] = toSvg(nx, ny);
      const alpha = ((i + 1) / pts.length) * 0.7;
      const r = 3 + (i / pts.length) * 3;
      return (
        <circle
          key={i}
          cx={sx}
          cy={sy}
          r={r}
          fill={color}
          opacity={alpha}
        />
      );
    });

  return (
    <div className="court-map-panel">
      <h3>Court Map</h3>
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ display: "block", margin: "0 auto" }}
      >
        {/* Court surface */}
        <rect x={pad} y={pad} width={cW} height={cH} fill="#1a3d6e" rx={2} />

        {/* Court lines */}
        {COURT_SVG_LINES.map(([x1, y1, x2, y2], i) => {
          const [sx1, sy1] = toSvg(x1, y1);
          const [sx2, sy2] = toSvg(x2, y2);
          const isNet = y1 === NET_Y_NORM && y2 === NET_Y_NORM;
          const isAttack = y1 === y2 && Math.abs(y1 - NET_Y_NORM) === ATTACK_NORM;
          return (
            <line
              key={i}
              x1={sx1} y1={sy1} x2={sx2} y2={sy2}
              stroke={isNet ? "#ccc" : isAttack ? "#facc15" : "#fff"}
              strokeWidth={isNet ? 1.5 : 1}
              opacity={isNet ? 0.9 : isAttack ? 0.8 : 0.65}
              strokeDasharray={isAttack ? "4 3" : undefined}
            />
          );
        })}

        {/* Trails */}
        {trailPath(trail1, "#4fc3f7")}
        {trailPath(trail2, "#ff8a65")}

        {/* Team 1 all-player dots (real video) */}
        {frame?.team1_img_positions?.map((imgPos, i) => {
          const pos = H ? (() => {
            const [cx, cy] = applyHomography(H, imgPos[0], imgPos[1]);
            if (cx >= -0.1 && cx <= 1.1 && cy >= -0.1 && cy <= 1.1) return [cx, cy] as [number, number];
            return null;
          })() : null;
          if (!pos) return null;
          const [sx, sy] = toSvg(pos[0], pos[1]);
          return <circle key={i} cx={sx} cy={sy} r={6} fill="#4fc3f7" stroke="#0288d1" strokeWidth={1.5} opacity={0.85} />;
        })}
        {/* Team 2 all-player dots (real video) */}
        {frame?.team2_img_positions?.map((imgPos, i) => {
          const pos = H ? (() => {
            const [cx, cy] = applyHomography(H, imgPos[0], imgPos[1]);
            if (cx >= -0.1 && cx <= 1.1 && cy >= -0.1 && cy <= 1.1) return [cx, cy] as [number, number];
            return null;
          })() : null;
          if (!pos) return null;
          const [sx, sy] = toSvg(pos[0], pos[1]);
          return <circle key={i} cx={sx} cy={sy} r={6} fill="#ff8a65" stroke="#e64a19" strokeWidth={1.5} opacity={0.85} />;
        })}

        {/* Fallback: representative player per team (dummy mode or when team arrays absent) */}
        {!frame?.team1_img_positions?.length && p1Pos && (() => {
          const [sx, sy] = toSvg(p1Pos[0], p1Pos[1]);
          return (
            <g>
              <circle cx={sx} cy={sy} r={9} fill="#4fc3f7" stroke="#0288d1" strokeWidth={2} />
              <text x={sx} y={sy + 4} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#000">T1</text>
            </g>
          );
        })()}
        {!frame?.team2_img_positions?.length && p2Pos && (() => {
          const [sx, sy] = toSvg(p2Pos[0], p2Pos[1]);
          return (
            <g>
              <circle cx={sx} cy={sy} r={9} fill="#ff8a65" stroke="#e64a19" strokeWidth={2} />
              <text x={sx} y={sy + 4} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#000">T2</text>
            </g>
          );
        })()}

        {/* Ball trail on court map */}
        {(() => {
          const ballTrailStart = Math.max(0, frameIdx - TRAIL_FRAMES);
          const ballTrailPts: { pos: [number, number]; interpolated: boolean }[] = [];
          for (let i = ballTrailStart; i < frameIdx; i++) {
            const f = frames[i];
            if (!f?.ball_img_pos) continue;
          const ballPos = f.ball_img_pos;
            if (!ballPos) continue;
            const bpos = H ? (() => {
              const [cx, cy] = applyHomography(H, ballPos[0], ballPos[1]);
              if (cx >= -0.15 && cx <= 1.15 && cy >= -0.15 && cy <= 1.15) return [cx, cy] as [number, number];
              return null;
            })() : null;
            if (bpos) ballTrailPts.push({ pos: bpos, interpolated: !!f.ball_interpolated });
          }
          return ballTrailPts.map(({ pos, interpolated }, i) => {
            const [sx, sy] = toSvg(pos[0], pos[1]);
            const alpha = ((i + 1) / ballTrailPts.length) * 0.7;
            return (
              <circle
                key={i}
                cx={sx} cy={sy}
                r={3 + (i / ballTrailPts.length) * 3}
                fill={interpolated ? "#aaa" : "#ffe082"}
                opacity={alpha}
              />
            );
          });
        })()}

        {/* Ball current position */}
        {frame?.ball_img_pos && (() => {
          const ballImgPos = frame.ball_img_pos;
          if (!ballImgPos) return null;
          const ballCourtPos = H ? (() => {
            const [cx, cy] = applyHomography(H, ballImgPos[0], ballImgPos[1]);
            if (cx >= -0.15 && cx <= 1.15 && cy >= -0.15 && cy <= 1.15) return [cx, cy] as [number, number];
            return null;
          })() : null;
          if (!ballCourtPos) return null;
          const [sx, sy] = toSvg(ballCourtPos[0], ballCourtPos[1]);
          const isInterp = !!frame.ball_interpolated;
          return (
            <g>
              <circle cx={sx} cy={sy} r={8} fill={isInterp ? "#ccc" : "#ffe082"}
                stroke={isInterp ? "#888" : "#ffb300"} strokeWidth={1.5}
                strokeDasharray={isInterp ? "3 2" : undefined} opacity={0.9} />
              <text x={sx} y={sy + 3} textAnchor="middle" fontSize={7} fontWeight="bold" fill="#333">B</text>
            </g>
          );
        })()}
      </svg>
      <div className="court-map-legend">
        <span className="legend-p1">Team 1</span>
        <span className="legend-p2">Team 2</span>
      </div>
    </div>
  );
}
