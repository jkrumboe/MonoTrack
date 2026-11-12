/**
 * The match at a glance: a scrubber with the rallies under the video, and the
 * score, every rally and its touches in the side panel. Clicking anything
 * jumps the video there. See utils/matchTimeline.ts for how rallies and points
 * are found.
 */
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { otherSide, SIDE_COLORS, SIDES, touchKind, type MatchTimeline, type Rally, type Side } from "../utils/matchTimeline";
import { TOUCH_KINDS, touchColor } from "../utils/rallyEdits";

/** Seconds of video shown before a touch picked from a list. */
const TOUCH_LEAD_S = 1;
/** Ruler spacings in seconds; the first that gives few enough labels is used. */
const RULER_STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const RULER_MAX_LABELS = 9;

export const sideName = (teams: Partial<Record<Side, string>> | undefined, s: Side) =>
  teams?.[s]?.trim() || (s === "near" ? "Near side" : "Far side");

export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

const SHORT: Record<string, string> = {
  toss: "T", serve: "Sv", reception: "R", set: "St", attack: "A", block: "B", dig: "D", touch: "?",
};

/* ── Scrubber ──────────────────────────────────────────────────────────── */

interface ScrubberProps {
  timeline?: MatchTimeline;
  totalFrames: number;
  currentFrame: number;
  /** Shows the time ruler and times in the hover readout. */
  fps?: number;
  /** Touch kinds to show; all when empty. */
  kinds?: ReadonlySet<string>;
  onSeek: (frame: number) => void;
}

/** Playback position over the rallies (coloured by who won the point) and their touches. */
export function Scrubber({ timeline, totalFrames, currentFrame, fps, kinds, onSeek }: ScrubberProps) {
  const n = Math.max(1, totalFrames - 1);
  const pct = (f: number) => `${(100 * f) / n}%`;
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const rallies = useMemo(() => timeline?.rallies ?? [], [timeline]);

  const ticks = useMemo(() => {
    if (!fps) return [];
    const duration = n / fps;
    const step = RULER_STEPS_S.find((s) => duration / s <= RULER_MAX_LABELS) ?? 7200;
    const out: number[] = [];
    for (let t = 0; t <= duration; t += step) out.push(t);
    return out;
  }, [n, fps]);

  const frameAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return Math.round(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * n);
  };

  const hoverText = (f: number) => {
    const parts = [fps ? formatClock(f / fps) : `Frame ${f + 1}`];
    const rally = rallies.find((r) => f >= r.clipStart && f <= r.clipEnd);
    if (rally) parts.push(`Rally ${rally.index}`);
    const near = fps ? fps / 3 : 5;
    const touch = rally?.touches.find((t) => Math.abs(t.frame - f) <= near);
    if (touch) parts.push(touch.label);
    return parts.join(" · ");
  };

  return (
    <div
      className="scrubber"
      role="slider"
      tabIndex={0}
      aria-label="Playback position"
      aria-valuemin={0}
      aria-valuemax={n}
      aria-valuenow={currentFrame}
      aria-valuetext={hoverText(currentFrame)}
      title="Drag to scrub · ←/→ step a frame · Shift for 10"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        onSeek(frameAt(e));
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) onSeek(frameAt(e));
        if (e.pointerType === "mouse") setHoverFrame(frameAt(e));
      }}
      onPointerLeave={() => setHoverFrame(null)}
    >
      <div className="scrubber-ruler">
        {ticks.map((t) => (
          <span key={t} className="scrubber-tick" style={{ left: pct(t * (fps ?? 1)) }}>{formatClock(t)}</span>
        ))}
      </div>
      <div className="scrubber-track">
        {rallies.map((r) => (
          <div
            key={r.index}
            className={`scrubber-rally ${currentFrame >= r.clipStart && currentFrame <= r.clipEnd ? "current" : ""}`}
            style={{
              left: pct(r.clipStart),
              width: `max(3px, ${(100 * (r.clipEnd - r.clipStart)) / n}%)`,
              "--rally": r.winner ? SIDE_COLORS[r.winner] : "var(--text-tertiary)",
            } as CSSProperties}
          />
        ))}
        {rallies.flatMap((r) =>
          r.touches
            .filter((t) => !kinds || kinds.size === 0 || kinds.has(touchKind(t.label)))
            .map((t) => (
              <div key={`${r.index}-${t.frame}`} className="scrubber-touch" style={{ left: pct(t.frame), background: touchColor(t.label) }} />
            )),
        )}
      </div>
      <div className="scrubber-playhead" style={{ left: pct(currentFrame) }} />
      {hoverFrame !== null && (
        <div className="scrubber-hover" style={{ left: `clamp(70px, ${pct(hoverFrame)}, calc(100% - 70px))` }}>
          {hoverText(hoverFrame)}
        </div>
      )}
    </div>
  );
}

/* ── Scoreboard ────────────────────────────────────────────────────────── */

interface ScoreboardProps {
  timeline: MatchTimeline;
  teams: Partial<Record<Side, string>> | undefined;
  onRenameTeam: (side: Side, name: string) => void;
}

export function Scoreboard({ timeline, teams, onRenameTeam }: ScoreboardProps) {
  return (
    <div className="scoreboard">
      {SIDES.map((s) => (
        <div key={s} className={`scoreboard-team ${s}`} style={{ "--side": SIDE_COLORS[s] } as CSSProperties}>
          <input
            className="scoreboard-name"
            value={teams?.[s] ?? ""}
            placeholder={sideName(undefined, s)}
            onChange={(e) => onRenameTeam(s, e.target.value)}
            aria-label={`Name of the team on the ${s} side`}
            title="Click to name this team"
          />
          <div className="scoreboard-score">{timeline.score[s]}</div>
        </div>
      ))}
    </div>
  );
}

/* ── Rallies ───────────────────────────────────────────────────────────── */

interface MatchPanelProps {
  timeline: MatchTimeline;
  fps: number;
  currentFrame: number;
  teams: Partial<Record<Side, string>> | undefined;
  kinds: ReadonlySet<string>;
  onKindsChange: (k: Set<string>) => void;
  onSeek: (frame: number) => void;
  /** Set a rally's point by hand; null goes back to the inferred one. */
  onSetWinner: (rallyStartFrame: number, side: Side | null) => void;
  /** The 3D ball path is not solved yet (no court calibration, or the solver is running). */
  pending?: string | null;
}

const ENDING_TEXT: Record<string, string> = {
  "ace": "Ace", "service-error": "Service error", "kill": "Kill", "block": "Block", "error": "Error",
};

export function MatchPanel(p: MatchPanelProps) {
  const { timeline, fps, teams, kinds } = p;
  const current = timeline.rallies.find((r) => p.currentFrame >= r.clipStart && p.currentFrame <= r.clipEnd);

  // Touches of the chosen kinds across the match, for jumping straight to them.
  const picked = useMemo(
    () => kinds.size === 0
      ? []
      : timeline.rallies.flatMap((r) => r.touches.filter((t) => kinds.has(touchKind(t.label))).map((t) => ({ rally: r, touch: t }))),
    [timeline, kinds],
  );

  const toggleKind = (k: string) => {
    const next = new Set(kinds);
    if (next.has(k)) next.delete(k);
    else next.add(k);
    p.onKindsChange(next);
  };

  return (
    <div className="match-panel">
      {p.pending && <div className="notice">{p.pending}</div>}

      <div className="kind-chips" role="group" aria-label="Show touches of type">
        <button className={`kind-chip ${kinds.size === 0 ? "active" : ""}`} onClick={() => p.onKindsChange(new Set())}>All</button>
        {TOUCH_KINDS.filter((k) => k !== "toss").map((k) => (
          <button key={k} className={`kind-chip ${kinds.has(k) ? "active" : ""}`} onClick={() => toggleKind(k)}>
            <span className="touch-dot" style={{ background: touchColor(k) }} />
            {k}
          </button>
        ))}
      </div>

      {kinds.size > 0 ? (
        <div className="match-list">
          {picked.length === 0 && <div className="match-empty">No {[...kinds].join(" / ")} found.</div>}
          {picked.map(({ rally, touch }) => (
            <button
              key={`${rally.index}-${touch.frame}`}
              className={`match-row ${Math.abs(p.currentFrame - touch.frame) < fps ? "current" : ""}`}
              onClick={() => p.onSeek(Math.max(0, touch.frame - Math.round(TOUCH_LEAD_S * fps)))}
            >
              <span className="match-time">{formatClock(touch.time)}</span>
              <span className="touch-dot" style={{ background: touchColor(touch.label) }} />
              <span className="match-row-label">{touch.label}</span>
              <span className="match-dim">
                {touch.side ? sideName(teams, touch.side) : "?"} · Rally {rally.index}
              </span>
            </button>
          ))}
        </div>
      ) : (
        <div className="match-list">
          {timeline.rallies.length === 0 && (
            <div className="match-empty">No rallies yet. They appear once the ball path is solved.</div>
          )}
          {timeline.rallies.map((r) => (
            <RallyRow
              key={r.index}
              rally={r}
              fps={fps}
              teams={teams}
              current={r === current}
              onSeek={p.onSeek}
              onSetWinner={p.onSetWinner}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function RallyRow({ rally: r, fps, teams, current, onSeek, onSetWinner }: {
  rally: Rally;
  fps: number;
  teams: MatchPanelProps["teams"];
  current: boolean;
  onSeek: (frame: number) => void;
  onSetWinner: MatchPanelProps["onSetWinner"];
}) {
  const auto = r.winnerSource && r.winnerSource !== "manual";
  const rowRef = useRef<HTMLDivElement>(null);

  // Keep the rally being played in view. Only where the panel scrolls by itself:
  // on a phone this would move the whole page away from the video.
  useEffect(() => {
    if (current && window.matchMedia("(min-width: 801px)").matches) rowRef.current?.scrollIntoView({ block: "nearest" });
  }, [current]);

  return (
    <div
      ref={rowRef}
      className={`rally-row ${current ? "current" : ""}`}
      style={{ "--winner": r.winner ? SIDE_COLORS[r.winner] : "var(--border-strong)" } as CSSProperties}
    >
      <button className="rally-head" onClick={() => onSeek(r.clipStart)}>
        <span className="rally-index">Rally {r.index}</span>
        <span className="match-time">{formatClock(r.startFrame / fps)}</span>
        <span className="match-dim">{((r.endFrame - r.startFrame) / fps).toFixed(1)} s</span>
        <span className="rally-score">{r.score.near}–{r.score.far}</span>
      </button>
      <div className="rally-touches">
        {r.touches.filter((t) => touchKind(t.label) !== "toss").map((t) => (
          <button
            key={t.frame}
            className="touch-chip"
            style={{ "--chip": touchColor(t.label), "--side": t.side ? SIDE_COLORS[t.side] : "transparent" } as CSSProperties}
            title={`${t.label} · ${t.side ? sideName(teams, t.side) : "side unknown"} · ${formatClock(t.time)}`}
            onClick={() => onSeek(Math.max(0, t.frame - Math.round(TOUCH_LEAD_S * fps)))}
          >
            {SHORT[touchKind(t.label)] ?? t.label}
          </button>
        ))}
      </div>
      <div className="rally-point">
        <span className="match-dim">Point</span>
        {SIDES.map((s) => (
          <button
            key={s}
            className={`point-btn ${r.winner === s ? "active" : ""} ${r.winner === s && auto ? "auto" : ""}`}
            style={{ "--side": SIDE_COLORS[s] } as CSSProperties}
            onClick={() => onSetWinner(r.startFrame, r.winner === s && r.winnerSource === "manual" ? null : s)}
            title={
              r.winner === s
                ? r.winnerSource === "manual" ? "Set by hand — click to go back to automatic" : `Automatic (${r.winnerSource === "next-serve" ? "they serve next" : "where the ball landed"}) — click to confirm`
                : `Give the point to ${sideName(teams, s)}`
            }
          >
            {sideName(teams, s)}
          </button>
        ))}
        {r.ending && r.winner && (
          <span className="rally-ending match-dim">
            {ENDING_TEXT[r.ending]}
            {r.ending === "error" || r.ending === "service-error" ? ` by ${sideName(teams, otherSide(r.winner))}` : ""}
          </span>
        )}
      </div>
    </div>
  );
}
