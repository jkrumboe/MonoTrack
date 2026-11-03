/**
 * The match at a glance: rallies on a timeline under the video, and a panel
 * with the score, every rally and its touches, and per-team counts. Clicking
 * anything jumps the video there. See utils/matchTimeline.ts for how rallies
 * and points are found.
 */
import { useMemo } from "react";
import { otherSide, SIDES, touchKind, type MatchTimeline, type Rally, type Side } from "../utils/matchTimeline";
import { TOUCH_KINDS, touchColor } from "../utils/rallyEdits";

const SIDE_COLORS: Record<Side, string> = { near: "#4fc3f7", far: "#ff8a65" };
/** Seconds of video shown before a touch picked from a list. */
const TOUCH_LEAD_S = 1;

export const sideName = (teams: Partial<Record<Side, string>> | undefined, s: Side) =>
  teams?.[s]?.trim() || (s === "near" ? "Near side" : "Far side");

function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

const SHORT: Record<string, string> = {
  toss: "T", serve: "Sv", reception: "R", set: "St", attack: "A", block: "B", dig: "D", touch: "?",
};

/* ── Timeline bar ──────────────────────────────────────────────────────── */

interface TimelineBarProps {
  timeline: MatchTimeline;
  totalFrames: number;
  currentFrame: number;
  /** Touch kinds to show; all when empty. */
  kinds: ReadonlySet<string>;
  onSeek: (frame: number) => void;
}

export function TimelineBar({ timeline, totalFrames, currentFrame, kinds, onSeek }: TimelineBarProps) {
  const n = Math.max(1, totalFrames - 1);
  const pct = (f: number) => `${(100 * f) / n}%`;
  return (
    <div
      className="timeline-bar"
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onSeek(Math.round(((e.clientX - r.left) / r.width) * n));
      }}
      title="Rallies (coloured by who won the point) and touches. Click to jump."
    >
      {timeline.rallies.map((r) => (
        <div
          key={r.index}
          className="timeline-rally"
          style={{
            left: pct(r.clipStart),
            width: `max(3px, ${(100 * (r.clipEnd - r.clipStart)) / n}%)`,
            background: r.winner ? SIDE_COLORS[r.winner] : "#666",
          }}
        />
      ))}
      {timeline.rallies.flatMap((r) =>
        r.touches
          .filter((t) => kinds.size === 0 || kinds.has(touchKind(t.label)))
          .map((t) => (
            <div key={`${r.index}-${t.frame}`} className="timeline-touch" style={{ left: pct(t.frame), background: touchColor(t.label) }} />
          )),
      )}
      <div className="timeline-playhead" style={{ left: pct(currentFrame) }} />
    </div>
  );
}

/* ── Match panel ───────────────────────────────────────────────────────── */

interface MatchPanelProps {
  timeline: MatchTimeline;
  fps: number;
  currentFrame: number;
  teams: Partial<Record<Side, string>> | undefined;
  kinds: ReadonlySet<string>;
  onKindsChange: (k: Set<string>) => void;
  ralliesOnly: boolean;
  onRalliesOnlyChange: (v: boolean) => void;
  onSeek: (frame: number) => void;
  /** Set a rally's point by hand; null goes back to the inferred one. */
  onSetWinner: (rallyStartFrame: number, side: Side | null) => void;
  onRenameTeam: (side: Side, name: string) => void;
  /** The 3D ball path is not solved yet (no court calibration, or the solver is running). */
  pending?: string | null;
}

const STAT_ROWS = ["points", "serve", "aces", "service errors", "reception", "set", "attack", "kills", "block", "block points", "dig", "errors"];

const ENDING_TEXT: Record<string, string> = {
  "ace": "ace", "service-error": "service error", "kill": "kill", "block": "block", "error": "error",
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
      <div className="scoreboard">
        {SIDES.map((s) => (
          <div key={s} className="scoreboard-team" style={{ borderColor: SIDE_COLORS[s] }}>
            <input
              className="scoreboard-name"
              value={teams?.[s] ?? ""}
              placeholder={sideName(undefined, s)}
              onChange={(e) => p.onRenameTeam(s, e.target.value)}
              aria-label={`Name of the team on the ${s} side`}
            />
            <div className="scoreboard-score">{timeline.score[s]}</div>
          </div>
        ))}
      </div>
      {p.pending && <div className="match-pending">{p.pending}</div>}

      <label className="match-toggle">
        <input type="checkbox" checked={p.ralliesOnly} onChange={(e) => p.onRalliesOnlyChange(e.target.checked)} />
        Play rallies only (skip the breaks)
      </label>

      <div className="match-section-title">Touches</div>
      <div className="kind-chips">
        {TOUCH_KINDS.filter((k) => k !== "toss").map((k) => (
          <button
            key={k}
            className={`kind-chip ${kinds.has(k) ? "active" : ""}`}
            style={{ borderColor: touchColor(k), background: kinds.has(k) ? touchColor(k) : undefined }}
            onClick={() => toggleKind(k)}
          >
            {k}
          </button>
        ))}
        {kinds.size > 0 && <button className="kind-chip" onClick={() => p.onKindsChange(new Set())}>all</button>}
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
              <span>{touch.label}</span>
              <span className="match-dim">
                {touch.side ? sideName(teams, touch.side) : "?"} · rally {rally.index}
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

      {timeline.rallies.length > 0 && (
        <>
          <div className="match-section-title">Stats</div>
          <table className="match-stats">
            <thead>
              <tr><th />{SIDES.map((s) => <th key={s} style={{ color: SIDE_COLORS[s] }}>{sideName(teams, s)}</th>)}</tr>
            </thead>
            <tbody>
              {STAT_ROWS.map((k) => (
                <tr key={k}><td>{k}</td>{SIDES.map((s) => <td key={s}>{timeline.stats[s][k] ?? 0}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </>
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
  return (
    <div className={`rally-row ${current ? "current" : ""}`}>
      <button className="rally-head" onClick={() => onSeek(r.clipStart)}>
        <span className="rally-index">#{r.index}</span>
        <span className="match-time">{formatClock(r.startFrame / fps)}</span>
        <span className="match-dim">{((r.endFrame - r.startFrame) / fps).toFixed(1)} s</span>
        <span className="rally-score">{r.score.near}–{r.score.far}</span>
      </button>
      <div className="rally-touches">
        {r.touches.filter((t) => touchKind(t.label) !== "toss").map((t) => (
          <button
            key={t.frame}
            className="touch-chip"
            style={{ background: touchColor(t.label), outline: t.side ? `2px solid ${SIDE_COLORS[t.side]}` : undefined }}
            title={`${t.label} · ${t.side ? sideName(teams, t.side) : "side unknown"} · ${formatClock(t.time)}`}
            onClick={() => onSeek(Math.max(0, t.frame - Math.round(TOUCH_LEAD_S * fps)))}
          >
            {SHORT[touchKind(t.label)] ?? t.label}
          </button>
        ))}
      </div>
      <div className="rally-point">
        <span className="match-dim">Point:</span>
        {SIDES.map((s) => (
          <button
            key={s}
            className={`point-btn ${r.winner === s ? "active" : ""} ${r.winner === s && auto ? "auto" : ""}`}
            style={r.winner === s ? { background: SIDE_COLORS[s] } : undefined}
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
          <span className="match-dim">
            {ENDING_TEXT[r.ending]}
            {r.ending === "error" || r.ending === "service-error" ? ` by ${sideName(teams, otherSide(r.winner))}` : ""}
          </span>
        )}
      </div>
    </div>
  );
}
