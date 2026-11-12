/** The Stats tab: the match in numbers, team against team, and the clip they come from. */
import type { CSSProperties } from "react";
import type { AnalysisComplete } from "../hooks/useAnalysis";
import { SIDE_COLORS, SIDES, type MatchTimeline, type Side } from "../utils/matchTimeline";
import { sideName } from "./MatchTimeline";
import { formatDuration } from "./ProgressBar";

interface StatsDisplayProps {
  result: AnalysisComplete;
  timeline: MatchTimeline;
  teams: Partial<Record<Side, string>> | undefined;
  analysisDurationMs: number | null;
}

const STAT_GROUPS: { title: string; rows: string[] }[] = [
  { title: "Scoring", rows: ["points", "aces", "kills", "block points", "service errors", "errors"] },
  { title: "Actions", rows: ["serve", "reception", "set", "attack", "block", "dig"] },
];

export default function StatsDisplay({ result, timeline, teams, analysisDurationMs }: StatsDisplayProps) {
  const { total_frames, fps } = result;
  const { rallies, stats } = timeline;
  const lengths = rallies.map((r) => (r.endFrame - r.startFrame) / fps);
  const touches = rallies.reduce((n, r) => n + r.touches.length, 0);

  return (
    <div className="stats-panel">
      {rallies.length > 0 ? (
        <>
          <div className="stats-grid">
            <Stat label="Rallies" value={String(rallies.length)} />
            <Stat label="Touches" value={String(touches)} />
            <Stat label="Avg rally" value={`${(lengths.reduce((a, b) => a + b, 0) / lengths.length).toFixed(1)} s`} />
            <Stat label="Longest" value={`${Math.max(...lengths).toFixed(1)} s`} />
          </div>

          <div className="compare-legend">
            {SIDES.map((s) => (
              <span key={s}><i style={{ background: SIDE_COLORS[s] }} />{sideName(teams, s)}</span>
            ))}
          </div>

          {STAT_GROUPS.map((g) => (
            <section key={g.title} className="compare-group">
              <h4 className="match-section-title">{g.title}</h4>
              {g.rows.map((k) => {
                const near = stats.near[k] ?? 0;
                const far = stats.far[k] ?? 0;
                const max = Math.max(1, near, far);
                return (
                  <div key={k} className="compare-row" title={`${k}: ${sideName(teams, "near")} ${near}, ${sideName(teams, "far")} ${far}`}>
                    <span className="compare-value">{near}</span>
                    <span className="compare-label">{k[0].toUpperCase() + k.slice(1)}</span>
                    <span className="compare-value far">{far}</span>
                    <span className="compare-bar near"><i style={{ width: `${(100 * near) / max}%`, "--side": SIDE_COLORS.near } as CSSProperties} /></span>
                    <span className="compare-bar far"><i style={{ width: `${(100 * far) / max}%`, "--side": SIDE_COLORS.far } as CSSProperties} /></span>
                  </div>
                );
              })}
            </section>
          ))}
        </>
      ) : (
        <div className="match-empty">Stats appear once rallies are found.</div>
      )}

      <section className="compare-group">
        <h4 className="match-section-title">Clip</h4>
        <dl className="clip-facts">
          <dt>Duration</dt><dd>{(total_frames / fps).toFixed(1)} s</dd>
          <dt>Frames analysed</dt><dd>{total_frames}</dd>
          <dt>Frame rate</dt><dd>{fps} fps</dd>
          {analysisDurationMs !== null && (
            <>
              <dt>Analysis time</dt>
              <dd title={`${(total_frames / (analysisDurationMs / 1000)).toFixed(1)} frames/s`}>{formatDuration(analysisDurationMs)}</dd>
            </>
          )}
        </dl>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
