import type { AnalysisComplete } from "../hooks/useAnalysis";

interface StatsDisplayProps {
  result: AnalysisComplete;
}

export default function StatsDisplay({ result }: StatsDisplayProps) {
  const { score, stats, total_frames, fps } = result;

  return (
    <div className="stats-panel">
      <h3>Match Summary</h3>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Score</div>
          <div className="stat-value">
            <span className="player1-color">{score.player1}</span>
            {" - "}
            <span className="player2-color">{score.player2}</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Avg Ball Speed</div>
          <div className="stat-value">{stats.avg_ball_speed} km/h</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Total Shots</div>
          <div className="stat-value">{stats.total_shots}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Frames Analyzed</div>
          <div className="stat-value">{total_frames}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Video FPS</div>
          <div className="stat-value">{fps}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Duration</div>
          <div className="stat-value">
            {(total_frames / fps).toFixed(1)}s
          </div>
        </div>
      </div>
    </div>
  );
}
