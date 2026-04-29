import type { AnalysisComplete } from "../hooks/useAnalysis";

interface StatsDisplayProps {
  result: AnalysisComplete;
}

export default function StatsDisplay({ result }: StatsDisplayProps) {
  const { total_frames, fps } = result;

  return (
    <div className="stats-panel">
      <h3>Rally Summary</h3>

      <div className="stats-grid">
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
