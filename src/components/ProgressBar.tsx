interface ProgressBarProps {
  percent: number;
  message: string;
  /** Time since the run started; enables the elapsed / remaining readout. */
  elapsedMs?: number;
}

export function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export default function ProgressBar({ percent, message, elapsedMs }: ProgressBarProps) {
  // Linear extrapolation from progress so far; wait for a few seconds and a
  // little progress before showing it, early estimates swing wildly.
  const remainingMs =
    elapsedMs !== undefined && percent >= 2 && percent < 100 && elapsedMs > 3000
      ? (elapsedMs * (100 - percent)) / percent
      : null;

  return (
    <div className="progress-container">
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        />
      </div>
      <div className="progress-info">
        <span className="progress-percent">{percent}%</span>
        <span className="progress-message">{message}</span>
      </div>
      {elapsedMs !== undefined && (
        <div className="progress-info progress-time">
          <span>Elapsed {formatDuration(elapsedMs)}</span>
          <span>
            {remainingMs !== null ? `~${formatDuration(remainingMs)} remaining` : "Estimating time…"}
          </span>
        </div>
      )}
    </div>
  );
}
