interface ProgressBarProps {
  percent: number;
  message: string;
}

export default function ProgressBar({ percent, message }: ProgressBarProps) {
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
    </div>
  );
}
