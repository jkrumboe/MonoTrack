interface PlaybackControlsProps {
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  /** When given, the counter shows clip time as well as the frame number. */
  fps?: number;
  onFrameChange: (frame: number) => void;
  onTogglePlay: () => void;
}

function formatTime(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

export default function PlaybackControls({
  currentFrame,
  totalFrames,
  isPlaying,
  fps,
  onFrameChange,
  onTogglePlay,
}: PlaybackControlsProps) {
  return (
    <div className="playback-controls">
      <button
        className="play-btn"
        onClick={onTogglePlay}
        title={isPlaying ? "Pause (Space)" : "Play (Space)"}
      >
        {isPlaying ? "⏸" : "▶"}
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, totalFrames - 1)}
        value={currentFrame}
        onChange={(e) => onFrameChange(Number(e.target.value))}
        className="frame-slider"
        title="Drag to scrub · ←/→ step a frame · Shift for 10"
      />
      <span className="frame-counter">
        {fps ? (
          <>
            {formatTime(currentFrame / fps)} / {formatTime(totalFrames / fps)}
            <span className="frame-counter-sub">
              {" "}· f{currentFrame + 1}/{totalFrames}
            </span>
          </>
        ) : (
          <>
            {currentFrame + 1} / {totalFrames}
          </>
        )}
      </span>
    </div>
  );
}
