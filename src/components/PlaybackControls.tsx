interface PlaybackControlsProps {
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  onFrameChange: (frame: number) => void;
  onTogglePlay: () => void;
}

export default function PlaybackControls({
  currentFrame,
  totalFrames,
  isPlaying,
  onFrameChange,
  onTogglePlay,
}: PlaybackControlsProps) {
  return (
    <div className="playback-controls">
      <button className="play-btn" onClick={onTogglePlay}>
        {isPlaying ? "⏸" : "▶"}
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, totalFrames - 1)}
        value={currentFrame}
        onChange={(e) => onFrameChange(Number(e.target.value))}
        className="frame-slider"
      />
      <span className="frame-counter">
        {currentFrame + 1} / {totalFrames}
      </span>
    </div>
  );
}
