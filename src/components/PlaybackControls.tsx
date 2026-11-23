import { nextRallyAfter, rallyAt, type MatchTimeline, type Rally } from "../utils/matchTimeline";
import { IconNextRally, IconPause, IconPlay, IconPrevRally, IconStepBack, IconStepForward } from "./Icons";
import { Scrubber } from "./MatchTimeline";

interface PlaybackControlsProps {
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  /** When given, the counter shows clip time as well as the frame number. */
  fps?: number;
  timeline?: MatchTimeline;
  /** Touch kinds to mark on the scrubber; all when empty. */
  kinds?: ReadonlySet<string>;
  ralliesOnly?: boolean;
  onRalliesOnlyChange?: (v: boolean) => void;
  onFrameChange: (frame: number) => void;
  onTogglePlay: () => void;
}

/** Past this far into a rally, "previous" goes to its start rather than the rally before. */
const RESTART_RALLY_S = 1;

function formatTime(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

/**
 * Playing time: the frames inside rallies, all of them and those before
 * `frame`. Rally clips can overlap at their lead-in and tail, so they are
 * merged first.
 */
function rallyTime(rallies: readonly Rally[], frame: number): { elapsed: number; total: number } {
  let elapsed = 0;
  let total = 0;
  let coveredTo = -1;
  for (const r of rallies) {
    const start = Math.max(r.clipStart, coveredTo + 1);
    if (r.clipEnd < start) continue;
    total += r.clipEnd - start + 1;
    elapsed += Math.max(0, Math.min(r.clipEnd + 1, frame) - start);
    coveredTo = r.clipEnd;
  }
  return { elapsed, total };
}

export default function PlaybackControls({
  currentFrame,
  totalFrames,
  isPlaying,
  fps,
  timeline,
  kinds,
  ralliesOnly,
  onRalliesOnlyChange,
  onFrameChange,
  onTogglePlay,
}: PlaybackControlsProps) {
  const rallies = timeline?.rallies ?? [];
  const current = rallyAt(rallies, currentFrame);
  const next = nextRallyAfter(rallies, currentFrame);
  const restartFrames = RESTART_RALLY_S * (fps ?? 30);
  const prev = [...rallies].reverse().find((r) => r.clipStart < currentFrame - restartFrames);
  const playing = fps && ralliesOnly && rallies.length > 0 ? rallyTime(rallies, currentFrame) : null;

  return (
    <div className="playback-controls">
      <div className="transport">
        <div className="transport-buttons">
          {rallies.length > 0 && (
            <button className="icon-btn" onClick={() => prev && onFrameChange(prev.clipStart)} disabled={!prev} title="Previous rally" aria-label="Previous rally">
              <IconPrevRally />
            </button>
          )}
          <button className="icon-btn" onClick={() => onFrameChange(currentFrame - 1)} title="Back one frame (←)" aria-label="Back one frame">
            <IconStepBack />
          </button>
          <button
            className="play-btn"
            onClick={onTogglePlay}
            title={isPlaying ? "Pause (Space)" : "Play (Space)"}
            aria-label={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? <IconPause size={15} /> : <IconPlay size={15} />}
          </button>
          <button className="icon-btn" onClick={() => onFrameChange(currentFrame + 1)} title="Forward one frame (→)" aria-label="Forward one frame">
            <IconStepForward />
          </button>
          {rallies.length > 0 && (
            <button className="icon-btn" onClick={() => next && onFrameChange(next.clipStart)} disabled={!next} title="Next rally" aria-label="Next rally">
              <IconNextRally />
            </button>
          )}
        </div>

        <span className="frame-counter">
          {fps && playing ? (
            <span title="Playing time: the rallies only, without the breaks between them">
              <b>{formatTime(playing.elapsed / fps)}</b> / {formatTime(playing.total / fps)}
              <span className="frame-counter-sub">
                Video {formatTime(currentFrame / fps)} · Frame {currentFrame + 1}
              </span>
            </span>
          ) : fps ? (
            <>
              <b>{formatTime(currentFrame / fps)}</b> / {formatTime(totalFrames / fps)}
              <span className="frame-counter-sub">Frame {currentFrame + 1}</span>
            </>
          ) : (
            <>
              <b>{currentFrame + 1}</b> / {totalFrames}
            </>
          )}
        </span>

        {rallies.length > 0 && (
          <span className="transport-rally">
            {current ? `Rally ${current.index} of ${rallies.length}` : `${rallies.length} ${rallies.length === 1 ? "rally" : "rallies"}`}
          </span>
        )}

        {rallies.length > 0 && onRalliesOnlyChange && (
          <label className="switch" title="Playback jumps over the breaks between rallies">
            <input type="checkbox" checked={!!ralliesOnly} onChange={(e) => onRalliesOnlyChange(e.target.checked)} />
            <span className="switch-track" />
            Rallies only
          </label>
        )}
      </div>
      <Scrubber
        timeline={timeline}
        totalFrames={totalFrames}
        currentFrame={currentFrame}
        fps={fps}
        kinds={kinds}
        onSeek={onFrameChange}
      />
    </div>
  );
}
