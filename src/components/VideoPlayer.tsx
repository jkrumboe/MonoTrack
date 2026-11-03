import { useState, useEffect, useRef, useCallback } from "react";
import { getBackend } from "../platform/backend";

interface VideoPlayerProps {
  videoPath: string;
  currentFrame: number;
  fps: number;
  isPlaying: boolean;
  onTimeUpdate?: (currentTime: number) => void;
  onMetadataLoaded?: (duration: number) => void;
  onVideoElement?: (el: HTMLVideoElement | null) => void;
  onCanPlay?: () => void;
  /** Called when playback stops on its own (clip ended or play() failed). */
  onPlaybackStopped?: () => void;
}

export default function VideoPlayer({
  videoPath,
  currentFrame,
  fps,
  isPlaying,
  onTimeUpdate,
  onMetadataLoaded,
  onVideoElement,
  onCanPlay,
  onPlaybackStopped,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const isDemo = videoPath === "demo.mp4";
  const rafRef = useRef<number>(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  // A playable URL for the video: a local file (desktop) or the server's copy.
  useEffect(() => {
    if (isDemo) return;
    let cancelled = false;
    let release: (() => void) | undefined;
    getBackend()
      .then((b) => b.videoSource(videoPath))
      .then((src) => {
        if (cancelled) {
          src.release?.();
          return;
        }
        release = src.release;
        setBlobUrl(src.url);
      })
      .catch((e) => console.error("[video] cannot load:", e));
    return () => {
      cancelled = true;
      release?.();
      setBlobUrl(null);
    };
  }, [videoPath, isDemo]);

  // Expose the video element to parent — re-run when blobUrl changes
  // because the <video> element only mounts after blobUrl is set
  useEffect(() => {
    if (onVideoElement) onVideoElement(videoRef.current);
    return () => { if (onVideoElement) onVideoElement(null); };
  }, [onVideoElement, blobUrl]);

  // Report metadata when video loads
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video && onMetadataLoaded) {
      onMetadataLoaded(video.duration);
    }
  }, [onMetadataLoaded]);

  // Frame shown at a given media time, and the time to seek to for a frame.
  // Seeking to the middle of a frame avoids landing on the boundary, which
  // browsers often render as the previous frame.
  const frameAt = useCallback((t: number) => Math.floor(t * fps + 1e-6), [fps]);
  const timeOf = useCallback((f: number) => (f <= 0 ? 0 : (f + 0.5) / fps), [fps]);

  // While playing, the video drives timing; report currentTime every animation frame
  useEffect(() => {
    const video = videoRef.current;
    if (!video || isDemo || !isPlaying) return;

    const tick = () => {
      if (!video.paused && onTimeUpdate) onTimeUpdate(video.currentTime);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying, isDemo, onTimeUpdate, blobUrl]);

  // Scrubbing fires seeks much faster than the decoder can serve them. Setting
  // currentTime mid-seek aborts the previous one, so while dragging the picture
  // would freeze until you let go. Instead keep at most one seek in flight and
  // remember only the latest target; it is applied when the current one lands.
  const pendingSeekRef = useRef<number | null>(null);
  const seekTo = useCallback((t: number) => {
    const video = videoRef.current;
    if (!video) return;
    if (video.seeking) {
      pendingSeekRef.current = t;
      return;
    }
    pendingSeekRef.current = null;
    video.currentTime = t;
  }, []);

  const handleSeeked = useCallback(() => {
    const video = videoRef.current;
    const next = pendingSeekRef.current;
    if (!video || next === null) return;
    pendingSeekRef.current = null;
    if (Math.abs(video.currentTime - next) > 1e-4) video.currentTime = next;
  }, []);

  // Start/stop the video and keep it on currentFrame while paused
  const wasPlayingRef = useRef(false);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || isDemo) return;

    const wasPlaying = wasPlayingRef.current;
    wasPlayingRef.current = isPlaying;
    const videoFrame = frameAt(pendingSeekRef.current ?? video.currentTime);

    if (isPlaying) {
      if (wasPlaying) return; // already running — video owns the clock
      if (videoFrame !== currentFrame || video.ended) {
        pendingSeekRef.current = null;
        video.currentTime = timeOf(currentFrame);
      }
      video.play().catch((err: DOMException) => {
        // AbortError just means a pause() interrupted the play() — expected
        if (err.name !== "AbortError") onPlaybackStopped?.();
      });
      return;
    }

    if (!video.paused) video.pause();

    if (wasPlaying && Math.abs(videoFrame - currentFrame) <= 1) {
      // Just stopped: currentFrame may trail the video by a tick. Keep the
      // video where it is and pull the frame index up to it instead of
      // seeking backwards (which looks like a stutter on pause).
      onTimeUpdate?.(video.currentTime);
      return;
    }

    if (videoFrame !== currentFrame) seekTo(timeOf(currentFrame));
  }, [isPlaying, currentFrame, isDemo, frameAt, timeOf, seekTo, onTimeUpdate, onPlaybackStopped, blobUrl]);

  const handleEnded = useCallback(() => {
    const video = videoRef.current;
    if (video) onTimeUpdate?.(video.currentTime);
    onPlaybackStopped?.();
  }, [onTimeUpdate, onPlaybackStopped]);

  if (isDemo) {
    return (
      <div className="video-placeholder">
        <span className="video-placeholder-text">
          Demo Mode — no video file
        </span>
      </div>
    );
  }

  if (!blobUrl) {
    return (
      <div className="video-container">
        <div className="video-placeholder">
          <span className="video-placeholder-text">Loading video...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="video-container">
      <video
        ref={videoRef}
        src={blobUrl}
        className="video-element"
        muted
        playsInline
        // A server video may come from another origin (the phone app); without
        // CORS mode, drawing its frames to a canvas would taint it.
        crossOrigin={/^https?:/.test(blobUrl) ? "anonymous" : undefined}
        preload="auto"
        onLoadedMetadata={handleLoadedMetadata}
        onCanPlay={onCanPlay}
        onEnded={handleEnded}
        onSeeked={handleSeeked}
      />
    </div>
  );
}
