import { useState, useEffect, useRef, useCallback } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";

interface VideoPlayerProps {
  videoPath: string;
  currentFrame: number;
  fps: number;
  isPlaying: boolean;
  onTimeUpdate?: (currentTime: number) => void;
  onMetadataLoaded?: (duration: number) => void;
  onVideoElement?: (el: HTMLVideoElement | null) => void;
  onCanPlay?: () => void;
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
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const isDemo = videoPath === "demo.mp4";
  const rafRef = useRef<number>(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  // Fetch the video as a blob to avoid cross-origin canvas taint
  useEffect(() => {
    if (isDemo) return;
    let revoke: string | null = null;
    const assetUrl = convertFileSrc(videoPath);
    fetch(assetUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        revoke = url;
        setBlobUrl(url);
      })
      .catch(() => {
        // Fallback to asset URL if blob fetch fails
        setBlobUrl(assetUrl);
      });
    return () => {
      if (revoke) URL.revokeObjectURL(revoke);
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

  // When playing: let video play natively, report currentTime via rAF
  useEffect(() => {
    const video = videoRef.current;
    if (!video || isDemo) return;

    const tick = () => {
      if (!video.paused && onTimeUpdate) {
        onTimeUpdate(video.currentTime);
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    if (isPlaying) {
      video.play().catch(() => {});
      rafRef.current = requestAnimationFrame(tick);
    } else {
      video.pause();
      cancelAnimationFrame(rafRef.current);
    }

    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying, isDemo, onTimeUpdate]);

  // When paused: seek to frame position (for slider scrubbing)
  useEffect(() => {
    const video = videoRef.current;
    if (!video || isDemo || isPlaying) return;

    const targetTime = currentFrame / fps;
    if (Math.abs(video.currentTime - targetTime) > 0.02) {
      video.currentTime = targetTime;
    }
  }, [currentFrame, fps, isDemo, isPlaying]);

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
        preload="auto"
        onLoadedMetadata={handleLoadedMetadata}
        onCanPlay={onCanPlay}
      />
    </div>
  );
}
