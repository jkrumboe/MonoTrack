import { useState, useEffect, useRef, useCallback } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { useAnalysis } from "./hooks/useAnalysis";
import SkeletonViewer from "./components/SkeletonViewer";
import VideoPlayer from "./components/VideoPlayer";
import CourtCalibration, { type CalibrationResult, solveCameraIterative } from "./components/CourtCalibration";
import ProgressBar from "./components/ProgressBar";
import StatsDisplay from "./components/StatsDisplay";
import PlaybackControls from "./components/PlaybackControls";
import "./App.css";

type ViewMode = "side-by-side" | "overlay";

/**
 * Convert video-frame-normalized corners [0,1] to container NDC [-1,1],
 * accounting for object-fit:contain letterboxing.
 */
function videoToNDC(
  corners: { x: number; y: number }[],
  video: HTMLVideoElement,
  container?: HTMLDivElement | null,
): { x: number; y: number }[] {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  // If no container, use video's native dimensions (no letterboxing)
  const cw = container ? container.clientWidth : vw;
  const ch = container ? container.clientHeight : vh;

  if (!vw || !vh || !cw || !ch) {
    // Fallback: assume no letterboxing
    return corners.map(p => ({ x: p.x * 2 - 1, y: -(p.y * 2 - 1) }));
  }

  const videoAR = vw / vh;
  const containerAR = cw / ch;

  let renderW: number, renderH: number, offsetX: number, offsetY: number;
  if (videoAR > containerAR) {
    renderW = cw;
    renderH = cw / videoAR;
    offsetX = 0;
    offsetY = (ch - renderH) / 2;
  } else {
    renderH = ch;
    renderW = ch * videoAR;
    offsetX = (cw - renderW) / 2;
    offsetY = 0;
  }

  return corners.map(p => {
    const px = offsetX + p.x * renderW;
    const py = offsetY + p.y * renderH;
    return {
      x: (px / cw) * 2 - 1,
      y: -((py / ch) * 2 - 1),
    };
  });
}

function App() {
  const {
    status,
    progress,
    progressMessage,
    frames,
    result,
    error,
    startAnalysis,
    stopAnalysis,
  } = useAnalysis();

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoPath, setVideoPath] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [_videoDuration, setVideoDuration] = useState<number | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isAdjusting, setIsAdjusting] = useState(false);
  const [detectedCorners, setDetectedCorners] = useState<{ x: number; y: number }[] | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const autoCalibrated = useRef(false);
  const playIntervalRef = useRef<number | null>(null);
  const runAutoCalibrationRef = useRef<() => Promise<boolean>>(async () => false);
  const isRealVideo = videoPath !== "" && videoPath !== "demo.mp4";

  // Auto-calibrate when we have a video (works in any view mode)
  const runAutoCalibration = useCallback(async () => {
    if (!videoPath || !isRealVideo) return false;
    const video = videoElRef.current;
    const container = overlayRef.current; // may be null in side-by-side mode
    if (!video || video.readyState < 2) return false;

    let corners = detectedCorners;

    if (!corners) {
      try {
        const result = await invoke<{
          action: string;
          corners?: number[][];
          message?: string;
        }>("detect_court", {
          videoPath,
          frameTime: video.currentTime > 0 ? video.currentTime : 1.0,
        });

        if (result.action === "court_detected" && result.corners) {
          corners = result.corners.map(([x, y]: number[]) => ({ x, y }));
          setDetectedCorners(corners);
        }
      } catch (e) {
        console.warn("[auto-cal] Detection failed:", e);
        return false;
      }
    }

    if (!corners) return false;

    const ndcPoints = videoToNDC(corners, video, container);
    const aspectRatio = container
      ? container.clientWidth / container.clientHeight
      : video.videoWidth / video.videoHeight;
    const cal = solveCameraIterative(ndcPoints, aspectRatio);
    if (cal) {
      console.log("[auto-cal] Success:", cal);
      setCalibration(cal);
      return true;
    }
    return false;
  }, [videoPath, isRealVideo, detectedCorners]);

  // Keep ref always pointing to latest runAutoCalibration
  useEffect(() => {
    runAutoCalibrationRef.current = runAutoCalibration;
  }, [runAutoCalibration]);

  // Store a reference to the video element when it mounts
  const handleVideoElement = useCallback((el: HTMLVideoElement | null) => {
    videoElRef.current = el;
    if (!el || autoCalibrated.current) return;

    const tryNow = async () => {
      if (!autoCalibrated.current && await runAutoCalibrationRef.current()) {
        autoCalibrated.current = true;
      }
    };

    if (el.readyState >= 2) {
      // Video already has data — calibrate now
      setTimeout(tryNow, 50);
    } else {
      // Wait for the video to become ready
      const onReady = () => {
        el.removeEventListener("canplay", onReady);
        setTimeout(tryNow, 50);
      };
      el.addEventListener("canplay", onReady);
    }
  }, []);

  // Try auto-calibration when video frame is ready
  const handleCanPlay = useCallback(() => {
    if (!autoCalibrated.current) {
      setTimeout(async () => {
        if (await runAutoCalibrationRef.current()) {
          autoCalibrated.current = true;
        }
      }, 50);
    }
  }, []);

  // Auto-calibrate as soon as pre-detected corners arrive (any view mode)
  useEffect(() => {
    if (!detectedCorners || autoCalibrated.current) return;
    // Need video ready (container is optional — may not exist in side-by-side)
    const video = videoElRef.current;
    const container = overlayRef.current;
    if (!video || video.readyState < 2) return;

    const ndcPoints = videoToNDC(detectedCorners, video, container);
    const aspectRatio = container
      ? container.clientWidth / container.clientHeight
      : video.videoWidth / video.videoHeight;
    const cal = solveCameraIterative(ndcPoints, aspectRatio);
    if (cal) {
      console.log("[auto-cal] Calibrated from pre-detected corners");
      setCalibration(cal);
      autoCalibrated.current = true;
    }
  }, [detectedCorners]);

  // Auto-calibrate when results are ready — wait for video element to be truly ready
  useEffect(() => {
    if (autoCalibrated.current || status !== "complete" || !isRealVideo) return;

    const tryCalibrate = async () => {
      if (autoCalibrated.current) return;
      if (await runAutoCalibrationRef.current()) {
        autoCalibrated.current = true;
      }
    };

    // The video element may not be mounted yet (React just rendered).
    // Poll briefly for the element, then attach a canplay listener.
    let cancelled = false;
    const pollId = window.setInterval(() => {
      if (cancelled) return;
      const video = videoElRef.current;
      if (!video) return; // element not mounted yet, keep polling

      // Element found — stop polling
      window.clearInterval(pollId);

      if (video.readyState >= 2) {
        // Video already has data — calibrate now
        tryCalibrate();
      } else {
        // Wait for the video to load enough data
        const onReady = () => {
          video.removeEventListener("canplay", onReady);
          if (!cancelled) tryCalibrate();
        };
        video.addEventListener("canplay", onReady);
      }
    }, 100);

    return () => {
      cancelled = true;
      window.clearInterval(pollId);
    };
  }, [status, isRealVideo]);

  // Continuous tracking: re-run detection periodically during playback to keep 3D aligned
  useEffect(() => {
    if (!isRealVideo) return;
    const TRACK_INTERVAL_MS = 5000; // re-detect every 5 seconds

    const intervalId = window.setInterval(async () => {
      const video = videoElRef.current;
      if (!video || video.paused || video.readyState < 2) return;

      try {
        const result = await invoke<{
          action: string;
          corners?: number[][];
          message?: string;
        }>("detect_court", {
          videoPath,
          frameTime: video.currentTime,
        });

        if (result.action === "court_detected" && result.corners) {
          const corners = result.corners.map(([x, y]: number[]) => ({ x, y }));
          const container = overlayRef.current;
          const ndcPoints = videoToNDC(corners, video, container);
          const aspectRatio = container
            ? container.clientWidth / container.clientHeight
            : video.videoWidth / video.videoHeight;
          const cal = solveCameraIterative(ndcPoints, aspectRatio);
          if (cal) {
            setCalibration(cal);
          }
        }
      } catch (e) {
        // keep last good calibration
      }
    }, TRACK_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [viewMode, videoPath, isRealVideo]);

  // Reset autoCalibrated flag when video changes
  useEffect(() => {
    autoCalibrated.current = false;
    setCalibration(null);
    setDetectedCorners(null);
  }, [videoPath]);

  // Pre-detect court corners as soon as a video is selected
  useEffect(() => {
    if (!videoPath || !isRealVideo) return;
    let cancelled = false;

    const timer = setTimeout(async () => {
      try {
        const res = await invoke<{ action: string; corners?: number[][] }>(
          "detect_court",
          { videoPath, frameTime: 1.0 }
        );
        if (!cancelled && res.action === "court_detected" && res.corners) {
          const corners = res.corners.map(([x, y]: number[]) => ({ x, y }));
          setDetectedCorners(corners);
          console.log("[early-detect] Court corners ready");
        }
      } catch {
        // Non-critical — detection will retry when overlay mode is entered
      }
    }, 500);

    return () => { cancelled = true; clearTimeout(timer); };
  }, [videoPath, isRealVideo]);

  const handleSelectVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "avi", "mov", "mkv"] }],
    });

    if (selected) {
      setVideoPath(selected as string);
      // Analysis will start after video metadata is loaded (see handleMetadataLoaded)
      // For now, start immediately — Python will try to detect metadata itself
      startAnalysis(selected as string);
    }
  }, [startAnalysis]);

  const handleDemoMode = useCallback(() => {
    setVideoPath("demo.mp4");
    startAnalysis("demo.mp4");
  }, [startAnalysis]);

  // Video reports its duration when metadata loads
  const handleMetadataLoaded = useCallback((duration: number) => {
    setVideoDuration(duration);
  }, []);

  // Video reports its currentTime during playback — sync 3D frame to it
  const handleVideoTimeUpdate = useCallback(
    (currentTime: number) => {
      if (!result) return;
      const frameIdx = Math.min(
        Math.floor(currentTime * result.fps),
        frames.length - 1
      );
      if (frameIdx >= 0) {
        setCurrentFrame(frameIdx);
      }
    },
    [result, frames.length]
  );

  // Playback loop — only used for demo mode (no real video to drive timing)
  useEffect(() => {
    if (!isPlaying || isRealVideo || frames.length === 0) {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
      return;
    }

    playIntervalRef.current = window.setInterval(() => {
      setCurrentFrame((prev) => {
        if (prev >= frames.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 33); // ~30fps playback

    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, isRealVideo, frames.length]);

  const togglePlay = useCallback(() => {
    if (currentFrame >= frames.length - 1) {
      setCurrentFrame(0);
    }
    setIsPlaying((prev) => !prev);
  }, [currentFrame, frames.length]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Tennis 3D Analyzer</h1>
        <div className="header-actions">
          {status === "idle" && (
            <>
              <button className="btn btn-primary" onClick={handleSelectVideo}>
                Open Video
              </button>
              <button className="btn btn-secondary" onClick={handleDemoMode}>
                Demo Mode
              </button>
            </>
          )}
          {status === "running" && (
            <button className="btn btn-danger" onClick={stopAnalysis}>
              Stop Analysis
            </button>
          )}
          {(status === "complete" || status === "error") && (
            <button className="btn btn-primary" onClick={handleSelectVideo}>
              Analyze Another
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {status === "idle" && (
          <div className="welcome">
            <div className="welcome-icon">🎾</div>
            <h2>Welcome to Tennis 3D Analyzer</h2>
            <p>
              Open a tennis match video to analyze player poses, ball
              trajectory, and match statistics.
            </p>
            <p className="hint">
              Or try <strong>Demo Mode</strong> to see sample data without a
              video file.
            </p>
          </div>
        )}

        {status === "running" && (
          <div className="analysis-running">
            <h2>Analyzing Video...</h2>
            <ProgressBar percent={progress} message={progressMessage} />
            <p className="frame-count">{frames.length} frames processed</p>
          </div>
        )}

        {status === "error" && (
          <div className="error-panel">
            <h2>Analysis Error</h2>
            <p className="error-message">{error}</p>
          </div>
        )}

        {status === "complete" && result && (
          <div className="results-layout">
            <div className="viewer-panel">
              <div className="view-toolbar">
                <button
                  className={`view-btn ${viewMode === "side-by-side" ? "active" : ""}`}
                  onClick={() => setViewMode("side-by-side")}
                >
                  Side by Side
                </button>
                <button
                  className={`view-btn ${viewMode === "overlay" ? "active" : ""}`}
                  onClick={() => setViewMode("overlay")}
                >
                  Overlay
                </button>
                {viewMode === "overlay" && !isCalibrating && !isAdjusting && (
                  <>
                    {calibration && (
                      <button
                        className="view-btn"
                        onClick={() => { setIsCalibrating(false); setIsAdjusting(true); }}
                      >
                        Adjust Corners
                      </button>
                    )}
                    <button
                      className="view-btn"
                      onClick={() => { setIsAdjusting(false); setIsCalibrating(true); }}
                    >
                      {calibration ? "Re-align Court" : "Align Court"}
                    </button>
                  </>
                )}
                {viewMode === "side-by-side" && isRealVideo && calibration && (
                  <span className="view-btn disabled" style={{ opacity: 0.6, cursor: "default" }}>
                    Court Aligned ✓
                  </span>
                )}
              </div>

              {viewMode === "side-by-side" && (
                <div className="split-view">
                  <div className="split-pane">
                    <VideoPlayer
                      videoPath={videoPath}
                      currentFrame={currentFrame}
                      fps={result.fps}
                      isPlaying={isPlaying}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onMetadataLoaded={handleMetadataLoaded}
                      onVideoElement={handleVideoElement}
                      onCanPlay={handleCanPlay}
                    />
                  </div>
                  <div className="split-pane">
                    <SkeletonViewer frames={frames} currentFrame={currentFrame} calibration={calibration} />
                  </div>
                </div>
              )}

              {viewMode === "overlay" && (
                <div className="overlay-view" ref={overlayRef}>
                  <div className="overlay-video">
                    <VideoPlayer
                      videoPath={videoPath}
                      currentFrame={currentFrame}
                      fps={result.fps}
                      isPlaying={isPlaying}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onMetadataLoaded={handleMetadataLoaded}
                      onVideoElement={handleVideoElement}
                      onCanPlay={handleCanPlay}
                    />
                  </div>
                  <div className="overlay-3d" style={(isCalibrating || isAdjusting) ? { display: "none" } : undefined}>
                    <SkeletonViewer
                      frames={frames}
                      currentFrame={currentFrame}
                      overlay
                      calibration={calibration}
                    />
                  </div>
                  {isCalibrating && (
                    <CourtCalibration
                      width={overlayRef.current?.clientWidth ?? 800}
                      height={overlayRef.current?.clientHeight ?? 600}
                      videoPath={videoPath}
                      onCalibrated={(result) => {
                        setCalibration(result);
                        setIsCalibrating(false);
                      }}
                      onCancel={() => setIsCalibrating(false)}
                    />
                  )}
                  {isAdjusting && calibration && (
                    <CourtCalibration
                      width={overlayRef.current?.clientWidth ?? 800}
                      height={overlayRef.current?.clientHeight ?? 600}
                      videoPath={videoPath}
                      adjustMode
                      initialCalibration={calibration}
                      onCalibrated={(result) => {
                        setCalibration(result);
                        setIsAdjusting(false);
                      }}
                      onCancel={() => setIsAdjusting(false)}
                    />
                  )}
                </div>
              )}

              <PlaybackControls
                currentFrame={currentFrame}
                totalFrames={frames.length}
                isPlaying={isPlaying}
                onFrameChange={setCurrentFrame}
                onTogglePlay={togglePlay}
              />
            </div>
            <div className="sidebar">
              <StatsDisplay result={result} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
