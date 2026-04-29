import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import * as THREE from "three";
import { useAnalysis } from "./hooks/useAnalysis";
import SkeletonViewer from "./components/SkeletonViewer";
import VideoPlayer from "./components/VideoPlayer";
import { type CalibrationResult, solveCameraIterative } from "./utils/cameraSolver";
import CourtKeypointOverlay from "./components/CourtKeypointOverlay";
import CourtCalibration from "./components/CourtCalibration";
import CourtMapView from "./components/CourtMapView";
import ProgressBar from "./components/ProgressBar";
import StatsDisplay from "./components/StatsDisplay";
import PlaybackControls from "./components/PlaybackControls";
import "./App.css";

type ViewMode = "side-by-side" | "overlay";

const COURT_KEYPOINTS_3D: [number, number, number][] = [
  [-4.5, 0, -9.0], // far-left corner
  [ 4.5, 0, -9.0], // far-right corner
  [-4.5, 0,  9.0], // near-left corner
  [ 4.5, 0,  9.0], // near-right corner
  [-4.5, 0, -3.0], // far attack left
  [ 4.5, 0, -3.0], // far attack right
  [-4.5, 0,  3.0], // near attack left
  [ 4.5, 0,  3.0], // near attack right
  [-4.5, 0,  0.0], // net left
  [ 4.5, 0,  0.0], // net right
];

/**
 * Convert video-frame-normalized corners [0,1] to container NDC [-1,1],
 * accounting for object-fit:contain letterboxing.
 */
function projectKeypointsFromCalibration(
  calibration: CalibrationResult,
  videoSize: { width: number; height: number },
): (number[] | null)[] {
  const vw = videoSize.width;
  const vh = videoSize.height;
  if (!vw || !vh) return COURT_KEYPOINTS_3D.map(() => null);

  const videoAR = vw / vh;
  const cam = new THREE.PerspectiveCamera(calibration.fov, videoAR, 0.1, 500);
  cam.position.set(...calibration.position);
  cam.lookAt(new THREE.Vector3(...calibration.lookAt));
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();

  return COURT_KEYPOINTS_3D.map(([x, y, z]) => {
    const v = new THREE.Vector3(x, y, z).project(cam);
    // NDC [-1,1] from videoAR camera → video-frame normalized [0,1]
    return [(v.x + 1) / 2, (1 - v.y) / 2];
  });
}

function App() {
  const {
    status,
    progress,
    progressMessage,
    frames,
    result,
    courtData,
    error,
    startAnalysis,
    stopAnalysis,
  } = useAnalysis();

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoPath, setVideoPath] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [_videoDuration, setVideoDuration] = useState<number | null>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [manualCalibrationLocked, setManualCalibrationLocked] = useState(false);
  const [showCalibration, setShowCalibration] = useState(false);
  const [calibrationViewTransform, setCalibrationViewTransform] = useState({ zoom: 1, panX: 0, panY: 0 });
  const [overlayEl, setOverlayEl] = useState<HTMLDivElement | null>(null);
  const [splitVideoEl, setSplitVideoEl] = useState<HTMLDivElement | null>(null);
  const [splitVideoSize, setSplitVideoSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const playIntervalRef = useRef<number | null>(null);
  const prevCalibrationRef = useRef<CalibrationResult | null>(null);
  const lastCalibratedCourtFrameRef = useRef<number | null>(null);

  /** Smooth a new calibration against the previous one to prevent 3D camera jumps. */
  const smoothCalibration = useCallback((newCal: CalibrationResult): CalibrationResult => {
    const prev = prevCalibrationRef.current;
    if (!prev) {
      prevCalibrationRef.current = newCal;
      return newCal;
    }
    const a = 0.35; // blend factor toward new value
    const lerp = (p: number, n: number) => p + a * (n - p);
    const smoothed: CalibrationResult = {
      position: [
        lerp(prev.position[0], newCal.position[0]),
        lerp(prev.position[1], newCal.position[1]),
        lerp(prev.position[2], newCal.position[2]),
      ],
      lookAt: [
        lerp(prev.lookAt[0], newCal.lookAt[0]),
        lerp(prev.lookAt[1], newCal.lookAt[1]),
        lerp(prev.lookAt[2], newCal.lookAt[2]),
      ],
      fov: lerp(prev.fov, newCal.fov),
    };
    prevCalibrationRef.current = smoothed;
    return smoothed;
  }, []);

  const isRealVideo = videoPath !== "" && videoPath !== "demo.mp4";
  const videoAspect = useMemo(() => {
    if (!videoSize.width || !videoSize.height) return undefined;
    return videoSize.width / videoSize.height;
  }, [videoSize]);

  useEffect(() => {
    if (!splitVideoEl) {
      setSplitVideoSize({ width: 0, height: 0 });
      return;
    }
    const update = () => {
      setSplitVideoSize({ width: splitVideoEl.clientWidth, height: splitVideoEl.clientHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(splitVideoEl);
    return () => ro.disconnect();
  }, [splitVideoEl]);

  /** Look up the nearest precomputed court corners for a given frame index. */
  const getCornersForFrame = useCallback((frameIdx: number): number[][] | null => {
    if (courtData.length === 0) return null;
    // Find the closest court data entry at or before this frame
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= frameIdx) {
        best = entry;
      } else {
        break; // courtData is sorted by frame
      }
    }
    return best.corners;
  }, [courtData]);

  /** Look up the nearest precomputed court keypoints for a given frame index. */
  const getKeypointsForFrame = useCallback((frameIdx: number): (number[] | null)[] | null => {
    if (courtData.length === 0) return null;
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= frameIdx) {
        best = entry;
      } else {
        break;
      }
    }
    return best.keypoints ?? null;
  }, [courtData]);

  /** Get the nearest court-data sample at or before the current frame. */
  const currentCourtEntry = useMemo(() => {
    if (courtData.length === 0) return null;
    let best = courtData[0];
    for (const entry of courtData) {
      if (entry.frame <= currentFrame) {
        best = entry;
      } else {
        break;
      }
    }
    return best;
  }, [courtData, currentFrame]);

  const currentKeypoints = (() => {
    if (manualCalibrationLocked && calibration && videoSize.width > 0 && videoSize.height > 0) {
      return projectKeypointsFromCalibration(calibration, videoSize);
    }
    return getKeypointsForFrame(currentFrame);
  })();

  /** Compute calibration from precomputed court corners for the current frame. */
  const updateCalibrationForFrame = useCallback((frameIdx: number) => {
    if (manualCalibrationLocked) return;
    const video = videoElRef.current;
    if (!video || video.readyState < 2) return;

    const corners = getCornersForFrame(frameIdx);
    if (!corners) return;

    const cornerObjs = corners.map(([x, y]: number[]) => ({ x, y }));
    // Court detection returns video-frame-normalised [0,1]; convert to video-frame NDC.
    const ndcPoints = cornerObjs.map(p => ({ x: p.x * 2 - 1, y: -(p.y * 2 - 1) }));
    if (!video.videoWidth || !video.videoHeight) return;
    const aspectRatio = video.videoWidth / video.videoHeight;
    const { result: cal } = solveCameraIterative(ndcPoints, aspectRatio);
    if (cal) {
      setCalibration(smoothCalibration(cal));
    }
  }, [getCornersForFrame, smoothCalibration, manualCalibrationLocked]);

  // Store a reference to the video element when it mounts
  const handleVideoElement = useCallback((el: HTMLVideoElement | null) => {
    videoElRef.current = el;
    if (!el) {
      setVideoSize({ width: 0, height: 0 });
      return;
    }
    setVideoSize({ width: el.videoWidth, height: el.videoHeight });
  }, []);

  const handleCanPlay = useCallback(() => {
    if (manualCalibrationLocked) return;
    // Calibrate from precomputed data if available
    if (courtData.length > 0) {
      updateCalibrationForFrame(currentFrame);
    }
  }, [courtData, currentFrame, updateCalibrationForFrame, manualCalibrationLocked]);

  // When courtData arrives (analysis complete) or video becomes ready, calibrate for frame 0
  useEffect(() => {
    if (courtData.length === 0 || status !== "complete" || !isRealVideo || manualCalibrationLocked) return;

    let cancelled = false;
    const pollId = window.setInterval(() => {
      if (cancelled) return;
      const video = videoElRef.current;
      if (!video) return;

      window.clearInterval(pollId);

      const doCalibrate = () => {
        if (!cancelled) {
          prevCalibrationRef.current = null;
          lastCalibratedCourtFrameRef.current = null;
          updateCalibrationForFrame(0);
        }
      };

      if (video.readyState >= 2) {
        doCalibrate();
      } else {
        const onReady = () => {
          video.removeEventListener("canplay", onReady);
          doCalibrate();
        };
        video.addEventListener("canplay", onReady);
      }
    }, 100);

    return () => {
      cancelled = true;
      window.clearInterval(pollId);
    };
  }, [courtData, status, isRealVideo, updateCalibrationForFrame, manualCalibrationLocked]);

  // Update calibration only when we cross into a new court-data sample frame.
  // This avoids running the expensive camera solver on every playback frame.
  useEffect(() => {
    if (!isRealVideo || manualCalibrationLocked || !currentCourtEntry) return;
    if (lastCalibratedCourtFrameRef.current === currentCourtEntry.frame) return;
    lastCalibratedCourtFrameRef.current = currentCourtEntry.frame;
    updateCalibrationForFrame(currentCourtEntry.frame);
  }, [currentCourtEntry, isRealVideo, manualCalibrationLocked, updateCalibrationForFrame]);

  // Reset calibration when video changes
  useEffect(() => {
    prevCalibrationRef.current = null;
    lastCalibratedCourtFrameRef.current = null;
    setCalibration(null);
    setManualCalibrationLocked(false);
    setCalibrationViewTransform({ zoom: 1, panX: 0, panY: 0 });
  }, [videoPath]);

  useEffect(() => {
    if (!showCalibration) {
      setCalibrationViewTransform({ zoom: 1, panX: 0, panY: 0 });
    }
  }, [showCalibration]);

  const handleSelectVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "avi", "mov", "mkv"] }],
    });

    if (selected) {
      setVideoPath(selected as string);
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
    const video = videoElRef.current;
    if (video) {
      setVideoSize({ width: video.videoWidth, height: video.videoHeight });
    }
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
        <h1>Volleyball 3D Analyzer</h1>
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
            <div className="welcome-icon">�</div>
            <h2>Welcome to Volleyball 3D Analyzer</h2>
            <p>
              Open a volleyball match video to analyze player poses, ball
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
                {isRealVideo && (status === "complete") && (
                  <button
                    className={`view-btn ${showCalibration ? "active" : ""}`}
                    onClick={() => setShowCalibration((v) => !v)}
                  >
                    {showCalibration ? "Close Calibration" : "Align Court"}
                  </button>
                )}
                {viewMode === "side-by-side" && isRealVideo && calibration && !showCalibration && (
                  <span className="view-btn disabled" style={{ opacity: 0.6, cursor: "default" }}>
                    Court Aligned ✓
                  </span>
                )}
              </div>

              {viewMode === "side-by-side" && (
                <div className="split-view">
                  <div className="split-pane" ref={setSplitVideoEl}>
                    <div
                      style={{
                        width: "100%",
                        height: "100%",
                        transformOrigin: "0 0",
                        transform: showCalibration
                          ? `translate(${calibrationViewTransform.panX}px, ${calibrationViewTransform.panY}px) scale(${calibrationViewTransform.zoom})`
                          : undefined,
                      }}
                    >
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
                    {currentKeypoints && !showCalibration && (
                      <CourtKeypointOverlay
                        keypoints={currentKeypoints}
                        videoEl={videoElRef.current}
                        containerEl={splitVideoEl}
                      />
                    )}
                    {showCalibration && splitVideoSize.width > 0 && splitVideoSize.height > 0 && (
                      <CourtCalibration
                        width={splitVideoSize.width}
                        height={splitVideoSize.height}
                        videoPath={videoPath}
                        adjustMode={!!calibration}
                        initialCalibration={calibration ?? undefined}
                        onViewTransformChange={setCalibrationViewTransform}
                        onCalibrated={(cal) => {
                          setCalibration(cal);
                          prevCalibrationRef.current = cal;
                          setManualCalibrationLocked(true);
                          setShowCalibration(false);
                        }}
                        onCancel={() => setShowCalibration(false)}
                      />
                    )}
                  </div>
                  <div className="split-pane">
                    <SkeletonViewer calibration={calibration} courtData={courtData} isRealVideo={isRealVideo} />
                  </div>
                </div>
              )}

              {viewMode === "overlay" && (
                <div className="overlay-view" ref={setOverlayEl}>
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
                  <div className="overlay-3d">
                    <SkeletonViewer
                      overlay
                      calibration={calibration}
                      courtData={courtData}
                      isRealVideo={isRealVideo}
                      overlayVideoAspect={videoAspect}
                    />
                  </div>
                  {currentKeypoints && (
                    <div className="overlay-lines">
                      <CourtKeypointOverlay
                        keypoints={currentKeypoints}
                        videoEl={videoElRef.current}
                        containerEl={overlayEl}
                      />
                    </div>
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
              <CourtMapView
                frames={frames}
                currentFrame={currentFrame}
                courtData={courtData}
                isRealVideo={isRealVideo}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
