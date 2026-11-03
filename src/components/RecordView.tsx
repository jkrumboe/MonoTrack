/**
 * Record a match with the phone on a tripod, then upload it for analysis.
 *
 * The analysis needs the whole court in view from behind an end line, as high
 * as possible: the court detector looks for all four corners, and the 3D
 * ball solver works from that one view. The framing guide shows where the
 * court should sit; the level shows when the phone is straight.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  addChunk, createRecording, deleteRecording, listRecordings, recordingBlob, requestPersistentStorage,
  storageEstimate, updateRecording, type Recording,
} from "../platform/recordings";
import type { Backend } from "../platform/backend";

/** Seconds per stored piece: what a crash can lose. */
const CHUNK_S = 4;
const MIME_TYPES = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];

interface RecordViewProps {
  backend: Backend | null;
  /** An uploaded recording is ready to analyse. */
  onUploaded: (video: string) => void;
  onClose: () => void;
}

function formatBytes(n: number) {
  if (n > 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  return `${(n / 1e6).toFixed(0)} MB`;
}

function formatDuration(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(Math.floor(s % 60)).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export default function RecordView({ backend, onUploaded, onClose }: RecordViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recRef = useRef<Recording | null>(null);
  const wakeLockRef = useRef<{ release: () => Promise<void> } | null>(null);
  /** Pieces are stored one after another, in order. */
  const writeQueueRef = useRef<Promise<void>>(Promise.resolve());
  const startedRef = useRef(0);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [resolution, setResolution] = useState<string>("");
  const [fps60, setFps60] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [roll, setRoll] = useState<number | null>(null);
  const [portrait, setPortrait] = useState(false);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [uploads, setUploads] = useState<Record<string, number>>({});
  const [uploadErrors, setUploadErrors] = useState<Record<string, string>>({});
  const [storage, setStorage] = useState<{ usage: number; quota: number } | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const refreshList = useCallback(() => {
    listRecordings().then(setRecordings).catch((e) => console.warn("[record] list:", e));
    storageEstimate().then(setStorage);
  }, []);

  // Recordings left "recording" by a crash still hold everything stored up to it.
  useEffect(() => {
    listRecordings()
      .then((all) => Promise.all(all.filter((r) => r.recording).map((r) => updateRecording(r.id, { recording: false }))))
      .finally(refreshList);
    void requestPersistentStorage();
  }, [refreshList]);

  // Camera preview: the back camera, as sharp and fast as the phone allows.
  useEffect(() => {
    if (recording) return;
    let cancelled = false;
    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: fps60 ? 60 : 30 },
      },
    };
    navigator.mediaDevices.getUserMedia(constraints)
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        const s = stream.getVideoTracks()[0]?.getSettings();
        if (s) setResolution(`${s.width}×${s.height} @ ${Math.round(s.frameRate ?? 0)} fps`);
        setCameraError(null);
      })
      .catch((e: Error) => {
        setCameraError(
          window.isSecureContext
            ? `Cannot open the camera: ${e.message}`
            : "The camera only works over https (or on localhost). Open this page through the server's https address.",
        );
      });
    return () => { cancelled = true; };
  }, [fps60, recording]);

  // Stop the camera when leaving.
  useEffect(() => () => {
    recorderRef.current?.state === "recording" && recorderRef.current.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    void wakeLockRef.current?.release();
  }, []);

  // Level: how far the picture is rolled from straight, from gravity in the
  // screen plane. Taken modulo 90 degrees, so it reads the same however the
  // phone is turned.
  useEffect(() => {
    const onMotion = (e: DeviceMotionEvent) => {
      const g = e.accelerationIncludingGravity;
      if (g?.x == null || g.y == null || Math.hypot(g.x, g.y) < 3) return; // lying flat: no roll to show
      const deg = (Math.atan2(g.x, g.y) * 180) / Math.PI;
      setRoll(((((deg + 45) % 90) + 90) % 90) - 45);
    };
    const onResize = () => setPortrait(window.innerHeight > window.innerWidth);
    onResize();
    window.addEventListener("devicemotion", onMotion);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("devicemotion", onMotion);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  useEffect(() => {
    if (!recording) return;
    const id = window.setInterval(() => setElapsed((performance.now() - startedRef.current) / 1000), 500);
    return () => window.clearInterval(id);
  }, [recording]);

  const start = useCallback(async () => {
    const stream = streamRef.current;
    if (!stream) return;
    const mimeType = MIME_TYPES.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
    const name = `Match ${new Date().toLocaleString()}`;
    const rec = await createRecording(name, mimeType || "video/webm");
    recRef.current = rec;
    const recorder = new MediaRecorder(stream, {
      mimeType: mimeType || undefined,
      videoBitsPerSecond: fps60 ? 16_000_000 : 10_000_000,
    });
    recorder.ondataavailable = (e) => {
      if (!e.data.size) return;
      const seconds = (performance.now() - startedRef.current) / 1000;
      writeQueueRef.current = writeQueueRef.current
        .then(() => addChunk(rec, e.data, seconds))
        .catch((err) => setSaveError(`Could not store the recording (storage full?): ${String(err)}`));
    };
    recorder.onstop = () => {
      void writeQueueRef.current.then(() => updateRecording(rec.id, { recording: false })).then(refreshList);
    };
    recorderRef.current = recorder;
    recorder.start(CHUNK_S * 1000);
    startedRef.current = performance.now();
    setElapsed(0);
    setSaveError(null);
    setRecording(true);
    try {
      const wl = (navigator as Navigator & { wakeLock?: { request: (t: "screen") => Promise<{ release: () => Promise<void> }> } }).wakeLock;
      wakeLockRef.current = (await wl?.request("screen")) ?? null;
    } catch {
      // Without a wake lock the screen may dim; recording continues.
    }
  }, [fps60, refreshList]);

  const stop = useCallback(() => {
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(false);
    void wakeLockRef.current?.release();
    wakeLockRef.current = null;
  }, []);

  const upload = useCallback(async (rec: Recording) => {
    if (!backend?.uploadVideo) return;
    setUploadErrors((e) => ({ ...e, [rec.id]: "" }));
    setUploads((u) => ({ ...u, [rec.id]: 0 }));
    try {
      const blob = await recordingBlob(rec);
      const ext = rec.mimeType.includes("mp4") ? "mp4" : "webm";
      const video = await backend.uploadVideo(blob, {
        name: `${rec.name}.${ext}`,
        source: "recording",
        resume: rec.uploadedTo,
        onCreated: (v) => void updateRecording(rec.id, { uploadedTo: v }),
        onProgress: (f) => setUploads((u) => ({ ...u, [rec.id]: f })),
      });
      await updateRecording(rec.id, { uploadedTo: video, uploadDone: true });
      refreshList();
      onUploaded(video);
    } catch (e) {
      setUploadErrors((er) => ({ ...er, [rec.id]: String(e) }));
    } finally {
      setUploads((u) => {
        const { [rec.id]: _, ...rest } = u;
        return rest;
      });
    }
  }, [backend, onUploaded, refreshList]);

  const download = useCallback(async (rec: Recording) => {
    const url = URL.createObjectURL(await recordingBlob(rec));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${rec.name.replace(/[^\w-]+/g, "_")}.${rec.mimeType.includes("mp4") ? "mp4" : "webm"}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }, []);

  const level = roll !== null && Math.abs(roll) < 1.5;

  return (
    <div className="record-view">
      <div className="record-stage">
        <video
          ref={videoRef}
          className="record-preview"
          autoPlay
          muted
          playsInline
          // iOS gives motion data (for the level) only after a tap asks for it.
          onClick={() => {
            const DME = DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> };
            DME.requestPermission?.().catch(() => undefined);
          }}
        />
        {/* Where the court should sit: both end lines inside the frame, the
            far one in the upper third, the near one above the bottom edge. */}
        <svg className="record-guide" viewBox="0 0 160 90" preserveAspectRatio="none" aria-hidden>
          <polygon points="58,22 102,22 140,84 20,84" fill="none" strokeDasharray="3 2" />
          <line x1="51" y1="40" x2="109" y2="40" />
          <line x1="20" y1="84" x2="140" y2="84" />
        </svg>
        {roll !== null && (
          <div className={`record-level ${level ? "ok" : ""}`} style={{ transform: `rotate(${-roll}deg)` }} />
        )}
        <div className="record-hud">
          {recording ? (
            <span className="record-live">● REC {formatDuration(elapsed)}</span>
          ) : (
            <span>{resolution}</span>
          )}
          {roll !== null ? <span>{level ? "Level ✓" : `Tilt ${roll.toFixed(1)}°`}</span> : <span>Tap the picture for the level</span>}
        </div>
        {portrait && !recording && <div className="record-rotate">Turn the phone sideways (landscape).</div>}
        {cameraError && <div className="record-error">{cameraError}</div>}
      </div>

      <div className="record-controls">
        {!recording ? (
          <>
            <button className="btn btn-secondary" onClick={onClose}>Back</button>
            <label className="match-toggle">
              <input type="checkbox" checked={fps60} onChange={(e) => setFps60(e.target.checked)} /> 60 fps
            </label>
            <button className="record-btn" onClick={start} disabled={!!cameraError} aria-label="Start recording" />
          </>
        ) : (
          <button className="record-btn recording" onClick={stop} aria-label="Stop recording" />
        )}
      </div>
      {saveError && <div className="record-error inline">{saveError}</div>}

      {!recording && (
        <div className="record-tips">
          <strong>Set-up:</strong> tripod behind an end line, centred, as high as it goes (2.5 m+). Fit the
          whole court inside the dashed outline, both end lines visible. Lock the tripod — the camera must not
          move. Plug in a power bank for a full match.
        </div>
      )}

      {!recording && recordings.length > 0 && (
        <div className="record-list">
          <div className="match-section-title">
            Recordings on this phone
            {storage && storage.quota > 0 && (
              <span className="match-dim"> · {formatBytes(storage.usage)} used of {formatBytes(storage.quota)}</span>
            )}
          </div>
          {recordings.map((r) => (
            <div key={r.id} className="record-item">
              <div>
                <div>{r.name}</div>
                <div className="match-dim">
                  {formatDuration(r.durationS)} · {formatBytes(r.bytes)}
                  {r.uploadDone ? " · uploaded" : r.uploadedTo ? " · upload unfinished" : ""}
                </div>
                {uploads[r.id] !== undefined && (
                  <div className="progress-bar"><div className="progress-fill" style={{ width: `${uploads[r.id] * 100}%` }} /></div>
                )}
                {uploadErrors[r.id] && <div className="record-error inline">{uploadErrors[r.id]}</div>}
              </div>
              <div className="record-item-actions">
                {r.uploadDone && r.uploadedTo ? (
                  <button className="btn btn-primary" onClick={() => onUploaded(r.uploadedTo!)}>Open</button>
                ) : (
                  <button
                    className="btn btn-primary"
                    disabled={!backend?.uploadVideo || uploads[r.id] !== undefined}
                    onClick={() => void upload(r)}
                  >
                    {r.uploadedTo ? "Resume upload" : "Upload & analyse"}
                  </button>
                )}
                <button className="btn btn-secondary" onClick={() => void download(r)}>Save file</button>
                <button
                  className="btn btn-danger"
                  onClick={() => {
                    if (confirm(`Delete "${r.name}" from this phone?`)) void deleteRecording(r.id).then(refreshList);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
