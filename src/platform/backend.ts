/**
 * Where analysis runs and files live, so one UI serves every platform:
 *
 * - desktop (Tauri on Windows/macOS/Linux): videos are local paths, the
 *   Python analyzer runs on this machine (src-tauri/src/lib.rs).
 * - server (a browser, the phone app, or the desktop app pointed at a
 *   server): videos are uploaded to python/server.py, which analyses them
 *   and keeps them as a match library. A video is then "job:<id>".
 *
 * The analyzer's messages are the same JSON either way (python/analyzer.py).
 */
import type { AnalysisComplete, FrameData, ProgressData } from "../hooks/useAnalysis";

export type AnalysisMessage =
  | FrameData
  | ProgressData
  | AnalysisComplete
  | { action: "error"; message: string };

export interface CourtDetectResponse {
  action: string;
  corners?: number[][];
  steps?: unknown[];
  log?: string[];
  message?: string;
}

/** A video uploaded to the server, analysed or not. */
export interface MatchSummary {
  /** The video reference to open it with ("job:<id>"). */
  video: string;
  name: string;
  created: string;
  size: number;
  uploaded: number;
  status: "uploading" | "queued" | "analyzing" | "complete" | "error" | "stopped";
  progress: number;
  message: string;
}

export interface UploadOptions {
  name: string;
  /** "recording" for videos made in the app: the server re-encodes them to a constant frame rate. */
  source?: "file" | "recording";
  onProgress?: (fraction: number) => void;
  /** An upload started earlier (its video reference), to continue where it stopped. */
  resume?: string;
  /** Called as soon as the server has a place for the video, so the upload can be resumed later. */
  onCreated?: (video: string) => void;
}

export interface Backend {
  readonly kind: "desktop" | "server";
  /** Let the user pick a video on this machine; its reference, or null when cancelled. Desktop only. */
  pickLocalVideo?(): Promise<string | null>;
  /** Upload a video; returns its reference. Server only. */
  uploadVideo?(file: Blob, opts: UploadOptions): Promise<string>;
  listMatches?(): Promise<MatchSummary[]>;
  deleteMatch?(video: string): Promise<void>;
  /**
   * Analyse a video (or replay a finished analysis) and pass every message to
   * `onMessage`. Returns a function that stops listening.
   */
  analyze(video: string, onMessage: (m: AnalysisMessage) => void, opts?: { rerun?: boolean }): Promise<() => void>;
  stopAnalysis(video: string): Promise<void>;
  /** A URL the <video> element can play; `release` frees it. */
  videoSource(video: string): Promise<{ url: string; release?: () => void }>;
  detectCourt(video: string, frameTime: number): Promise<CourtDetectResponse>;
  saveCourt(video: string, frame: number, corners: number[][]): Promise<string>;
  loadEdits(video: string): Promise<string | null>;
  saveEdits(video: string, content: string): Promise<void>;
  /** Offer `content` to the user as a file to save. */
  saveTextFile(defaultName: string, content: string): Promise<void>;
}

const SETTINGS_KEY = "vb.settings";

export interface Settings {
  /** Server base URL; empty means the server that serves this page. */
  serverUrl: string;
  /** Access token, when the server was started with VB_TOKEN. */
  token: string;
  /** Desktop app only: analyse on a server instead of this machine. */
  useServerOnDesktop: boolean;
}

export function loadSettings(): Settings {
  const defaults: Settings = {
    // Not `import.meta.env` directly: the eval scripts compile this without Vite's types.
    serverUrl: (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_URL ?? "",
    token: "",
    useServerOnDesktop: false,
  };
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") };
  } catch {
    return defaults;
  }
}

export function saveSettings(s: Settings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    // Private mode: settings last for this session only.
  }
  backend = null;
}

export const inTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
export const isMobileDevice = () =>
  typeof navigator !== "undefined" && /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
/** The phone/tablet can record: a camera API and a recorder. */
export const canRecord = () =>
  typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined";

let backend: Backend | null = null;

/** The backend for this platform. A Tauri app on a phone cannot run the analyzer, so it uses the server. */
export async function getBackend(): Promise<Backend> {
  if (backend) return backend;
  const s = loadSettings();
  if (inTauri() && !isMobileDevice() && !s.useServerOnDesktop) {
    const { TauriBackend } = await import("./tauriBackend");
    backend = new TauriBackend();
  } else {
    const { ServerBackend } = await import("./serverBackend");
    backend = new ServerBackend(s.serverUrl, s.token);
  }
  return backend;
}

/** The backend kind getBackend() will pick, without loading it. */
export function backendKind(): Backend["kind"] {
  return inTauri() && !isMobileDevice() && !loadSettings().useServerOnDesktop ? "desktop" : "server";
}
