/** Desktop: local video files, the analyzer runs on this machine (src-tauri/src/lib.rs). */
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import type { AnalysisMessage, Backend, CourtDetectResponse } from "./backend";

const EVENTS = ["analysis-frame", "analysis-progress", "analysis-complete", "analysis-error"];

export class TauriBackend implements Backend {
  readonly kind = "desktop" as const;

  async pickLocalVideo() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "avi", "mov", "mkv", "webm"] }],
    });
    return typeof selected === "string" ? selected : null;
  }

  async analyze(video: string, onMessage: (m: AnalysisMessage) => void) {
    // Listen before starting, so the first messages are not missed.
    const unlisten = await Promise.all(EVENTS.map((e) => listen<AnalysisMessage>(e, (ev) => onMessage(ev.payload))));
    const stop = () => unlisten.forEach((u) => u());
    try {
      await invoke("start_analysis", { videoPath: video });
    } catch (e) {
      stop();
      throw e;
    }
    return stop;
  }

  async stopAnalysis() {
    await invoke("stop_analysis");
  }

  async videoSource(video: string) {
    // Fetched as a blob so drawing frames to a canvas does not taint it.
    const assetUrl = convertFileSrc(video);
    try {
      const blob = await (await fetch(assetUrl)).blob();
      const url = URL.createObjectURL(blob);
      return { url, release: () => URL.revokeObjectURL(url) };
    } catch {
      return { url: assetUrl };
    }
  }

  detectCourt(video: string, frameTime: number) {
    return invoke<CourtDetectResponse>("detect_court", { videoPath: video, frameTime });
  }

  saveCourt(video: string, frame: number, corners: number[][]) {
    return invoke<string>("save_court", { videoPath: video, frame, corners });
  }

  loadEdits(video: string) {
    return invoke<string | null>("load_edits", { videoPath: video });
  }

  async saveEdits(video: string, content: string) {
    await invoke("save_edits", { videoPath: video, content });
  }

  async saveTextFile(defaultName: string, content: string) {
    const path = await save({ defaultPath: defaultName, filters: [{ name: "JSON", extensions: ["json"] }] });
    if (path) await invoke("write_text_file", { path, content });
  }
}
