/**
 * Browser and phone: videos are uploaded to the analysis server
 * (python/server.py) and referenced as "job:<id>". The server keeps every
 * message the analyzer sent, so a finished match opens again without being
 * analysed again.
 */
import type { AnalysisMessage, Backend, CourtDetectResponse, MatchSummary, UploadOptions } from "./backend";

const JOB_PREFIX = "job:";
/** Upload piece size. Small enough that a dropped connection loses little. */
const CHUNK_BYTES = 8 * 1024 * 1024;
const POLL_MS = 1000;
/** Give up on an unreachable server after this long. */
const OFFLINE_GIVE_UP_MS = 60_000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function jobId(video: string): string {
  if (!video.startsWith(JOB_PREFIX)) throw new Error(`Not a server video: ${video}`);
  return encodeURIComponent(video.slice(JOB_PREFIX.length));
}

interface JobMeta extends Omit<MatchSummary, "video"> {
  id: string;
}

/**
 * The server address as typed in the settings, made usable: "192.168.2.101"
 * without a scheme would otherwise be a path on the current page. Bare addresses
 * get http:// and, without a port, :8765. Empty means the server that serves this page.
 */
export function normalizeServerUrl(raw: string): string {
  let s = raw.trim().replace(/\/+$/, "").replace(/\/api$/, "");
  if (s && !/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) {
    // Typed bare ("192.168.2.101"): the server's own default port, unless one is given.
    const host = s.split("/")[0];
    s = `http://${s}`;
    if (!/:\d+$/.test(host)) s = s.replace(host, `${host}:8765`);
  }
  return s;
}

export class ServerBackend implements Backend {
  readonly kind = "server" as const;
  private readonly base: string;

  constructor(baseUrl: string, private readonly token: string) {
    this.base = normalizeServerUrl(baseUrl);
  }

  /** `withToken`: only for URLs the browser loads itself (<video src>), which cannot send a header. */
  private url(path: string, query: Record<string, string | number> = {}, withToken = false) {
    const q = new URLSearchParams(Object.entries(query).map(([k, v]) => [k, String(v)]));
    if (withToken && this.token) q.set("token", this.token);
    const qs = q.toString();
    return `${this.base}/api${path}${qs ? `?${qs}` : ""}`;
  }

  private async request<T>(method: string, path: string, opts: { query?: Record<string, string | number>; body?: BodyInit; json?: unknown } = {}): Promise<T> {
    // A header, not ?token=: query strings end up in server and proxy logs.
    const headers: Record<string, string> = this.token ? { Authorization: `Bearer ${this.token}` } : {};
    let body = opts.body;
    if (opts.json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(opts.json);
    }
    const res = await fetch(this.url(path, opts.query), { method, headers, body });
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const err = new Error(data?.error ?? `${method} ${path}: HTTP ${res.status}`) as Error & { status?: number; data?: unknown };
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data as T;
  }

  private toSummary(m: JobMeta): MatchSummary {
    return { ...m, video: JOB_PREFIX + m.id };
  }

  async listMatches() {
    const jobs = await this.request<JobMeta[]>("GET", "/jobs");
    return jobs.map((j) => this.toSummary(j));
  }

  async deleteMatch(video: string) {
    await this.request("DELETE", `/jobs/${jobId(video)}`);
  }

  async uploadVideo(file: Blob, opts: UploadOptions) {
    let video = opts.resume;
    if (video) {
      // The job may have been deleted on the server since.
      try {
        await this.request("GET", `/jobs/${jobId(video)}`);
      } catch {
        video = undefined;
      }
    }
    if (!video) {
      const job = await this.request<JobMeta>("POST", "/jobs", {
        json: { name: opts.name, size: file.size, source: opts.source ?? "file", mime: file.type },
      });
      video = JOB_PREFIX + job.id;
    }
    opts.onCreated?.(video);
    const id = jobId(video);

    let offset = (await this.request<JobMeta>("GET", `/jobs/${id}`)).uploaded;
    let failures = 0;
    while (offset < file.size) {
      opts.onProgress?.(offset / file.size);
      try {
        const r = await this.request<{ uploaded: number }>("PUT", `/jobs/${id}/video`, {
          query: { offset },
          body: file.slice(offset, Math.min(offset + CHUNK_BYTES, file.size)),
        });
        offset = r.uploaded;
        failures = 0;
      } catch (e) {
        const err = e as Error & { status?: number; data?: { uploaded?: number } };
        // The server has a different amount than we thought: carry on from its count.
        if (err.status === 409 && typeof err.data?.uploaded === "number") {
          offset = err.data.uploaded;
          continue;
        }
        if (++failures > 6) throw err;
        await sleep(1000 * 2 ** failures);
        try {
          offset = (await this.request<JobMeta>("GET", `/jobs/${id}`)).uploaded;
        } catch {
          // Still offline; the next attempt retries.
        }
      }
    }
    opts.onProgress?.(1);
    return video;
  }

  async analyze(video: string, onMessage: (m: AnalysisMessage) => void, opts?: { rerun?: boolean }) {
    const id = jobId(video);
    await this.request("POST", `/jobs/${id}/analyze`, { json: { rerun: !!opts?.rerun } });
    let cancelled = false;
    void (async () => {
      let after = 0;
      let offlineSince: number | null = null;
      while (!cancelled) {
        let page: { events: AnalysisMessage[]; next: number; done: boolean; status: string };
        try {
          page = await this.request("GET", `/jobs/${id}/events`, { query: { after, limit: 5000 } });
          offlineSince = null;
        } catch (e) {
          offlineSince ??= Date.now();
          if (Date.now() - offlineSince > OFFLINE_GIVE_UP_MS) {
            onMessage({ action: "error", message: `Lost the analysis server: ${String(e)}` });
            return;
          }
          await sleep(POLL_MS * 2);
          continue;
        }
        if (cancelled) return;
        for (const m of page.events) onMessage(m);
        after = page.next;
        if (page.done) {
          const ended = page.events.some((m) => m.action === "complete" || m.action === "error");
          if (!ended && page.status === "stopped") onMessage({ action: "error", message: "The analysis was stopped." });
          return;
        }
        if (page.events.length === 0) await sleep(POLL_MS);
      }
    })();
    return () => { cancelled = true; };
  }

  async stopAnalysis(video: string) {
    await this.request("POST", `/jobs/${jobId(video)}/stop`);
  }

  async videoSource(video: string) {
    return { url: this.url(`/jobs/${jobId(video)}/video`, {}, true) };
  }

  detectCourt(video: string, frameTime: number) {
    return this.request<CourtDetectResponse>("POST", `/jobs/${jobId(video)}/detect-court`, { json: { frame_time: frameTime } });
  }

  async saveCourt(video: string, frame: number, corners: number[][]) {
    const r = await this.request<{ saved: string }>("POST", `/jobs/${jobId(video)}/court`, { json: { frame, corners } });
    return r.saved;
  }

  async loadEdits(video: string) {
    const r = await this.request<{ content: string | null }>("GET", `/jobs/${jobId(video)}/edits`);
    return r.content;
  }

  async saveEdits(video: string, content: string) {
    await this.request("PUT", `/jobs/${jobId(video)}/edits`, { body: content });
  }

  async saveTextFile(defaultName: string, content: string) {
    const url = URL.createObjectURL(new Blob([content], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = defaultName;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }
}
