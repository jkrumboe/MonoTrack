/**
 * npm run inspect           # opens http://127.0.0.1:8766
 *
 * Local page for seeing how the ball pipeline handles one clip, stage by
 * stage, over the clip's video: detection, camera motion, calibration, arc
 * fit, touch splits, rally rules. Each run bundles src/eval/inspectRun.ts
 * afresh with esbuild and runs it in a new node process, so solver edits show
 * up on the next Run without restarting. (vite-node takes ~15 s to transform
 * ballReconstruction.ts; esbuild takes well under one.)
 *
 * Local only: listens on 127.0.0.1. Videos come from eval/videos/.
 */
import { spawn, execFile } from "node:child_process";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { build } from "esbuild";

const PORT = 8766;
const SRC = path.resolve(__dirname);
const EVAL_DIR = path.resolve(SRC, "../../eval");
const VIDEO_DIR = path.join(EVAL_DIR, "videos");
const PAGE = path.join(SRC, "inspect.html");
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "rally-inspect-"));
const NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;

/** Bundle inspectRun.ts (with the current solver source) to one file plain node can run. */
async function bundle(): Promise<string> {
  const out = path.join(TMP, "inspectRun.cjs");
  await build({
    entryPoints: [path.join(SRC, "inspectRun.ts")],
    bundle: true, platform: "node", format: "cjs", outfile: out, logLevel: "error",
    // dataset.ts finds eval/ from its own folder; the bundle lives elsewhere.
    define: { __dirname: JSON.stringify(SRC) },
  });
  return out;
}

/** Run one case (or --all) with optional rally-param overrides; resolves to the JSON it wrote. */
async function runCase(target: string, params: unknown): Promise<unknown> {
  const script = await bundle();
  const id = Math.random().toString(36).slice(2, 10);
  const out = path.join(TMP, `${id}.json`);
  const args = [script, out, target];
  if (params && Object.keys(params as object).length) {
    const pf = path.join(TMP, `${id}.params.json`);
    fs.writeFileSync(pf, JSON.stringify(params));
    args.push(pf);
  }
  return new Promise((resolve, reject) => {
    // The solver logs a lot; keep only the tail of stderr for errors.
    let err = "";
    const p = spawn(process.execPath, args, { stdio: ["ignore", "ignore", "pipe"] });
    p.stderr.on("data", (d) => { err = (err + d).slice(-4000); });
    p.on("close", (code) => {
      if (code !== 0 || !fs.existsSync(out)) return reject(new Error(err || `exit ${code}`));
      const data = JSON.parse(fs.readFileSync(out, "utf8"));
      fs.rmSync(out, { force: true });
      resolve(data);
    });
  });
}

/**
 * Time of a video's first frame (s). Often 0.04 rather than 0 on the
 * downloaded clips; the analyzer's frame 0 is the first decoded frame, so the
 * page must offset by it or it draws each frame's data over the previous one.
 * Needs ffprobe (comes with ffmpeg); 0 without it.
 */
const startCache = new Map<string, number>();
function videoStart(file: string): Promise<number> {
  if (startCache.has(file)) return Promise.resolve(startCache.get(file)!);
  return new Promise((resolve) => {
    execFile("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time", "-of", "csv=p=0", file], (err, out) => {
      const t = err ? 0 : Number.parseFloat(out) || 0;
      startCache.set(file, t);
      resolve(t);
    });
  });
}

/** Every case the eval knows: real fixtures first, then synthetic scenarios. */
async function listCases() {
  const fx = fs.readdirSync(path.join(EVAL_DIR, "fixtures")).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)).sort();
  const baseline = fs.existsSync(path.join(EVAL_DIR, "baseline.json"))
    ? JSON.parse(fs.readFileSync(path.join(EVAL_DIR, "baseline.json"), "utf8")) : {};
  const { SCENARIOS } = await import("./scenarios");
  return [
    ...fx.map((name) => ({
      name, synthetic: false,
      video: fs.existsSync(path.join(VIDEO_DIR, name + ".mp4")),
      labelled: fs.existsSync(path.join(EVAL_DIR, "labels", name + ".json")),
      baseline: baseline[name] ?? null,
    })),
    ...SCENARIOS.map((s) => ({ name: s.name, synthetic: true, knownLimit: !!s.knownLimit, video: false, labelled: true, baseline: baseline[s.name] ?? null })),
  ];
}

function sendJson(res: http.ServerResponse, obj: unknown, code = 200) {
  const body = Buffer.from(JSON.stringify(obj));
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": body.length });
  res.end(body);
}

/** Video with Range support, so the page can seek. */
function sendVideo(req: http.IncomingMessage, res: http.ServerResponse, file: string) {
  const size = fs.statSync(file).size;
  const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? "");
  if (!m) {
    res.writeHead(200, { "Content-Type": "video/mp4", "Content-Length": size, "Accept-Ranges": "bytes" });
    fs.createReadStream(file).pipe(res);
    return;
  }
  const start = m[1] ? Number(m[1]) : Math.max(0, size - Number(m[2]));
  const end = m[1] && m[2] ? Math.min(Number(m[2]), size - 1) : size - 1;
  res.writeHead(206, {
    "Content-Type": "video/mp4", "Content-Length": end - start + 1, "Accept-Ranges": "bytes",
    "Content-Range": `bytes ${start}-${end}/${size}`,
  });
  fs.createReadStream(file, { start, end }).pipe(res);
}

function readBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let s = "";
    req.on("data", (d) => { s += d; });
    req.on("end", () => { try { resolve(s ? JSON.parse(s) : {}); } catch (e) { reject(e); } });
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://127.0.0.1:${PORT}`);
  try {
    if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      // Read on every request, so page edits show on reload.
      const body = fs.readFileSync(PAGE);
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Content-Length": body.length });
      res.end(body);
    } else if (req.method === "GET" && url.pathname === "/api/cases") {
      sendJson(res, { cases: await listCases() });
    } else if (req.method === "GET" && url.pathname.startsWith("/video/")) {
      const name = decodeURIComponent(url.pathname.slice(7)).replace(/\.mp4$/, "");
      const file = path.join(VIDEO_DIR, name + ".mp4");
      if (!NAME_RE.test(name) || !fs.existsSync(file)) return sendJson(res, { error: "no video" }, 404);
      sendVideo(req, res, file);
    } else if (req.method === "POST" && url.pathname === "/api/run") {
      const { name, params } = await readBody(req);
      if (name !== "--all" && !NAME_RE.test(name ?? "")) return sendJson(res, { error: "bad case name" }, 400);
      const data = (await runCase(name, params)) as Record<string, unknown>;
      const video = path.join(VIDEO_DIR, name + ".mp4");
      if (data.hasVideo) data.videoStart = await videoStart(video);
      sendJson(res, data);
    } else {
      sendJson(res, { error: "not found" }, 404);
    }
  } catch (e) {
    sendJson(res, { error: String((e as Error).message ?? e) }, 500);
  }
});

server.listen(PORT, "127.0.0.1", () => {
  const link = `http://127.0.0.1:${PORT}`;
  console.log(`Rally inspector at ${link}  (Ctrl+C to stop)`);
  if (!process.argv.includes("--no-browser")) {
    if (process.platform === "win32") execFile("cmd", ["/c", "start", "", link]);
    else execFile(process.platform === "darwin" ? "open" : "xdg-open", [link]);
  }
});
