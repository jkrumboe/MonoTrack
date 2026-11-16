"""
Analysis server: the web page and the phone app upload videos here and get
the analysis back. It does for them what the desktop app's Rust side
(src-tauri/src/lib.rs) does for a local file: keeps one warm analyzer process
(python/analyzer.py --warm), feeds it one video at a time, and relays its
JSON messages. The messages are also kept, so a finished match opens again
without being analysed again - the server is the match library.

Also serves the built web app (dist/), so one address is the whole service:

    npm run build
    python python/server.py                 # http://localhost:8765
    python python/server.py --cert c.pem --key k.pem   # https, needed for the
                                            # phone camera on anything but localhost

Standard library only. Uploads come in pieces (PUT .../video?offset=N) so an
upload over gym wifi can stop and carry on. A video recorded on a phone has a
variable frame rate; with ffmpeg on PATH it is re-encoded to a constant rate
first, so frame numbers and video time agree.

Environment:
    VB_DATA   where matches are kept (default ~/.tennis-3d/server)
    VB_TOKEN  if set, every /api request needs it (Authorization: Bearer <token>, or ?token=)

API (all JSON unless noted):
    GET    /api/health
    GET    /api/jobs                          list of matches
    POST   /api/jobs                          {name, size, source, mime} -> job
    GET    /api/jobs/<id>                     job
    DELETE /api/jobs/<id>
    PUT    /api/jobs/<id>/video?offset=N      body: raw bytes from N -> {uploaded}; 409 {uploaded} if N is wrong
    GET    /api/jobs/<id>/video               the video (Range supported)
    POST   /api/jobs/<id>/analyze             {rerun} -> job
    POST   /api/jobs/<id>/stop
    GET    /api/jobs/<id>/events?after=N&limit=M  -> {events, next, done, status}
    GET    /api/jobs/<id>/edits               -> {content}
    PUT    /api/jobs/<id>/edits               body: the edits file
    POST   /api/jobs/<id>/detect-court        {frame_time}
    POST   /api/jobs/<id>/court               {frame, corners}: save a hand-placed court
"""

import argparse
import gzip
import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import ssl
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZER = os.path.join(HERE, "analyzer.py")
DATA = os.environ.get("VB_DATA") or os.path.join(os.path.expanduser("~"), ".tennis-3d", "server")
JOBS = os.path.join(DATA, "jobs")
TOKEN = os.environ.get("VB_TOKEN") or ""
STATIC = os.path.normpath(os.path.join(HERE, "..", "dist"))

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
READ_CHUNK = 1 << 20
#: Largest upload piece accepted in one request.
MAX_PIECE = 64 << 20

_meta_lock = threading.RLock()


def log(*a):
    print("[server]", *a, file=sys.stderr, flush=True)


# ── Jobs on disk ──────────────────────────────────────────────────────────────
# jobs/<id>/meta.json     name, size, uploaded, status, progress, ...
#           upload.bin    the video as uploaded
#           video.mp4     constant-frame-rate copy (recordings, with ffmpeg)
#           events.jsonl  every message the analyzer sent for the last run
#           edits.json    hand corrections, points and team names


def job_dir(jid):
    return os.path.join(JOBS, jid)


def read_meta(jid):
    with _meta_lock:
        with open(os.path.join(job_dir(jid), "meta.json"), encoding="utf-8") as f:
            return json.load(f)


def write_meta(jid, create=False, **change):
    """Update a job's meta. Only `create` makes a new one: a deleted job stays deleted."""
    with _meta_lock:
        if create:
            meta = {}
        elif job_exists(jid):
            meta = read_meta(jid)
        else:
            return {}
        meta.update(change)
        tmp = os.path.join(job_dir(jid), "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(job_dir(jid), "meta.json"))
        return meta


def job_exists(jid):
    return bool(JOB_ID.match(jid)) and os.path.exists(os.path.join(job_dir(jid), "meta.json"))


def analysis_video(jid):
    """The file the analysis reads and the player plays: they must be the same."""
    d = job_dir(jid)
    converted = os.path.join(d, "video.mp4")
    return converted if os.path.exists(converted) else os.path.join(d, "upload.bin")


def list_jobs():
    if not os.path.isdir(JOBS):
        return []
    out = []
    for jid in os.listdir(JOBS):
        if job_exists(jid):
            try:
                out.append(read_meta(jid))
            except (OSError, ValueError):
                pass
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


# ── Event log with an index, so a client can read from any message ─────────


class EventLog:
    """Byte offsets of each line of a job's events.jsonl, extended as it grows."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index = {}  # jid -> (offsets list, scanned bytes)

    def reset(self, jid):
        with self._lock:
            self._index.pop(jid, None)

    def read(self, jid, after, limit):
        """Up to `limit` messages after the first `after`; returns (messages, next, total)."""
        path = os.path.join(job_dir(jid), "events.jsonl")
        if not os.path.exists(path):
            return [], 0, 0
        with self._lock:
            offsets, scanned = self._index.get(jid, ([], 0))
            with open(path, "rb") as f:
                f.seek(scanned)
                pos = scanned
                for line in f:
                    if not line.endswith(b"\n"):
                        break  # being written; next time
                    offsets.append(pos)
                    pos += len(line)
                self._index[jid] = (offsets, pos)
                total = len(offsets)
                if after >= total:
                    return [], total, total
                f.seek(offsets[after])
                lines = []
                for _ in range(min(limit, total - after)):
                    lines.append(f.readline())
        return [json.loads(l) for l in lines], after + len(lines), total


EVENTS = EventLog()


# ── Video preparation ─────────────────────────────────────────────────────────


def _probe_fps(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den or 1)
    except (OSError, ValueError, ZeroDivisionError, subprocess.SubprocessError):
        return 30
    # Phone recordings report e.g. 29.87 for a 30 fps stream.
    return min((24, 25, 30, 50, 60), key=lambda r: abs(r - fps)) if fps > 0 else 30


def prepare_video(jid, meta):
    """Re-encode phone recordings (variable frame rate, often WebM) to constant-rate MP4."""
    src = os.path.join(job_dir(jid), "upload.bin")
    dst = os.path.join(job_dir(jid), "video.mp4")
    if os.path.exists(dst):
        return
    name = meta.get("name", "").lower()
    needs = meta.get("source") == "recording" or name.endswith(".webm") or "webm" in meta.get("mime", "")
    if not needs:
        return
    if not shutil.which("ffmpeg"):
        log(f"{jid}: ffmpeg not found; analysing the recording as uploaded (frame times may drift)")
        return
    fps = _probe_fps(src)
    write_meta(jid, message=f"Converting the recording to {fps} fps")
    tmp = dst + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-an", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-500:]}")
    os.replace(tmp, dst)


# ── The warm analyzer ─────────────────────────────────────────────────────────


class Analyzer(threading.Thread):
    """Runs queued analyses one at a time on one long-lived analyzer process."""

    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.proc = None
        self.current = None
        self.stopping = False
        self._lock = threading.Lock()

    def _spawn(self):
        self._kill()
        log("starting analyzer")
        self.proc = subprocess.Popen(
            [sys.executable, ANALYZER, "--warm"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,  # stderr: the server's console
            text=True, encoding="utf-8", bufsize=1,
        )

    def _kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def submit(self, jid):
        write_meta(jid, status="queued", progress=0, message="Waiting for the analyzer", error=None)
        self.queue.put(jid)

    def stop(self, jid, wait=False):
        """Stop a job's analysis; with `wait`, until the worker has let go of its files."""
        with self._lock:
            if self.current == jid:
                self.stopping = True
                self._kill()  # the run loop sees the pipe close
        if read_meta(jid).get("status") in ("queued", "analyzing"):
            write_meta(jid, status="stopped", message="Stopped")
        deadline = time.time() + 10
        while wait and self.current == jid and time.time() < deadline:
            time.sleep(0.05)

    def run(self):
        while True:
            jid = self.queue.get()
            if not job_exists(jid) or read_meta(jid).get("status") != "queued":
                continue  # stopped or deleted while waiting
            with self._lock:
                self.current, self.stopping = jid, False
            try:
                self._analyze(jid)
            except Exception as e:  # keep serving the next job whatever happened
                log(f"{jid}: {e}")
                if self.stopping:  # killed mid-write by stop()
                    self.proc = None
                    if job_exists(jid):
                        write_meta(jid, status="stopped", message="Stopped")
                elif job_exists(jid):
                    write_meta(jid, status="error", message=str(e))
                    self._append(jid, {"action": "error", "message": str(e)})
            finally:
                with self._lock:
                    self.current = None

    def _append(self, jid, msg):
        try:
            with open(os.path.join(job_dir(jid), "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(msg) + "\n")
        except OSError:
            pass  # deleted meanwhile

    def _analyze(self, jid):
        meta = write_meta(jid, status="analyzing", progress=0, message="Preparing the video", started=_now())
        events = os.path.join(job_dir(jid), "events.jsonl")
        open(events, "w").close()
        EVENTS.reset(jid)
        prepare_video(jid, meta)
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()
        request = {"action": "analyze", "video_path": analysis_video(jid)}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        last_meta = 0.0
        with open(events, "a", encoding="utf-8") as out:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    log(f"analyzer: {line[:200]}")
                    continue
                action = msg.get("action")
                if action in ("ready", "pong"):
                    continue
                out.write(line + "\n")
                if action == "progress" and time.time() - last_meta > 1:
                    out.flush()
                    write_meta(jid, progress=int(msg.get("percent", 0)), message=msg.get("message", ""))
                    last_meta = time.time()
                elif action == "complete":
                    write_meta(jid, status="complete", progress=100, message="Done", finished=_now(),
                               frames=msg.get("total_frames"), fps=msg.get("fps"))
                    return
                elif action == "error":
                    write_meta(jid, status="error", message=msg.get("message", "Analysis failed"))
                    return
        # The analyzer's output ended without an answer: it was stopped or died.
        self.proc = None
        if self.stopping:
            write_meta(jid, status="stopped", message="Stopped")
        else:
            raise RuntimeError("The analyzer process exited unexpectedly")


class CourtDetector:
    """A second analyzer process for court detection, so it answers while an analysis runs."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()

    def detect(self, video_path, frame_time):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = subprocess.Popen(
                    [sys.executable, ANALYZER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    text=True, encoding="utf-8", bufsize=1,
                )
            req = {"action": "detect_court", "video_path": video_path, "frame_time": frame_time}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("action") in ("court_detected", "court_failed", "error"):
                    return msg
            self.proc = None
            return {"action": "error", "message": "Court detector process ended unexpectedly"}


ANALYZER_WORKER = Analyzer()
COURT_DETECTOR = CourtDetector()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_torch_cuda = None


def gpu_status():
    """What the analyzer will run on: torch's CUDA view (checked once) and a live nvidia-smi reading.

    nvidia-smi alone is not enough: inside a container it can see the card while
    the installed torch is a CPU build, and the analysis then silently runs on the CPU.
    """
    global _torch_cuda
    if _torch_cuda is None:
        code = ("import json, torch; ok = torch.cuda.is_available(); print(json.dumps({"
                "'cuda': ok, 'device': torch.cuda.get_device_name(0) if ok else None, "
                "'torch': torch.__version__, 'cuda_version': torch.version.cuda}))")
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
            _torch_cuda = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception as e:
            _torch_cuda = {"cuda": False, "error": f"torch check failed: {e}"}
    status = dict(_torch_cuda)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw,driver_version",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        name, total, used, util, power, driver = [v.strip() for v in out.split(",")]

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        status.update(smi_name=name, vram_total_mib=num(total), vram_used_mib=num(used),
                      util_pct=num(util), power_w=num(power), driver=driver)
    except Exception:
        status["smi"] = "nvidia-smi not available"
    return status


def _public(meta):
    keys = ("id", "name", "created", "size", "uploaded", "status", "progress", "message", "source", "frames", "fps")
    return {k: meta.get(k) for k in keys}


# ── HTTP ──────────────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    server_version = "VolleyballAnalysis/1.0"

    def log_message(self, fmt, *args):
        if "/events" not in self.path:  # polled every second; too chatty
            # <video src> carries the token in the query; keep it out of the log.
            log(self.address_string(), re.sub(r"(token=)[^&\s\"]+", r"\1***", fmt % args))

    # helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin") or "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")
        self.send_header("Vary", "Origin")

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        gz = len(body) > 4096 and "gzip" in self.headers.get("Accept-Encoding", "")
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, **extra):
        self._json({"error": message, **extra}, status)

    def _body(self, limit=MAX_PIECE):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("request too large")
        return self.rfile.read(n) if n else b""

    def _json_body(self):
        raw = self._body(1 << 20)
        return json.loads(raw) if raw else {}

    def _authorized(self, query):
        if not TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        given = auth[7:] if auth.startswith("Bearer ") else (query.get("token") or [""])[0]
        return secrets.compare_digest(given, TOKEN)

    # dispatch
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if not parts or parts[0] != "api":
                # e.g. /192.168.2.101/api/jobs from a server address set without http://:
                # an error, not the app's index.html with 200.
                if "api" in parts:
                    return self._error(404, "not found: the API is at /api/ (check the server address)")
                if method in ("GET", "HEAD"):
                    return self._static(url.path)
                return self._error(404, "not found")
            if not self._authorized(query):
                return self._error(401, "access token required")
            return self._api(method, parts[1:], query)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # report instead of dropping the connection
            log(f"{method} {self.path}: {e!r}")
            try:
                self._error(500, str(e))
            except OSError:
                pass

    def _api(self, method, p, query):
        if p == ["health"]:
            health = {"ok": True, "ffmpeg": bool(shutil.which("ffmpeg")), "auth": bool(TOKEN)}
            if query.get("gpu", ["0"])[0] == "1":  # not on the Docker healthcheck: nvidia-smi costs ~0.1 s
                health["gpu"] = gpu_status()
            return self._json(health)
        if p == ["jobs"] and method == "GET":
            return self._json([_public(m) for m in list_jobs()])
        if p == ["jobs"] and method == "POST":
            body = self._json_body()
            jid = uuid.uuid4().hex
            os.makedirs(job_dir(jid))
            open(os.path.join(job_dir(jid), "upload.bin"), "wb").close()
            name = os.path.basename(str(body.get("name") or "video.mp4"))[:200]
            meta = write_meta(jid, create=True, id=jid, name=name, created=_now(), size=int(body.get("size") or 0), uploaded=0,
                              source=body.get("source") or "file", mime=str(body.get("mime") or ""),
                              status="uploading", progress=0, message="")
            return self._json(_public(meta), 201)
        if len(p) < 2 or p[0] != "jobs" or not job_exists(p[1]):
            return self._error(404, "no such match")
        jid, rest = p[1], p[2:]

        if rest == [] and method == "GET":
            return self._json(_public(read_meta(jid)))
        if rest == [] and method == "DELETE":
            ANALYZER_WORKER.stop(jid, wait=True)
            # meta.json first, so a half-deleted folder (a file still open on
            # Windows) no longer counts as a match.
            os.remove(os.path.join(job_dir(jid), "meta.json"))
            shutil.rmtree(job_dir(jid), ignore_errors=True)
            EVENTS.reset(jid)
            return self._json({"deleted": jid})
        if rest == ["video"] and method == "PUT":
            return self._upload_piece(jid, query)
        if rest == ["video"] and method in ("GET", "HEAD"):
            return self._file(analysis_video(jid), method, "video/mp4" if analysis_video(jid).endswith(".mp4") else None)
        if rest == ["analyze"] and method == "POST":
            body = self._json_body()
            meta = read_meta(jid)
            if meta["status"] == "uploading":
                if meta["uploaded"] < meta["size"]:
                    return self._error(409, "the upload is not finished", uploaded=meta["uploaded"])
                ANALYZER_WORKER.submit(jid)
            elif meta["status"] in ("error", "stopped") or (meta["status"] == "complete" and body.get("rerun")):
                ANALYZER_WORKER.submit(jid)
            return self._json(_public(read_meta(jid)))
        if rest == ["stop"] and method == "POST":
            ANALYZER_WORKER.stop(jid)
            return self._json(_public(read_meta(jid)))
        if rest == ["events"] and method == "GET":
            after = max(0, int((query.get("after") or ["0"])[0]))
            limit = min(20000, max(1, int((query.get("limit") or ["5000"])[0])))
            status = read_meta(jid)["status"]  # before reading, so "done" never skips the last lines
            events, nxt, total = EVENTS.read(jid, after, limit)
            done = status in ("complete", "error", "stopped") and nxt >= total
            return self._json({"events": events, "next": nxt, "done": done, "status": status})
        if rest == ["edits"] and method == "GET":
            path = os.path.join(job_dir(jid), "edits.json")
            content = open(path, encoding="utf-8").read() if os.path.exists(path) else None
            return self._json({"content": content})
        if rest == ["edits"] and method == "PUT":
            raw = self._body(16 << 20)
            json.loads(raw)  # refuse anything that is not JSON
            path = os.path.join(job_dir(jid), "edits.json")
            with open(path + ".tmp", "wb") as f:
                f.write(raw)
            os.replace(path + ".tmp", path)
            return self._json({"saved": True})
        if rest == ["detect-court"] and method == "POST":
            body = self._json_body()
            return self._json(COURT_DETECTOR.detect(analysis_video(jid), float(body.get("frame_time") or 0)))
        if rest == ["court"] and method == "POST":
            body = self._json_body()
            req = {"video_path": analysis_video(jid), "frame": int(body.get("frame") or 0), "corners": body["corners"]}
            r = subprocess.run([sys.executable, ANALYZER, "--save-court", json.dumps(req)], capture_output=True, text=True)
            if r.returncode != 0:
                return self._error(500, r.stderr.strip()[-500:])
            return self._json(json.loads(r.stdout.strip().splitlines()[-1]))
        return self._error(404, "not found")

    def _upload_piece(self, jid, query):
        offset = int((query.get("offset") or ["0"])[0])
        path = os.path.join(job_dir(jid), "upload.bin")
        with _meta_lock:
            meta = read_meta(jid)
            if meta["status"] != "uploading":
                return self._error(409, "already uploaded", uploaded=meta["uploaded"])
            if offset != meta["uploaded"]:
                return self._error(409, "wrong offset", uploaded=meta["uploaded"])
            data = self._body()
            if meta["size"] and offset + len(data) > meta["size"]:
                return self._error(400, "more data than the declared size", uploaded=meta["uploaded"])
            with open(path, "r+b") as f:
                f.seek(offset)
                f.write(data)
                f.truncate()
            meta = write_meta(jid, uploaded=offset + len(data))
        return self._json({"uploaded": meta["uploaded"]})

    def _file(self, path, method, content_type=None):
        if not os.path.isfile(path):
            return self._error(404, "not found")
        size = os.path.getsize(path)
        ctype = content_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        status = 200
        if rng and size:
            a, b = rng.groups()
            if a:
                start, end = int(a), min(int(b) if b else size - 1, size - 1)
            elif b:  # last N bytes
                start, end = max(0, size - int(b)), size - 1
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self._cors()
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if method == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(READ_CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _static(self, path):
        if not os.path.isdir(STATIC):
            return self._error(404, "web app not built: run `npm run build`")
        rel = os.path.normpath(unquote(path).lstrip("/")) if path not in ("", "/") else "index.html"
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
            full = os.path.join(STATIC, "index.html")  # single-page app
        return self._file(full, self.command)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets phones on the same network connect")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cert", help="TLS certificate (PEM); with --key serves https")
    ap.add_argument("--key", help="TLS private key (PEM)")
    args = ap.parse_args()

    os.makedirs(JOBS, exist_ok=True)
    # Analyses cut short by a restart are queued again.
    for meta in list_jobs():
        if meta.get("status") in ("queued", "analyzing"):
            ANALYZER_WORKER.submit(meta["id"])
    ANALYZER_WORKER.start()

    # On Windows, address reuse lets a second server bind the same port and
    # silently share its requests with the first; fail instead.
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    scheme = "http"
    if args.cert and args.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    log(f"matches in {DATA}")
    log(f"listening on {scheme}://{args.host}:{args.port}" + ("  (token required)" if TOKEN else ""))
    if not os.path.isdir(STATIC):
        log("dist/ not found - API only. Run `npm run build` to serve the web app too.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ANALYZER_WORKER._kill()


if __name__ == "__main__":
    main()
