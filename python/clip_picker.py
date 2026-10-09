#!/usr/bin/env python3
"""
Web page for picking eval clips from YouTube and downloading them.

    python python/clip_picker.py          # opens http://127.0.0.1:8765

Search YouTube (or open a suggested match), play it, mark the rally's start
and end, and Fetch: only that section is downloaded (python/fetch_clip.py),
then the analyzer writes the eval fixture. The clip list shows what each
clip has (video, fixture, labels) and opens the touch labeller.

Local only: the server listens on 127.0.0.1. Standard library only.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "eval"))
VIDEO_DIR = os.path.join(EVAL_DIR, "videos")
SOURCES = os.path.join(EVAL_DIR, "clip_sources.json")
PAGE = os.path.join(HERE, "clip_picker.html")
PORT = 8765
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Full matches on the official Volleyball World channel, chosen for variety
# (arenas, years, men's and women's game). See eval/clips.md.
SUGGESTIONS = [
    {"id": "DykP0EdwJYk", "title": "France vs Japan · VNL 2025 week 2 (men)", "why": "Same match and camera as France_vs_Japan.mp4"},
    {"id": "M2IfS9CHo30", "title": "Poland vs Italy · VNL 2025 Gold Match (men)", "why": "Finals arena, top level"},
    {"id": "yQsN_rcj7Qs", "title": "Japan vs Poland · VNL 2025 quarterfinal (men)", "why": "Fast attacks, many blocks"},
    {"id": "lMq8llmamCA", "title": "USA vs Japan · VNL 2025 week 3 (men)", "why": "Another arena"},
    {"id": "pxs7U7TT8VA", "title": "Bulgaria vs Cuba · VNL 2025 week 3 (men)", "why": "Power serves, long match"},
    {"id": "KKRvJ379pUQ", "title": "Brazil vs Serbia · World Championship 2025 (men)", "why": "Different tournament and arena"},
    {"id": "eVez6NGb8mk", "title": "Poland vs Netherlands · World Championship 2025 (men)", "why": "Same tournament, other teams"},
    {"id": "g7GlUj61BYA", "title": "Japan vs France · VNL 2024 Gold Match (men)", "why": "Different year's production"},
    {"id": "ZGMSvIfdqZY", "title": "Italy vs Brazil · VNL 2025 Gold Match (women)", "why": "Long rallies"},
    {"id": "uMAuZu5bq6U", "title": "Japan vs Türkiye · VNL 2025 quarterfinal (women)", "why": "Dig, set, attack sequences"},
    {"id": "hUgn8XAdyJc", "title": "Thailand vs Netherlands · World Championship 2025 (women)", "why": "Very long rallies, other arena"},
    {"id": "qltlS20Ac64", "title": "USA vs Argentina · World Championship 2025 (women)", "why": "Another venue"},
]

JOBS = {}        # id -> {name, kind, status, log, started}
JOBS_LOCK = threading.Lock()


def load_sources():
    if os.path.isfile(SOURCES):
        with open(SOURCES, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def clip_status():
    """Every recorded clip, plus what exists for it on disk."""
    out = []
    for e in load_sources():
        name = e["name"]
        labels = os.path.join(EVAL_DIR, "labels", name + ".json")
        n_touches = None
        if os.path.isfile(labels):
            try:
                with open(labels, encoding="utf-8") as fh:
                    n_touches = len(json.load(fh).get("touches", []))
            except (OSError, ValueError):
                n_touches = None
        out.append({
            **e,
            "video": os.path.isfile(os.path.join(VIDEO_DIR, name + ".mp4")),
            "fixture": os.path.isfile(os.path.join(EVAL_DIR, "fixtures", name + ".json")),
            "touches": n_touches,
        })
    return out


def run_job(name, kind, cmd):
    """Run a command in the background, keeping its output for the page."""
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "name": name, "kind": kind, "status": "running", "log": "", "started": time.time()}
    with JOBS_LOCK:
        JOBS[jid] = job

    def work():
        try:
            proc = subprocess.Popen(cmd, cwd=os.path.join(HERE, os.pardir), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                # The analyzer is chatty; keep the lines that say what is happening.
                if line.startswith(("[fetch]", "Wrote", "[ball]", "[tracknet] Peak", "[court] Chose", "Traceback")) \
                        or "Error" in line or "error" in line:
                    with JOBS_LOCK:
                        job["log"] = (job["log"] + line)[-4000:]
            code = proc.wait()
            job["status"] = "done" if code == 0 else f"failed (exit {code})"
        except Exception as ex:  # surface anything to the page
            job["status"] = f"failed ({ex})"

    threading.Thread(target=work, daemon=True).start()
    return job


def yt_search(q, n=15):
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
        res = ydl.extract_info(f"ytsearch{n}:{q}", download=False)
    return [{"id": e.get("id"), "title": e.get("title"), "channel": e.get("channel"),
             "duration": e.get("duration")} for e in res.get("entries", []) if e.get("id")]


def yt_info(vid):
    """Title, length and best video quality, so the page can warn about low resolution."""
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    vids = [f for f in info.get("formats", []) if f.get("vcodec") not in (None, "none") and f.get("height")]
    best = max(vids, key=lambda f: (f["height"], f.get("fps") or 0), default=None)
    return {"id": vid, "title": info.get("title"), "channel": info.get("channel"), "duration": info.get("duration"),
            "height": best and best["height"], "fps": best and best.get("fps")}


def video_id(url_or_id):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    return url_or_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id) else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(PAGE, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with JOBS_LOCK:
                jobs = sorted(JOBS.values(), key=lambda j: -j["started"])[:20]
            self.send_json({"clips": clip_status(), "jobs": jobs, "suggestions": SUGGESTIONS})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self.read_json()
            if self.path == "/api/search":
                self.send_json({"results": yt_search(data.get("q", "").strip() or "volleyball full match")})
            elif self.path == "/api/info":
                vid = video_id(data.get("url", ""))
                if not vid:
                    return self.send_json({"error": "That doesn't look like a YouTube link."}, 400)
                self.send_json(yt_info(vid))
            elif self.path == "/api/fetch":
                vid = video_id(data.get("url", ""))
                name = data.get("name", "")
                start, end = float(data["start"]), float(data["end"])
                if not vid:
                    return self.send_json({"error": "That doesn't look like a YouTube link."}, 400)
                if not NAME_RE.match(name):
                    return self.send_json({"error": "Name: letters, digits, _ and - only."}, 400)
                if not 0 <= start < end or end - start > 120:
                    return self.send_json({"error": "Set a start before the end, at most 2 minutes apart."}, 400)
                cmd = [sys.executable, os.path.join(HERE, "fetch_clip.py"), f"https://www.youtube.com/watch?v={vid}",
                       f"{start:.2f}-{end:.2f}", "--name", name, "--note", data.get("note", "")]
                if data.get("fixture", True):
                    cmd.append("--fixture")
                self.send_json(run_job(name, "fetch", cmd))
            elif self.path == "/api/fixture":
                name = data.get("name", "")
                path = os.path.join(VIDEO_DIR, name + ".mp4")
                if not NAME_RE.match(name) or not os.path.isfile(path):
                    return self.send_json({"error": "No downloaded video for that clip."}, 400)
                self.send_json(run_job(name, "analyze", [sys.executable, os.path.join(HERE, "export_fixture.py"), path, "--name", name]))
            elif self.path == "/api/label":
                name = data.get("name", "")
                path = os.path.join(VIDEO_DIR, name + ".mp4")
                if not NAME_RE.match(name) or not os.path.isfile(path):
                    return self.send_json({"error": "No downloaded video for that clip."}, 400)
                # Opens the OpenCV labelling window on this machine.
                subprocess.Popen([sys.executable, os.path.join(HERE, "label_touches.py"), path],
                                 cwd=os.path.join(HERE, os.pardir))
                self.send_json({"ok": True})
            elif self.path == "/api/eval":
                npm = "npm.cmd" if os.name == "nt" else "npm"
                self.send_json(run_job("all", "eval", [npm, "run", "eval"]))
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as ex:
            self.send_json({"error": str(ex)}, 500)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Clip picker at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
