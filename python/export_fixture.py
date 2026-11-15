#!/usr/bin/env python3
"""
Save the analyzer's output for a clip as an eval fixture.

    python python/export_fixture.py match.mp4            # runs the analyzer (GPU, ~40 s)
    python python/export_fixture.py match.mp4 --from-jsonl saved.jsonl
    python python/export_fixture.py match.mp4 --server http://gpu-box:8765

With --server (or VB_ANALYSIS_SERVER set), the video is uploaded to an
analysis server (python/server.py, e.g. the Docker container in compose.yaml)
and analysed on its GPU; this machine needs neither torch nor CUDA.
fetch_clip.py and clip_picker.py run this script, so the variable reaches them too.
VB_TOKEN is sent if the server requires one.

Writes eval/fixtures/<clip>.json: the same frame messages the app receives,
plus the court corners and video size. The TypeScript eval (npm run eval)
then runs the 3D solver on it in seconds, without video, Python or a GPU.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EVAL_DIR = os.path.join(HERE, os.pardir, "eval")


def collect_messages(video):
    """Run analyze_video_cv and capture what it would send to the app."""
    import analyzer
    messages = []
    analyzer.send = messages.append  # instead of printing JSON lines to stdout
    analyzer.analyze_video_cv(video)
    return messages


def collect_messages_remote(video, server, token="", name=None, keep=False, timings=None):
    """Analyse `video` on an analysis server; the same messages as collect_messages.

    `timings`, if given, is filled with upload_s and analyze_s (from the analyze
    request to the last message, so it includes the server's video preparation).
    """
    timings = {} if timings is None else timings
    base = server.rstrip("/") + "/api"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def call(method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        h = dict(headers)
        if body is not None:
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            sys.exit(f"{method} {base + path}: {e.code} {e.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError as e:
            sys.exit(f"Analysis server {server} not reachable: {e.reason}")

    size = os.path.getsize(video)
    job = call("POST", "/jobs", {"name": name or os.path.basename(video), "size": size,
                                 "source": "file", "mime": "video/mp4"})
    jid = job["id"]
    try:
        t0 = time.perf_counter()
        piece = 32 << 20
        with open(video, "rb") as f:
            offset = 0
            while offset < size:
                chunk = f.read(piece)
                offset = call("PUT", f"/jobs/{jid}/video?offset={offset}", raw=chunk)["uploaded"]
                print(f"\rUploading {offset * 100 // size}%", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        timings["upload_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        call("POST", f"/jobs/{jid}/analyze", {})
        messages, after, last = [], 0, ""
        while True:
            r = call("GET", f"/jobs/{jid}/events?after={after}&limit=20000")
            messages += r["events"]
            after = r["next"]
            for m in r["events"]:
                if m.get("action") == "progress":
                    last = f"{m.get('percent', 0):3d}% {m.get('message', '')}"
                elif m.get("action") == "error":
                    print(f"\nServer: {m.get('message')}", file=sys.stderr)
            print(f"\r{last[:100]:<100}", end="", file=sys.stderr, flush=True)
            if r["done"]:
                timings["analyze_s"] = time.perf_counter() - t0
                print(file=sys.stderr)
                return messages
            if not r["events"]:
                time.sleep(1)
    finally:
        if not keep:
            try:
                call("DELETE", f"/jobs/{jid}")
            except SystemExit:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--from-jsonl", help="analyzer stdout saved earlier, instead of running it")
    ap.add_argument("--name", help="clip name (default: video file name)")
    ap.add_argument("--out-dir", default=os.path.join(EVAL_DIR, "fixtures"),
                    help="where to write it (default eval/fixtures; score another folder with EVAL_FIXTURES_DIR)")
    ap.add_argument("--server", default=os.environ.get("VB_ANALYSIS_SERVER", ""),
                    help="analyse on this analysis server, e.g. http://gpu-box:8765 (default: $VB_ANALYSIS_SERVER; "
                         "empty runs the analyzer here)")
    ap.add_argument("--keep-job", action="store_true",
                    help="with --server: leave the match in the server's library instead of deleting it")
    args = ap.parse_args()

    if args.from_jsonl:
        with open(args.from_jsonl, encoding="utf-8") as fh:
            messages = [json.loads(line) for line in fh if line.strip()]
    elif args.server:
        messages = collect_messages_remote(args.video, args.server, os.environ.get("VB_TOKEN", ""),
                                           name=os.path.basename(args.video), keep=args.keep_job)
    else:
        messages = collect_messages(args.video)

    frames = [{k: v for k, v in m.items() if k != "action"} for m in messages if m.get("action") == "frame"]
    done = next((m for m in messages if m.get("action") == "complete"), None)
    if not frames or done is None:
        sys.exit("Analyzer produced no frames / no completion message")
    court = done.get("court_data") or []
    if not court:
        # Still worth saving: the corners can be clicked in label_touches.py (key k).
        print("No court detected: place the court corners by hand in label_touches.py (key k)")

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = float(done.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    name = args.name or os.path.splitext(os.path.basename(args.video))[0]
    fixture = {
        "clip": name,
        "video": {"width": width, "height": height, "fps": fps},
        "courtCorners": court[0]["corners"] if court else None,
        # Frame the corners were detected in: the reference image when the camera moves.
        "courtFrame": court[0].get("frame", 0) if court else 0,
        "cameraMoving": bool(done.get("camera_moving", False)),
        "frames": frames,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, name + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, separators=(",", ":"))
    n_ball = sum(1 for f in frames if f.get("ball_img_pos") and not f.get("ball_interpolated"))
    print(f"Wrote {os.path.normpath(out)}: {len(frames)} frames, {n_ball} with a detected ball")


if __name__ == "__main__":
    main()
