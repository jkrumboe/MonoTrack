#!/usr/bin/env python3
"""
Quick check that a GPU analysis server works, in about a minute instead of hours.

    python python/server_check.py --server http://gpu-box:8765
    python python/server_check.py --server http://gpu-box:8765 --gpu-only          # 2 s: no analysis
    python python/server_check.py eval/videos/vb14/graz-gleisdorf_1.avi --start 600
    python python/server_check.py eval/videos/*.mp4 --seconds 10                   # several videos

The server comes from --server or $VB_ANALYSIS_SERVER; $VB_TOKEN is sent if set.

What it does:
  1. Asks the server what it runs on: torch with CUDA, the GPU's name and VRAM.
     A CPU-only torch is a FAIL even though the analysis would still finish (hours later).
  2. Cuts a short stretch (default 20 s, from 2 min in) out of each video with
     ffmpeg, so a 20-minute match costs a 20 s upload and about a minute of analysis.
     Any video works: vb14 .avi, the eval clips, phone recordings.
  3. Analyses a 2 s warm-up clip first (loads the models; reported as "cold start"),
     then the stretch, while polling the server's GPU load and VRAM.
  4. Prints PASS / WARN / FAIL per check, the speed in seconds of analysis per
     second of video, and what the whole video would take at that speed.

Each result is appended to runs/server_check.jsonl, to compare servers and days.
Exit code 1 if any check FAILs. The jobs are deleted from the server afterwards.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))

from export_fixture import collect_messages_remote  # noqa: E402

#: Laptop baseline (EVAL.md section 7): RTX 3070 Ti Laptop, 1080p 25 fps, warm.
LAPTOP_S_PER_S = 3.0
DEFAULT_VIDEOS = [
    os.path.join(ROOT, "eval", "videos", "vb14", "graz-arbesbach_2.avi"),
    os.path.join(ROOT, "eval", "videos", "usa_jap_r3.mp4"),
]


# ── server ───────────────────────────────────────────────────────────────────


def get_health(server, token, timeout=150):
    """/api/health?gpu=1, or raises. The first call can take a while: the server imports torch once."""
    req = urllib.request.Request(server.rstrip("/") + "/api/health?gpu=1",
                                 headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class GpuPoll(threading.Thread):
    """Polls the server's GPU once a second while an analysis runs."""

    def __init__(self, server, token):
        super().__init__(daemon=True)
        self.server, self.token = server, token
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                g = get_health(self.server, self.token, timeout=10).get("gpu") or {}
                if g.get("util_pct") is not None:
                    self.samples.append(g)
            except Exception:
                pass
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        self.join(timeout=15)

    def summary(self, idle_vram):
        if not self.samples:
            return {}
        util = [s["util_pct"] for s in self.samples]
        vram = [s["vram_used_mib"] for s in self.samples if s.get("vram_used_mib") is not None]
        power = [s["power_w"] for s in self.samples if s.get("power_w") is not None]
        return {
            "gpu_util_mean_pct": round(sum(util) / len(util)),
            "gpu_util_peak_pct": max(util),
            "vram_peak_mib": max(vram) if vram else None,
            "vram_added_mib": round(max(vram) - idle_vram) if vram and idle_vram is not None else None,
            "gpu_power_mean_w": round(sum(power) / len(power), 1) if power else None,
            "gpu_samples": len(self.samples),
        }


# ── video ────────────────────────────────────────────────────────────────────


def probe(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, n / fps if n > 0 else 0.0, size


def cut(video, start, seconds, out_dir, tag):
    """`seconds` of `video` from `start` as an MP4 of the same size and frame rate."""
    out = os.path.join(out_dir, f"{tag}.mp4")
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", video, "-t", f"{seconds:.3f}",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out]
        if subprocess.run(cmd).returncode == 0 and os.path.getsize(out) > 0:
            return out
        print("ffmpeg failed, cutting with OpenCV", file=sys.stderr)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(round(seconds * fps))):
        ok, frame = cap.read()
        if not ok:
            break
        wr.write(frame)
    wr.release()
    cap.release()
    return out


# ── report ───────────────────────────────────────────────────────────────────


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, level, what, detail=""):
        self.rows.append((level, what, detail))
        print(f"  [{level:<4}] {what}" + (f": {detail}" if detail else ""))

    @property
    def failed(self):
        return any(r[0] == "FAIL" for r in self.rows)


def fmt_duration(s):
    s = int(round(s))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def check_server(server, token, checks, expect):
    print(f"\nServer {server}")
    try:
        health = get_health(server, token)
    except urllib.error.HTTPError as e:
        checks.add("FAIL", "reachable", f"HTTP {e.code}" + (" (set VB_TOKEN)" if e.code == 401 else ""))
        return None
    except Exception as e:
        checks.add("FAIL", "reachable", str(getattr(e, "reason", e)))
        return None
    checks.add("PASS", "reachable", "token required" if health.get("auth") else "no token")
    checks.add("PASS" if health.get("ffmpeg") else "WARN", "ffmpeg on the server",
               "" if health.get("ffmpeg") else "phone videos (variable frame rate) won't be re-encoded")
    g = health.get("gpu")
    if g is None:
        checks.add("FAIL", "GPU status", "server too old to report it (/api/health?gpu=1); rebuild the image")
        return health
    if g.get("cuda"):
        checks.add("PASS", "torch sees CUDA", f"{g.get('device')}, torch {g.get('torch')}, CUDA {g.get('cuda_version')}")
    else:
        checks.add("FAIL", "torch sees CUDA", g.get("error") or f"torch {g.get('torch')} runs on the CPU only")
    if g.get("smi_name"):
        total, used = g.get("vram_total_mib"), g.get("vram_used_mib")
        checks.add("PASS", "nvidia-smi", f"{g['smi_name']}, driver {g.get('driver')}, "
                   f"VRAM {used:.0f} / {total:.0f} MiB used, {g.get('util_pct'):.0f}% busy")
        if used is not None and total and total - used < 2048:
            checks.add("WARN", "free VRAM", f"only {total - used:.0f} MiB free; the analyzer needs ~1-1.5 GB")
    else:
        checks.add("WARN", "nvidia-smi", "not available in the container; GPU load won't be measured")
    if expect:
        name = g.get("device") or g.get("smi_name") or ""
        checks.add("PASS" if expect.lower() in name.lower() else "FAIL", f"GPU is a {expect}", name or "none")
    return health


def run_clip(server, token, clip, label, timings):
    msgs = collect_messages_remote(clip, server, token, name=f"server_check {label}", timings=timings)
    return msgs


def check_video(args, video, idle_vram, tmp, checks):
    fps, duration, size = probe(video)
    seconds = min(args.seconds, max(duration - 1, 1)) if duration else args.seconds
    start = min(args.start, max(duration - seconds, 0)) if duration else args.start
    print(f"\n{os.path.basename(video)}: {fmt_duration(duration)}, {size[0]}x{size[1]}, {fps:g} fps"
          f" -> testing {seconds:g} s from {start:g} s")
    clip = cut(video, start, seconds, tmp, os.path.splitext(os.path.basename(video))[0])

    poll = GpuPoll(args.server, args.token)
    poll.start()
    t = {}
    msgs = run_clip(args.server, args.token, clip, os.path.basename(video), t)
    poll.stop()

    frames = [m for m in msgs if m.get("action") == "frame"]
    done = next((m for m in msgs if m.get("action") == "complete"), None)
    errors = [m.get("message") for m in msgs if m.get("action") == "error"]
    expected = int(round(seconds * fps))
    if done is None:
        checks.add("FAIL", "analysis finished", "; ".join(e for e in errors if e) or "no completion message")
    elif abs(len(frames) - expected) > max(3, expected * 0.02):
        checks.add("WARN", "analysis finished", f"{len(frames)} frames, expected ~{expected}")
    else:
        checks.add("PASS", "analysis finished", f"{len(frames)} frames")

    balls = sum(1 for f in frames if f.get("ball_img_pos") and not f.get("ball_interpolated"))
    if frames:
        share = balls / len(frames)
        checks.add("PASS" if share >= 0.2 else "WARN", "ball detected",
                   f"{balls} of {len(frames)} frames ({share:.0%})" + ("" if share >= 0.2 else "; a break in play, or the detector failed"))
    court = (done or {}).get("court_data") or []
    checks.add("PASS" if court else "WARN", "court found", "" if court else "not detected (can be placed by hand)")

    gpu = poll.summary(idle_vram)
    analyze_s = t.get("analyze_s")
    s_per_s = analyze_s / seconds if analyze_s else None
    if gpu:
        busy = gpu["gpu_util_peak_pct"] >= 15
        checks.add("PASS" if busy else "FAIL", "GPU used during analysis",
                   f"{gpu['gpu_util_mean_pct']}% mean, {gpu['gpu_util_peak_pct']:.0f}% peak, "
                   f"+{gpu['vram_added_mib']} MiB VRAM, {gpu['gpu_power_mean_w']} W")
    if s_per_s:
        level = "PASS" if s_per_s <= LAPTOP_S_PER_S * 1.5 else "WARN"
        checks.add(level, "speed", f"{analyze_s:.1f} s for {seconds:g} s of video = {s_per_s:.2f} s per s "
                   f"(laptop 3070 Ti: ~{LAPTOP_S_PER_S:g}); upload {t.get('upload_s', 0):.1f} s")
        if duration:
            print(f"  -> the whole video ({fmt_duration(duration)}) would take about {fmt_duration(duration * s_per_s)}")

    return {
        "video": os.path.basename(video), "start_s": start, "seconds": seconds, "fps": fps,
        "size": f"{size[0]}x{size[1]}", "video_duration_s": round(duration, 1),
        "frames": len(frames), "frames_with_ball": balls, "court": bool(court),
        "upload_s": round(t.get("upload_s", 0), 1), "analysis_s": round(analyze_s, 1) if analyze_s else None,
        "s_per_s_of_video": round(s_per_s, 2) if s_per_s else None,
        "full_video_estimate_s": round(duration * s_per_s) if s_per_s and duration else None,
        **gpu,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", help="videos or globs (default: vb14 graz-arbesbach_2, else usa_jap_r3)")
    ap.add_argument("--server", default=os.environ.get("VB_ANALYSIS_SERVER", ""),
                    help="e.g. http://gpu-box:8765 (default $VB_ANALYSIS_SERVER)")
    ap.add_argument("--token", default=os.environ.get("VB_TOKEN", ""), help="default $VB_TOKEN")
    ap.add_argument("--seconds", type=float, default=20, help="length of the stretch analysed per video (default 20)")
    ap.add_argument("--start", type=float, default=120, help="where the stretch starts, in seconds (default 120; "
                    "clamped for short clips)")
    ap.add_argument("--gpu-only", action="store_true", help="only check the server and its GPU, analyse nothing")
    ap.add_argument("--no-warmup", action="store_true", help="skip the 2 s warm-up clip (then the first video "
                    "includes loading the models)")
    ap.add_argument("--expect-gpu", default="", help="FAIL unless the GPU name contains this, e.g. 3060")
    args = ap.parse_args()
    if not args.server:
        sys.exit("No server: pass --server http://<host>:8765 or set VB_ANALYSIS_SERVER")

    videos = []
    for v in args.videos:
        videos += sorted(glob.glob(v)) or [v]
    if not videos and not args.gpu_only:
        videos = [next((v for v in DEFAULT_VIDEOS if os.path.exists(v)), None)]
        if videos[0] is None:
            sys.exit("No video given and none of the default ones exist")

    checks = Checks()
    health = check_server(args.server, args.token, checks, args.expect_gpu)
    results = []
    if health is not None and not args.gpu_only:
        idle_vram = (health.get("gpu") or {}).get("vram_used_mib")
        tmp = tempfile.mkdtemp(prefix="server_check_")
        try:
            if not args.no_warmup:
                print("\nWarm-up (2 s clip; loads the models if the analyzer isn't running yet)")
                clip = cut(videos[0], 0, 2, tmp, "warmup")
                t = {}
                run_clip(args.server, args.token, clip, "warm-up", t)
                checks.add("PASS", "cold start", f"{t.get('analyze_s', 0):.1f} s for a 2 s clip")
            for video in videos:
                results.append(check_video(args, video, idle_vram, tmp, checks))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    record = {"when": time.strftime("%Y-%m-%d %H:%M"), "server": args.server,
              "gpu": (health or {}).get("gpu"), "failed": checks.failed, "videos": results}
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "server_check.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    counts = {lvl: sum(1 for r in checks.rows if r[0] == lvl) for lvl in ("PASS", "WARN", "FAIL")}
    print(f"\n{'FAILED' if checks.failed else 'OK'}: {counts['PASS']} passed, {counts['WARN']} warnings, "
          f"{counts['FAIL']} failed  (saved to runs/server_check.jsonl)")
    sys.exit(1 if checks.failed else 0)


if __name__ == "__main__":
    main()
