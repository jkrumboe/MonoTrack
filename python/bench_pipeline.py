#!/usr/bin/env python3
"""
Time the analyzer on a clip, warm, with peak VRAM and GPU power.

    python python/bench_pipeline.py eval/videos/usa_jap_r3.mp4 --seconds 10
    python python/bench_pipeline.py eval/videos/usa_jap_r3.mp4            # the whole clip

"Warm" means the models are loaded and the GPU has run them once: a short
warm-up clip is analysed first and not counted. --seconds cuts the first N
seconds of the clip into a temporary file, so any length can be measured from
one video. The 3D solver runs in the app (TypeScript) and is timed by
`npm run eval`, not here.

Prints one line of JSON at the end (also appended to runs/bench_pipeline.jsonl):
analysis seconds, seconds per stage, seconds per second of video, peak VRAM added (MiB, from
nvidia-smi, so it covers ONNX and OpenCV too), mean and peak GPU power (W),
GPU utilisation, CPU model.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def smi(fields):
    """One reading of nvidia-smi fields for GPU 0, as floats (None if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return [float(v) for v in out.split(",")]
    except Exception:
        return None


class GpuWatch(threading.Thread):
    """Samples GPU memory, power and utilisation twice a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            r = smi("memory.used,power.draw,utilization.gpu")
            if r:
                self.samples.append(r)
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(timeout=3)


def cut(video, seconds):
    """The first `seconds` of a video as a temporary file (same size and frame rate)."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(round(seconds * fps))
    out = os.path.join(tempfile.mkdtemp(prefix="bench_"), f"cut_{seconds:g}s.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while written < n:
        ok, frame = cap.read()
        if not ok:
            break
        wr.write(frame)
        written += 1
    wr.release()
    cap.release()
    if written < n:
        sys.exit(f"{video} is only {written / fps:.1f} s long")
    return out


def cpu_model():
    if platform.system() == "Windows":
        try:
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def analyse(video):
    import analyzer
    messages = []
    analyzer.send = messages.append
    t0 = time.perf_counter()
    analyzer.analyze_video_cv(video)
    dt = time.perf_counter() - t0
    frames = sum(1 for m in messages if m.get("action") == "frame")
    balls = sum(1 for m in messages if m.get("action") == "frame" and m.get("ball_img_pos") and not m.get("ball_interpolated"))
    return dt, frames, balls, {k: round(v, 2) for k, v in analyzer.stage_seconds.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, help="measure the first N seconds only")
    ap.add_argument("--warmup-seconds", type=float, default=2.0, help="length of the uncounted warm-up clip")
    ap.add_argument("--repeat", type=int, default=1, help="timed runs (the fastest and the median are reported)")
    args = ap.parse_args()

    idle = smi("memory.used,power.draw")
    video = cut(args.video, args.seconds) if args.seconds else args.video
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    # Warm-up: loads every model and lets the GPU libraries pick their kernels.
    t0 = time.perf_counter()
    analyse(cut(args.video, args.warmup_seconds))
    warmup = time.perf_counter() - t0

    watch = GpuWatch()
    watch.start()
    times = []
    for _ in range(args.repeat):
        dt, frames, balls, stages = analyse(video)
        times.append(dt)
    watch.stop()

    times.sort()
    mem = [s[0] for s in watch.samples]
    power = [s[1] for s in watch.samples]
    util = [s[2] for s in watch.samples]
    result = {
        "video": os.path.basename(args.video), "seconds_of_video": round(n / fps, 2), "fps": fps,
        "size": f"{size[0]}x{size[1]}", "frames": frames, "frames_with_ball": balls,
        "analysis_s": round(times[len(times) // 2], 2), "analysis_s_best": round(times[0], 2),
        "s_per_s_of_video": round(times[len(times) // 2] / (n / fps), 2),
        # Seconds per stage of the last run. Camera motion and the court search
        # run on their own threads, alongside the players.
        "stages_s": stages,
        "warmup_s": round(warmup, 1),
        "vram_idle_mib": idle[0] if idle else None,
        "vram_peak_mib": max(mem) if mem else None,
        "vram_added_mib": round(max(mem) - idle[0]) if mem and idle else None,
        "gpu_power_mean_w": round(sum(power) / len(power), 1) if power else None,
        "gpu_power_peak_w": max(power) if power else None,
        "gpu_power_idle_w": idle[1] if idle else None,
        "gpu_util_mean_pct": round(sum(util) / len(util)) if util else None,
        "cpu": cpu_model(),
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }
    try:
        import torch
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
        result["torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
    except Exception:
        pass
    line = json.dumps(result)
    runs = os.path.join(HERE, os.pardir, "runs")
    os.makedirs(runs, exist_ok=True)
    with open(os.path.join(runs, "bench_pipeline.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
