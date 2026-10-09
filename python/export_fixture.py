#!/usr/bin/env python3
"""
Save the analyzer's output for a clip as an eval fixture.

    python python/export_fixture.py match.mp4            # runs the analyzer (GPU, ~40 s)
    python python/export_fixture.py match.mp4 --from-jsonl saved.jsonl

Writes eval/fixtures/<clip>.json: the same frame messages the app receives,
plus the court corners and video size. The TypeScript eval (npm run eval)
then runs the 3D solver on it in seconds, without video, Python or a GPU.
"""

import argparse
import json
import os
import sys

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--from-jsonl", help="analyzer stdout saved earlier, instead of running it")
    ap.add_argument("--name", help="clip name (default: video file name)")
    args = ap.parse_args()

    if args.from_jsonl:
        with open(args.from_jsonl, encoding="utf-8") as fh:
            messages = [json.loads(line) for line in fh if line.strip()]
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
    out_dir = os.path.join(EVAL_DIR, "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, separators=(",", ":"))
    n_ball = sum(1 for f in frames if f.get("ball_img_pos") and not f.get("ball_interpolated"))
    print(f"Wrote {os.path.normpath(out)}: {len(frames)} frames, {n_ball} with a detected ball")


if __name__ == "__main__":
    main()
