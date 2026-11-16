#!/usr/bin/env python3
"""
Compare person detectors without re-running the whole analyzer.

    python python/bench_players.py v8s yolov8s.pt                 # a model, every frame
    python python/bench_players.py v8s_s2 yolov8s.pt --stride 2   # every 2nd frame
    python python/bench_players.py mine python/models/players.pt --stride 2
    EVAL_FIXTURES_DIR=runs/fx_players_v8s npm run eval

Runs the detector over the labelled eval clips, prints its time per frame and
people found per frame, and writes fixtures to runs/fx_players_<tag> with the
player fields replaced; the ball, camera motion and court are kept from
eval/fixtures. The eval then shows what the detector does to the touches.

Two runs of the same detector can differ by a point on the holdout score (see
EVAL.md section 13, cuDNN), so only larger differences mean something; the
touch position ("touch pos") is the steadier number.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
PLAYER_KEYS = ("players", "players_all_ref", "players_near_ball", "players_ref", "players_arms", "players_arms_ref")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="name of this run (output folder runs/fx_players_<tag>)")
    ap.add_argument("weights", help="detector weights, e.g. yolov8s.pt (downloaded if missing)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=1, help="detect on every Nth frame; the others repeat the last result")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--base", default=os.path.join(ROOT, "eval", "fixtures"), help="fixtures to take everything else from")
    args = ap.parse_args()

    import analyzer as an
    from camera_motion import apply_h, map_box
    from ultralytics import YOLO
    model = YOLO(args.weights)
    out_dir = os.path.join(ROOT, "runs", "fx_players_" + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    # The pose model's raw result per frame, for the arms; the same for every detector.
    pose_path = os.path.join(ROOT, "runs", "players_cache", "pose.pkl")
    pose_cache = pickle.load(open(pose_path, "rb")) if os.path.exists(pose_path) else {}
    pose = None

    def detect(frame):
        h, w = frame.shape[:2]
        r = model.predict(frame, classes=[0], conf=args.conf, verbose=False, imgsz=args.imgsz,
                          quantize=16 if an._HALF_OK else None)[0]
        return [[round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5), round(c, 3)]
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]

    seconds = detected = people = 0
    for fp in sorted(glob.glob(os.path.join(args.base, "*.json"))):
        clip = os.path.splitext(os.path.basename(fp))[0]
        fx = json.load(open(fp, encoding="utf-8"))
        video = os.path.join(ROOT, "eval", "videos", clip + ".mp4")
        if os.path.exists(video):
            cap = cv2.VideoCapture(video)
            aspect = fx["video"]["width"] / fx["video"]["height"]
            last = None
            for fr in fx["frames"]:
                ok, img = cap.read()
                if not ok:
                    break
                fi = fr["frame"]
                if fi == 0:
                    detect(img)   # warm-up, not timed
                if fi % args.stride == 0 or last is None:
                    t0 = time.perf_counter()
                    last = detect(img)
                    seconds += time.perf_counter() - t0
                    detected += 1
                    people += len(last)
                had_arms = "players_arms" in fr
                for k in PLAYER_KEYS:
                    fr.pop(k, None)
                fr["players"] = last
                H = np.array(fr["ref_from_frame"], np.float64).reshape(3, 3) if "ref_from_frame" in fr else None
                if H is not None:
                    fr["players_all_ref"] = [map_box(H, b) for b in last]
                if fr.get("ball_img_pos") and not fr.get("ball_interpolated"):
                    near = an._players_near_ball(last, aspect, fr["ball_img_pos"][0], fr["ball_img_pos"][1])
                    fr["players_near_ball"] = near
                    if H is not None:
                        fr["players_ref"] = [map_box(H, b) for b in near]
                if had_arms:
                    key = (clip, fi)
                    if key not in pose_cache:
                        pose = pose or an._get_pose_model()
                        pose_cache[key] = an._detect_players(img, pose)
                    pose_boxes, pose_arms = pose_cache[key]
                    fr["players_arms"] = an._match_arms(last, pose_boxes, pose_arms or [])
                    if H is not None:
                        fr["players_arms_ref"] = [[apply_h(H, k[:2]) + [k[2]] for k in a] for a in fr["players_arms"]]
            cap.release()
        with open(os.path.join(out_dir, clip + ".json"), "w", encoding="utf-8") as fh:
            json.dump(fx, fh, separators=(",", ":"))
    os.makedirs(os.path.dirname(pose_path), exist_ok=True)
    with open(pose_path, "wb") as fh:
        pickle.dump(pose_cache, fh)
    if not detected:
        sys.exit("No eval videos found (eval/videos/<clip>.mp4)")
    per = 1000 * seconds / detected
    print(f"{args.tag}: {per:.1f} ms per detection, {per / args.stride:.1f} ms per video frame, "
          f"{people / detected:.1f} people per frame")
    print(f"Fixtures in {os.path.normpath(out_dir)}; score them with\n"
          f"  EVAL_FIXTURES_DIR={os.path.relpath(out_dir, ROOT).replace(os.sep, '/')} npm run eval")


if __name__ == "__main__":
    main()
