#!/usr/bin/env python3
"""
Check the camera-motion estimate for a clip by eye and by numbers.

    python python/debug_camera.py eval/videos/usa_jap_r3.mp4

Writes <video>_camdebug/:
  stabilised.mp4   every frame warped into the reference frame, with the
                   calibrated court lines drawn on top. If the motion estimate
                   is right, the court stands still and only the image edges
                   move; any slide or wobble of the court is estimation error.
  motion.csv       per frame: pan of the image centre (px), zoom vs reference,
                   tracking inliers, trusted or not.
and prints a summary. The reference is the court-calibration frame from
eval/fixtures/<clip>.json when it exists, else frame 0.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_motion import CameraMotion, apply_h  # noqa: E402

EVAL_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "eval"))
# Court lines in world (x, z) metres: outline, centre line, attack lines.
LINES = [((-4.5, -9), (4.5, -9)), ((4.5, -9), (4.5, 9)), ((4.5, 9), (-4.5, 9)), ((-4.5, 9), (-4.5, -9)),
         ((-4.5, 0), (4.5, 0)), ((-4.5, -3), (4.5, -3)), ((-4.5, 3), (4.5, 3))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--ref", type=int, help="reference frame (default: court frame from the fixture, else 0)")
    args = ap.parse_args()

    clip = os.path.splitext(os.path.basename(args.video))[0]
    corners, ref = None, 0
    fx_path = os.path.join(EVAL_DIR, "fixtures", clip + ".json")
    if os.path.isfile(fx_path):
        with open(fx_path, encoding="utf-8") as fh:
            fx = json.load(fh)
        corners = fx.get("courtCorners")
        ref = fx.get("courtFrame", 0)
    if args.ref is not None:
        ref = args.ref

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    motion = CameraMotion()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        motion.update(f)
        frames.append(f)
    cap.release()
    if not frames:
        sys.exit("Could not read the video")
    ref = min(ref, len(frames) - 1)
    motion.set_reference(ref)
    H_img, W_img = frames[0].shape[:2]

    # World court plane -> reference image (normalised), from the calibration corners.
    court_h = None
    if corners:
        world = np.float32([[-4.5, -9], [4.5, -9], [4.5, 9], [-4.5, 9]])
        court_h = cv2.getPerspectiveTransform(world, np.float32(corners))

    out_dir = os.path.splitext(args.video)[0] + "_camdebug"
    os.makedirs(out_dir, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(out_dir, "stabilised.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    N = np.diag([W_img, H_img, 1.0])
    rows, untrusted = [], 0
    for i, f in enumerate(frames):
        Hn = motion.ref_from_frame(i)
        if Hn is None:
            untrusted += 1
            canvas = np.zeros_like(f)
            cv2.putText(canvas, "motion not trusted", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            canvas = cv2.warpPerspective(f, N @ Hn @ np.linalg.inv(N), (W_img, H_img))
        if court_h is not None:
            for a, b in LINES:
                pa = cv2.perspectiveTransform(np.float32([[a]]), court_h)[0, 0] * [W_img, H_img]
                pb = cv2.perspectiveTransform(np.float32([[b]]), court_h)[0, 0] * [W_img, H_img]
                cv2.line(canvas, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (0, 0, 255), 2, cv2.LINE_AA)
        centre = apply_h(Hn, [0.5, 0.5]) if Hn is not None else None
        pan = None if centre is None else float(np.hypot((centre[0] - 0.5) * W_img, (centre[1] - 0.5) * H_img))
        zoom = motion.zoom(i)
        rows.append([i, None if pan is None else round(pan, 1), None if zoom is None else round(zoom, 4),
                     motion.quality[i], Hn is not None])
        label = f"frame {i}  ref {ref}  pan {pan or 0:.0f}px  zoom {zoom or 0:.3f}  inliers {motion.quality[i]}"
        cv2.rectangle(canvas, (0, 0), (W_img, 44), (0, 0, 0), -1)
        cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        vw.write(cv2.resize(canvas, (960, 540)))
    vw.release()

    with open(os.path.join(out_dir, "motion.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "pan_px", "zoom", "inliers", "trusted"])
        wr.writerows(rows)

    pans = [r[1] for r in rows if r[1] is not None]
    zooms = [r[2] for r in rows if r[2] is not None]
    print(f"{len(frames)} frames, reference {ref}, untrusted {untrusted}")
    if pans:
        print(f"pan from reference: max {max(pans):.0f} px; zoom vs reference: {min(zooms):.3f}-{max(zooms):.3f}")
    print(f"static camera: {'yes' if motion.is_static() else 'no'}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
