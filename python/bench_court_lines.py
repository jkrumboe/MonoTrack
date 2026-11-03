"""Score court detection against the hand-placed courts in eval/labels.

For every labelled clip with a video in eval/videos, runs the detector on
frames across the clip and compares the court it finds with the labelled one,
carried into that frame by the fixture's camera-motion homographies. The error
is the mean distance, in 1920x1080 pixels, over the court keypoints
(sideline x baseline / 3 m line / centre line) that are in view.

    python python/bench_court_lines.py [--stride 25] [--only pol_ita_r1,...]
                                       [--detector lines|colour] [--draw DIR]
                                       [--label-frames]

--label-frames scores only the frames the court points were clicked on, where
the truth does not depend on the camera-motion estimate.
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EVAL = os.path.join(HERE, "..", "eval")

KEYPOINTS = np.array([[x, z] for x in (-4.5, 4.5) for z in (-9, -3, 0, 3, 9)], np.float64)
CORNERS = np.array([[-4.5, -9], [4.5, -9], [4.5, 9], [-4.5, 9]], np.float64)


def _apply(H, pts):
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    return p[:, :2] / p[:, 2:3]


def _ref_from_frame(fx, i):
    H = fx["frames"][i].get("ref_from_frame") if i < len(fx["frames"]) else None
    return np.eye(3) if H is None else np.array(H, np.float64).reshape(3, 3)


def truth_homography(fx, court):
    """Centred court (x, z) -> normalised reference-image coords."""
    world, img = [], []
    for p in court["points"]:
        if len(p["world"]) > 2 and p["world"][2]:
            continue      # points above the floor (antennas) are not on the plane
        q = _apply(_ref_from_frame(fx, p["frame"]), np.array([p["pt"]], np.float64))[0]
        world.append(p["world"][:2])
        img.append(q)
    H, _ = cv2.findHomography(np.array(world, np.float64), np.array(img, np.float64))
    return H


def frame_error(H_truth_frame, corners_det, W=1920, Hh=1080):
    """Mean px error over in-view keypoints; None if nothing is in view."""
    Hd, _ = cv2.findHomography(CORNERS, np.array(corners_det, np.float64))
    t = _apply(H_truth_frame, KEYPOINTS)
    d = _apply(Hd, KEYPOINTS)
    vis = (t[:, 0] > 0.0) & (t[:, 0] < 1.0) & (t[:, 1] > 0.0) & (t[:, 1] < 1.0)
    if not vis.any():
        return None
    e = np.hypot((t[vis, 0] - d[vis, 0]) * W, (t[vis, 1] - d[vis, 1]) * Hh)
    return float(e.mean())


def draw(frame, corners, color):
    h, w = frame.shape[:2]
    Hd, _ = cv2.findHomography(CORNERS, np.array(corners, np.float64))
    for x1, z1, x2, z2 in ([(-4.5, -9, -4.5, 9), (4.5, -9, 4.5, 9)] +
                           [(-4.5, z, 4.5, z) for z in (-9, -3, 0, 3, 9)]):
        seg = _apply(Hd, np.linspace([x1, z1], [x2, z2], 40))
        pts = (seg * [w, h]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--only", default="")
    ap.add_argument("--detector", default="lines", choices=["lines", "colour"])
    ap.add_argument("--draw", default="")
    ap.add_argument("--label-frames", action="store_true")
    args = ap.parse_args()

    if args.detector == "lines":
        from court_lines import detect_court_lines as detect
    else:
        from court_detect import detect_court_corners as detect

    only = set(filter(None, args.only.split(",")))
    all_err, n_frames, n_found, total_t = [], 0, 0, 0.0
    for label_path in sorted(glob.glob(os.path.join(EVAL, "labels", "*.json"))):
        clip = os.path.splitext(os.path.basename(label_path))[0]
        if only and clip not in only:
            continue
        video = os.path.join(EVAL, "videos", clip + ".mp4")
        fx_path = os.path.join(EVAL, "fixtures", clip + ".json")
        court = json.load(open(label_path)).get("court")
        if not court or not os.path.exists(video) or not os.path.exists(fx_path):
            continue
        fx = json.load(open(fx_path))
        Href = truth_homography(fx, court)
        label_frames = {p["frame"] for p in court["points"]}
        cap = cv2.VideoCapture(video)
        errs, misses, i = [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (i in label_frames) if args.label_frames else (i % args.stride == 0):
                n_frames += 1
                Ht = np.linalg.inv(_ref_from_frame(fx, i)) @ Href
                t0 = time.time()
                corners, info = detect(frame)
                total_t += time.time() - t0
                if corners is None:
                    misses.append(i)
                else:
                    n_found += 1
                    e = frame_error(Ht, corners)
                    if e is not None:
                        errs.append((i, e, info.get("view", info.get("hypothesis"))))
                if args.draw:
                    os.makedirs(args.draw, exist_ok=True)
                    out = frame.copy()
                    Hc = _apply(Ht, CORNERS)
                    draw(out, Hc, (0, 255, 0))
                    if corners is not None:
                        draw(out, corners, (0, 0, 255))
                    cv2.imwrite(os.path.join(args.draw, f"{clip}_{i:04d}.jpg"), cv2.resize(out, (960, 540)))
            i += 1
        cap.release()
        es = [e for _, e, _ in errs]
        all_err += es
        worst = max(errs, key=lambda r: r[1]) if errs else None
        print(f"{clip:12s} found {len(errs)}/{len(errs) + len(misses)}  "
              f"median {np.median(es) if es else float('nan'):7.1f} px  "
              f"max {max(es) if es else float('nan'):7.1f} px"
              + (f"  (worst frame {worst[0]}, {worst[2]})" if worst else ""))
    if all_err:
        a = np.array(all_err)
        print(f"\nALL  found {n_found}/{n_frames}  median {np.median(a):.1f} px  "
              f"<20px {np.mean(a < 20) * 100:.0f}%  <50px {np.mean(a < 50) * 100:.0f}%  "
              f"mean time {total_t / max(1, n_frames) * 1000:.0f} ms/frame")


if __name__ == "__main__":
    main()
