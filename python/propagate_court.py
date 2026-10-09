#!/usr/bin/env python3
"""
Copy a hand-placed court to the other clips of the same match.

    python python/propagate_court.py pol_ita_r1            # to every clip of that match without a court
    python python/propagate_court.py pol_ita_r1 --to pol_ita_r3 pol_ita_r5 --force

All clips of one broadcast come from the same camera standing in the same
spot; only its pan and zoom differ. Such views are related by one homography,
exactly, so a frame of each target clip is matched (ORB + RANSAC) against the
frame of the source clip the court was placed on, and the court points are
mapped across. "Same match" = same YouTube URL in eval/clip_sources.json.

Writes the court into eval/labels/<target>.json (touches are kept) and a check
image per clip in eval/labels/court_checks/ with the court drawn on: the red
lines must sit on the real court lines. Open the clip in label_touches.py to
adjust points if needed.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "eval"))
MATCH_W = 960
MIN_INLIERS = 80
LINES = [((-4.5, -9), (4.5, -9)), ((4.5, -9), (4.5, 9)), ((4.5, 9), (-4.5, 9)), ((-4.5, 9), (-4.5, -9)),
         ((-4.5, 0), (4.5, 0)), ((-4.5, -3), (4.5, -3)), ((-4.5, 3), (4.5, 3))]


def load_json(path, default=None):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return default


def read_frame(video, index):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def ref_from(fixture, frame):
    """Frame -> reference homography (normalised) of a clip; identity when static."""
    for f in fixture.get("frames", []):
        if f["frame"] == frame and f.get("ref_from_frame"):
            return np.array(f["ref_from_frame"], float).reshape(3, 3)
    return np.eye(3)


def to_norm_h(H_px, w, h):
    """Pixel homography (at MATCH_W scale) -> normalised [0,1] coordinates."""
    s = MATCH_W / w
    N = np.diag([w * s, h * s, 1.0])
    Hn = np.linalg.inv(N) @ H_px @ N
    return Hn / Hn[2, 2]


class Matcher:
    def __init__(self):
        self.orb = cv2.ORB_create(4000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, img):
        h, w = img.shape[:2]
        g = cv2.cvtColor(cv2.resize(img, (MATCH_W, int(round(h * MATCH_W / w)))), cv2.COLOR_BGR2GRAY)
        # Ignore the bottom score bar and the top rows: overlays don't move with the camera.
        mask = np.zeros_like(g)
        mask[int(0.05 * g.shape[0]):int(0.85 * g.shape[0]), :] = 255
        return self.orb.detectAndCompute(g, mask)

    def homography(self, fa, fb):
        """H mapping image a -> image b (MATCH_W px), inlier count."""
        (ka, da), (kb, db) = fa, fb
        if da is None or db is None:
            return None, 0
        pairs = self.bf.knnMatch(da, db, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        return (H, int(mask.sum())) if H is not None else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="clip whose labels have a court")
    ap.add_argument("--to", nargs="*", help="target clips (default: same match, no court yet)")
    ap.add_argument("--force", action="store_true", help="replace a court the target already has")
    args = ap.parse_args()

    src_label = load_json(os.path.join(EVAL_DIR, "labels", args.source + ".json"), {})
    court = src_label.get("court")
    if not court or len(court.get("points", [])) < 4:
        sys.exit(f"{args.source} has no court with 4+ points (place it in label_touches.py, key k)")
    src_fx = load_json(os.path.join(EVAL_DIR, "fixtures", args.source + ".json"), {})
    sources = {e["name"]: e for e in load_json(os.path.join(EVAL_DIR, "clip_sources.json"), [])}

    targets = args.to
    if not targets:
        url = sources.get(args.source, {}).get("url")
        if not url:
            sys.exit(f"{args.source} is not in eval/clip_sources.json; name the targets with --to")
        targets = [n for n, e in sources.items() if e["url"] == url and n != args.source]

    # Source points, all expressed on one source frame (the one most were placed on).
    frames = [p["frame"] for p in court["points"]]
    f0 = max(set(frames), key=frames.count)
    to_f0 = np.linalg.inv(ref_from(src_fx, f0))
    src_pts = []
    for p in court["points"]:
        q = cv2.perspectiveTransform(np.float32([[p["pt"]]]), to_f0 @ ref_from(src_fx, p["frame"]))[0, 0]
        src_pts.append(q)
    src_img = read_frame(os.path.join(EVAL_DIR, "videos", args.source + ".mp4"), f0)
    if src_img is None:
        sys.exit(f"Cannot read eval/videos/{args.source}.mp4")
    H_img, W_img = src_img.shape[:2]
    matcher = Matcher()
    f_src = matcher.features(src_img)
    check_dir = os.path.join(EVAL_DIR, "labels", "court_checks")
    os.makedirs(check_dir, exist_ok=True)

    for name in targets:
        lb_path = os.path.join(EVAL_DIR, "labels", name + ".json")
        label = load_json(lb_path, {"clip": name, "touches": []})
        if label.get("court") and not args.force:
            print(f"{name}: has a court already (use --force to replace)")
            continue
        video = os.path.join(EVAL_DIR, "videos", name + ".mp4")
        cap = cv2.VideoCapture(video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        best = (None, 0, None, None)   # frame, inliers, H (src -> tgt, px), image
        for fi in range(0, max(1, n), 8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                break
            H, inl = matcher.homography(f_src, matcher.features(img))
            if H is not None and inl > best[1]:
                best = (fi, inl, H, img)
        cap.release()
        fi, inl, H, img = best
        if H is None or inl < MIN_INLIERS:
            print(f"{name}: no frame matches {args.source} well enough ({inl} inliers): place its court by hand")
            continue
        Hn = to_norm_h(H, W_img, H_img)
        pts = cv2.perspectiveTransform(np.float32([src_pts]), Hn)[0]
        label["court"] = {
            "view": court["view"],
            "propagatedFrom": args.source,
            "points": [{"id": p["id"], "name": p["name"], "world": p["world"], "frame": fi,
                        "pt": [round(float(q[0]), 5), round(float(q[1]), 5)]}
                       for p, q in zip(court["points"], pts)],
        }
        with open(lb_path, "w", encoding="utf-8") as fh:
            json.dump(label, fh, indent=2)

        # Check image: court lines fitted to the mapped points, drawn on the target frame.
        world = np.float32([p["world"] for p in court["points"]])
        Hc, _ = cv2.findHomography(world, np.float32(pts), 0)
        h, w = img.shape[:2]
        for a, b in LINES:
            seg = cv2.perspectiveTransform(np.float32([[a, b]]), Hc)[0] * [w, h]
            cv2.line(img, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"{name} frame {fi}: court from {args.source} ({inl} matches)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(check_dir, name + ".jpg"), cv2.resize(img, (1280, int(1280 * h / w))))
        print(f"{name}: court from {args.source} on frame {fi} ({inl} matched points)")
    print(f"Check images: {os.path.relpath(check_dir)}")


if __name__ == "__main__":
    main()
