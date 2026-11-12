#!/usr/bin/env python3
"""
Compare and tune ball detectors without re-running the whole analyzer.

A full export takes minutes a clip, nearly all of it on court and player
detection, which a ball-detector change does not affect. So:

    # 1. Heatmaps once per network and stride (compressed, kept in runs/ball_cache)
    python python/bench_ball.py cache --detector vballnet --stride 3

    # 2. Rebuild fixtures from them in seconds: ball fields replaced, players,
    #    camera motion and court kept from the base fixtures
    python python/bench_ball.py replay --detector vballnet --stride 3 \
        --threshold 0.3 --out runs/fx_vball
    EVAL_FIXTURES_DIR=runs/fx_vball npm run eval

Replay uses the analyzer's own tracker, outlier rejection and interpolation.
Check a final choice with a real export_fixture.py run.
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
CACHE = os.path.join(ROOT, "runs", "ball_cache")
BALL_KEYS = {"ball_img_pos", "ball_interpolated", "ball_conf", "ball_bbox", "ball_diameter_px",
             "players_near_ball", "ball_depth_estimate", "ball_ref_pos", "ball_ref_diameter_px",
             "players_ref", "players_arms", "players_arms_ref"}
VBALL_GRID = "VballNetGridV2b_seq9_grayscale_20260909_001145.onnx"


def make_detector(name):
    name = name.replace("+stab", "")
    if name == "tracknetv2":
        from ball_tracknet import TrackNetBallDetector
        return TrackNetBallDetector()
    if name == "tracknetv3":
        from ball_tracknetv3 import TrackNetV3BallDetector
        return TrackNetV3BallDetector()
    from ball_vballnet import VballNetBallDetector, ensure_weights
    if name == "vballnet":
        return VballNetBallDetector()
    if name == "vballnet-grid":
        return VballNetBallDetector(model_path=ensure_weights(VBALL_GRID))
    raise ValueError(name)


def clips(only=""):
    keep = set(filter(None, only.split(",")))
    out = []
    for lp in sorted(glob.glob(os.path.join(ROOT, "eval", "labels", "*.json"))):
        c = os.path.splitext(os.path.basename(lp))[0]
        if (not keep or c in keep) and os.path.exists(os.path.join(ROOT, "eval", "videos", c + ".mp4")):
            out.append(c)
    return out


def cache_path(detector, stride, clip):
    # "vballnet+stab" is the same network on motion-stabilised windows.
    return os.path.join(CACHE, f"{detector}_s{stride}", clip + ".npz")


def cmd_cache(args):
    from ball_tracknet import WindowStream
    det = make_detector(args.detector)
    stride = args.stride or det.window
    for clip in clips(args.only):
        out = cache_path(args.detector, stride, clip)
        if os.path.exists(out):
            continue
        video = os.path.join(ROOT, "eval", "videos", clip + ".mp4")
        motion = None
        if "+stab" in args.detector:
            frames = json.load(open(os.path.join(ROOT, "eval", "fixtures", clip + ".json"), encoding="utf-8"))["frames"]
            Hs = {f["frame"]: np.array(f["ref_from_frame"], np.float64).reshape(3, 3)
                  for f in frames if f.get("ref_from_frame")}
            motion = Hs.get
        if hasattr(det, "begin_clip"):
            det.begin_clip(video, motion)
        stream = WindowStream(det, stride, motion)
        cap = cv2.VideoCapture(video)
        hms, t0 = {}, time.time()

        def keep(items):
            for f, _frame, hm in items:
                hms[f] = (np.clip(hm, 0, 1) * 255).astype(np.uint8)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            keep(stream.push(frame))
        keep(stream.flush())
        w, h = int(cap.get(3)), int(cap.get(4))
        cap.release()
        # Below 0.05 is far under any threshold tried; zeros compress away.
        arr = np.stack([hms[f] for f in sorted(hms)])
        arr[arr < 13] = 0
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, hm=arr, size=[w, h])
        print(f"{args.detector} s{stride} {clip}: {len(hms)} frames, "
              f"{len(hms) / (time.time() - t0):.1f} fps", flush=True)


class _ToNorm:
    """Heatmap pixels -> normalised frame, as the given detector maps them."""

    def __init__(self, detector, hm_shape, w, h):
        H, W = hm_shape
        if detector.startswith("tracknetv2"):
            from ball_tracknet import letterbox_affine, heatmap_to_image_norm
            aff = letterbox_affine(w, h)
            self.fn = lambda x, y: heatmap_to_image_norm(x, y, aff, w, h)
        else:
            self.fn = lambda x, y: (x / W, y / H)


def load_candidates(detector, stride, clip, threshold):
    from ball_tracknet import candidates_from_heatmap
    d = np.load(cache_path(detector, stride, clip))
    hm = d["hm"].astype(np.float32) / 255.0
    w, h = (int(v) for v in d["size"])
    to = _ToNorm(detector, hm.shape[1:], w, h).fn
    return [candidates_from_heatmap(hm[i], threshold, to) for i in range(len(hm))]


def fuse(per_source):
    """Candidates of several networks, frame by frame, nearby ones merged.

    Two networks agreeing on a spot is better evidence than either's own
    confidence, so the merged candidate's confidence is lifted towards 1.
    """
    out = []
    for frame_cands in zip(*per_source):
        merged = []
        for c in sorted((c for cs in frame_cands for c in cs), key=lambda c: -c[2]):
            for m in merged:
                if abs(m[0] - c[0]) < 0.012 and abs(m[1] - c[1]) < 0.02:
                    m[2] = round(1 - (1 - m[2]) * (1 - c[2]), 4)
                    break
            else:
                merged.append(list(c))
        out.append(sorted(merged, key=lambda c: -c[2]))
    return out


def replay(clip, cands, base_dir, out_dir, params):
    """Rebuild one fixture's ball fields from per-frame candidates."""
    import analyzer as an
    from ball_tracknet import measure_ball_diameter
    from camera_motion import apply_h, local_scale, map_box

    for k, v in params.get("analyzer", {}).items():
        setattr(an, k, v)
    fx = json.load(open(os.path.join(base_dir, clip + ".json"), encoding="utf-8"))
    video = os.path.join(ROOT, "eval", "videos", clip + ".mp4")
    frames = fx["frames"]
    fps = fx["video"]["fps"]
    tracker = an._BallTracker()
    raw = {}
    for i, c in enumerate(cands):
        det = tracker.update(i, c)
        if det is not None:
            raw[i] = det
    raw = an._reject_ball_outliers(raw)
    floor = params.get("conf_floor")
    if floor:
        for det in raw.values():
            det[2] = max(det[2], floor)
    path = an._interpolate_ball_path(raw)

    all_players = {f["frame"]: f["players"] for f in frames if "players" in f}
    arms = cached_arms(clip, video, raw, all_players, fps, an)
    cap = cv2.VideoCapture(video)
    want = sorted(raw)
    diam = {}
    i = 0
    for f in want:
        while i < f:
            cap.grab()
            i += 1
        ok, img = cap.read()
        i += 1
        if ok:
            d = measure_ball_diameter(img, raw[f][0], raw[f][1])
            if d is not None:
                diam[f] = d
    cap.release()
    aspect = fx["video"]["width"] / fx["video"]["height"]
    for fr in frames:
        for k in BALL_KEYS:
            fr.pop(k, None)
        fi = fr["frame"]
        H = np.array(fr["ref_from_frame"], np.float64).reshape(3, 3) if "ref_from_frame" in fr else None
        if fi in arms:
            fr["players_arms"] = arms[fi]
            if H is not None:
                fr["players_arms_ref"] = [[apply_h(H, k[:2]) + [k[2]] for k in a] for a in arms[fi]]
        bd = path.get(fi)
        if bd is None:
            continue
        fr["ball_img_pos"] = bd["pos"]
        fr["ball_interpolated"] = bd["interpolated"]
        fr["ball_conf"] = bd["conf"]
        if bd["bbox"] is not None:
            fr["ball_bbox"] = bd["bbox"]
        if not bd["interpolated"]:
            if fi in diam:
                fr["ball_diameter_px"] = diam[fi]
            if fi in all_players:
                fr["players_near_ball"] = an._players_near_ball(all_players[fi], aspect, raw[fi][0], raw[fi][1])
        if H is not None:
            fr["ball_ref_pos"] = [round(v, 6) for v in apply_h(H, bd["pos"])]
            if "ball_diameter_px" in fr:
                fr["ball_ref_diameter_px"] = round(fr["ball_diameter_px"] * local_scale(H, bd["pos"]), 2)
            if "players_near_ball" in fr:
                fr["players_ref"] = [map_box(H, b) for b in fr["players_near_ball"]]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, clip + ".json"), "w", encoding="utf-8") as fh:
        json.dump(fx, fh, separators=(",", ":"))
    return {"tracked": len(raw), "with_size": len(diam), "frames": len(frames)}


def cached_arms(clip, video, raw, all_players, fps, an):
    """_arms_in_gaps with each frame's pose result remembered between replays."""
    seen = sorted(f for f, d in raw.items() if d[2] >= an._ARMS_MIN_CONF)
    want = {f for a, b in zip(seen, seen[1:]) if 1 < b - a <= an._ARMS_GAP_S * fps for f in range(a + 1, b)}
    want &= set(all_players)
    path = os.path.join(CACHE, "arms", clip + ".pkl")
    store = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    missing = sorted(want - set(store))
    if missing:
        pose = an._get_pose_model()
        cap = cv2.VideoCapture(video)
        for fi in missing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok and pose is not None:
                boxes, pose_arms = an._detect_players(frame, pose)
                store[fi] = an._match_arms(all_players[fi], boxes, pose_arms or [])
        cap.release()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}"   # sweeps replay clips in parallel
        with open(tmp, "wb") as fh:
            pickle.dump(store, fh)
        try:
            os.replace(tmp, path)
        except PermissionError:   # another replay holds it open; only a cache
            os.remove(tmp)
    return {f: store[f] for f in want if f in store}


def cmd_replay(args):
    base = args.base or os.path.join(ROOT, "eval", "fixtures")
    params = json.loads(args.analyzer_params) if args.analyzer_params else {}
    for clip in clips(args.only):
        if args.sources:
            # "vballnet:9:0.3,tracknetv2:3:0.5"
            per = []
            for src in args.sources.split(","):
                d, st, th = src.split(":")
                per.append(load_candidates(d, int(st), clip, float(th)))
            n = min(len(p) for p in per)
            cands = fuse([p[:n] for p in per])
        else:
            stride = args.stride or make_stride(args.detector)
            cands = load_candidates(args.detector, stride, clip, args.threshold)
        print(clip, replay(clip, cands, base, args.out, {"analyzer": params, "conf_floor": args.conf_floor}), flush=True)


def make_stride(name):
    return {"tracknetv2": 3, "tracknetv3": 8, "vballnet": 9, "vballnet-grid": 9}[name.replace("+stab", "")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cache", "replay"])
    ap.add_argument("--detector", required=True)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--only", default="")
    ap.add_argument("--out")
    ap.add_argument("--base", help="fixtures to take players, motion and court from (default eval/fixtures)")
    ap.add_argument("--sources", help="fuse networks: detector:stride:threshold,...")
    ap.add_argument("--conf-floor", type=float, help="lift tracked detections' confidence to at least this")
    ap.add_argument("--analyzer-params", help='JSON of analyzer constants to override, e.g. {"_BALL_MAX_STEP":0.2}')
    args = ap.parse_args()
    (cmd_cache if args.command == "cache" else cmd_replay)(args)


if __name__ == "__main__":
    main()
