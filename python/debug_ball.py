#!/usr/bin/env python3
"""
Ball-tracking debugger: runs the exact analyzer ball pipeline on a clip and
shows, stage by stage, where the ball is found or lost.

    python python/debug_ball.py match.mp4
    python python/debug_ball.py match.mp4 --start 600 --frames 300
    python python/debug_ball.py match.mp4 --threshold 0.3     # is it just gated?

Writes to <video>_balldebug/ (or --out):
  overlay.mp4   left: frame with every stage drawn, right: model input + heatmap
  timeline.png  heatmap peak per frame vs threshold, and what each stage kept
  frames.csv    one row per frame, every stage, for spreadsheets / plotting
and prints a diagnosis that names the stage losing the ball.

Pipeline stages (same code the app runs, from analyzer.py):
  1. heatmap  TrackNet peak response per frame        (does the model fire?)
  2. raw      blobs above threshold                   (candidates)
  3. tracker  _BallTracker motion gate picks one      (accepted)
  4. outlier  _reject_ball_outliers                   (cleaned)
  5. final    _interpolate_ball_path: short fragments dropped, gaps filled
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ball_tracknet as bt  # noqa: E402
import analyzer as az       # noqa: E402

# BGR colours for each stage
C_RAW = (0, 255, 255)       # yellow: every raw candidate
C_TRACK = (0, 255, 0)       # green: accepted by tracker and kept
C_REJECT = (0, 0, 255)      # red: accepted by tracker, then removed by filters
C_INTERP = (255, 160, 0)    # blue: interpolated position


def check(ok, msg):
    print(f"  [{'OK ' if ok else 'BAD'}] {msg}")
    return ok


def environment_checks(weights):
    print("\n== Environment")
    check(bt.HAS_TORCH, "torch installed")
    if not bt.HAS_TORCH:
        return False
    import torch
    check(True, f"torch {torch.__version__}, cuda={'yes' if torch.cuda.is_available() else 'no (CPU will be slow)'}")
    path = weights or bt.default_weights_path()
    if not check(os.path.isfile(path), f"weights at {path}"):
        print("        -> run once with internet + gdown to download, or pass --weights")
        return False
    size_mb = os.path.getsize(path) / 1e6
    check(size_mb > 40, f"weights size {size_mb:.1f} MB (expect ~45 MB; tiny = failed/HTML download)")
    return True


def preprocessing_checks(det, frame):
    """Verify the model input looks like what WASB trained on."""
    print("\n== Preprocessing")
    h, w = frame.shape[:2]
    inp = det._preprocess([frame] * 3)[0].cpu().numpy()
    m, s = float(inp.mean()), float(inp.std())
    # ImageNet-normalised natural images sit around mean ~0, std ~1.
    check(-1.0 < m < 1.0 and 0.4 < s < 2.0,
          f"input tensor mean={m:.2f} std={s:.2f} (ImageNet-normalised -> ~0 / ~1; "
          f"raw /255 pixels would be ~0.4 / ~0.25)")
    k = det.affine[0, 0]
    check(abs(w / h - 16 / 9) < 0.05,
          f"frame {w}x{h}, aspect {w / h:.2f} (weights trained on 16:9 broadcast; "
          f"other aspects are letterboxed)")
    ball_px = 0.22 / 18.0 * w * k  # rough: ball ~1.2% of a 18 m-wide view
    print(f"        model sees the frame at {w * k:.0f}x{h * k:.0f}; "
          f"a ball across the court is roughly {ball_px:.1f} px at that scale")


def run(args):
    if not environment_checks(args.weights):
        return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n== Video\n  {args.video}\n  {w}x{h} @ {fps:.1f} fps, {total} frames")
    if fps > 40:
        print(f"  [WARN] {fps:.0f} fps: TrackNet was trained on ~25-30 fps, so the ball "
              f"moves less between the 3 input frames. Try --stride 2.")

    det = bt.TrackNetBallDetector(model_path=args.weights, threshold=args.threshold)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, first = cap.read()
    if not ok:
        print("Could not read the start frame")
        return 1
    preprocessing_checks(det, first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    out_dir = args.out or os.path.splitext(args.video)[0] + "_balldebug"
    os.makedirs(out_dir, exist_ok=True)

    # ── Pass 1: run the model, keep per-frame records and heatmaps ─────────
    print(f"\n== Running TrackNet (threshold {args.threshold}, stride {args.stride})")
    rows = {}            # frame_idx -> dict
    heatmaps = {}        # frame_idx -> (288, 512) float16
    model_inputs = {}    # frame_idx -> letterboxed BGR the model saw
    tracker = az._BallTracker()
    accepted = {}
    buf, idx = [], []
    fi = args.start
    end = args.start + args.frames if args.frames else total
    t0 = time.time()

    def flush(buf, idx):
        n = len(idx)
        win = buf + [buf[-1]] * (3 - n)
        per_frame = det.detect_window(win)
        hms = det.last_heatmaps
        for j in range(n):
            f = idx[j]
            cands = per_frame[j]
            rows[f] = {"frame": f, "peak": float(hms[j].max()), "cands": cands}
            heatmaps[f] = hms[j].astype(np.float16)
            model_inputs[f] = cv2.warpAffine(win[j], det.affine, (bt.INPUT_W, bt.INPUT_H))
            d = tracker.update(f, cands)
            if d is not None:
                accepted[f] = d

    while fi < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - args.start) % args.stride == 0:
            buf.append(frame)
            idx.append(fi)
            if len(buf) == 3:
                flush(buf, idx)
                buf, idx = [], []
        fi += 1
        if (fi - args.start) % 100 == 0:
            print(f"  frame {fi}/{end}  ({(fi - args.start) / (time.time() - t0):.1f} fps)")
    if buf:
        flush(buf, idx)
    cap.release()

    # ── Pass 2: the same post-processing the analyzer runs ─────────────────
    cleaned = az._reject_ball_outliers(dict(accepted))
    final = az._interpolate_ball_path(cleaned)
    kept_real = {f for f, v in final.items() if not v["interpolated"]}

    # ── Report ────────────────────────────────────────────────────────────
    frames = sorted(rows)
    n = len(frames)
    peaks = np.array([rows[f]["peak"] for f in frames])
    n_fire = int((peaks > args.threshold).sum())
    n_cands = sum(len(rows[f]["cands"]) for f in frames)
    n_multi = sum(1 for f in frames if len(rows[f]["cands"]) > 1)
    n_interp = sum(1 for v in final.values() if v["interpolated"])

    print("\n== Stage funnel (frames with a ball at each stage)")
    funnel = [
        ("1 heatmap peak > threshold", n_fire),
        ("2 raw candidates (frames)", sum(1 for f in frames if rows[f]["cands"])),
        ("3 accepted by tracker", len(accepted)),
        ("4 after outlier rejection", len(cleaned)),
        ("5 final detected (fragments >= 5)", len(kept_real)),
        ("  + interpolated", n_interp),
    ]
    for name, v in funnel:
        bar = "#" * int(40 * v / max(n, 1))
        print(f"  {name:<36} {v:>6} / {n:<6} {100 * v / max(n, 1):5.1f}%  {bar}")
    pct = np.percentile(peaks, [50, 90, 99]) if n else [0, 0, 0]
    print(f"\n  heatmap peak: max={peaks.max() if n else 0:.3f}  median={pct[0]:.3f}  "
          f"p90={pct[1]:.3f}  p99={pct[2]:.3f}")
    print(f"  candidates: {n_cands} total, {n_multi} frames with >1 (distractors)")

    print("\n== Diagnosis")
    diagnose(n, peaks, n_fire, rows, accepted, cleaned, kept_real, args.threshold)

    write_csv(out_dir, frames, rows, accepted, cleaned, final)
    write_timeline(out_dir, frames, peaks, accepted, cleaned, final, args.threshold)
    if not args.no_video:
        write_overlay(out_dir, args.video, fps / args.stride, frames, rows, heatmaps,
                      model_inputs, accepted, cleaned, final, w, h)
    print(f"\n== Output\n  {out_dir}")
    return 0


def diagnose(n, peaks, n_fire, rows, accepted, cleaned, kept_real, thr):
    if n == 0:
        print("  No frames processed.")
        return
    max_peak = float(peaks.max())
    fire = n_fire / n
    if max_peak < 0.1:
        print("  MODEL NEVER FIRES (max peak < 0.1). The network is not seeing a ball at all.")
        print("   - Check the right half of overlay.mp4: is the ball visible in the model input?")
        print("   - If the 'Preprocessing' check above says BAD, the input is wrong.")
        print("   - Otherwise the footage is out of domain: the volleyball weights were trained on")
        print("     1280x720 side-on broadcast video. Phone footage from behind the court, very")
        print("     zoomed-out shots, or 60 fps (try --stride 2) can all defeat it.")
        return
    if max_peak < thr:
        print(f"  MODEL FIRES WEAKLY (max peak {max_peak:.2f} < threshold {thr}).")
        print(f"   - Re-run with --threshold {max(0.1, round(max_peak * 0.7, 2))} and check overlay.mp4:")
        print("     if the heatmap blob is on the ball, lower SCORE_THRESHOLD in ball_tracknet.py.")
        return
    if fire < 0.3:
        print(f"  Model fires on only {fire * 100:.0f}% of frames.")
        print("   - Normal if much of the clip is between rallies / ball off-screen.")
        print("   - Check timeline.png: gaps during live play mean missed detections.")
    multi = sum(1 for f in rows if len(rows[f]["cands"]) > 1) / n
    if multi > 0.2:
        print(f"  {multi * 100:.0f}% of frames have several candidates: distractors (heads, other")
        print("   balls, logos). The tracker has to pick; watch for yellow dots away from the ball.")
    raw_frames = sum(1 for f in rows if rows[f]["cands"])
    if raw_frames and len(accepted) < 0.6 * raw_frames:
        print(f"  TRACKER REJECTS {raw_frames - len(accepted)} of {raw_frames} frames with candidates.")
        print(f"   - A track only starts on conf >= {az._BALL_CONF_TRUST}; steps > {az._BALL_MAX_STEP}")
        print("     (fraction of frame width per frame) are rejected. Fast spikes may exceed it:")
        print("     raise _BALL_MAX_STEP in analyzer.py if yellow dots on the ball are not green.")
    if accepted and len(cleaned) < 0.85 * len(accepted):
        print(f"  OUTLIER FILTER removed {len(accepted) - len(cleaned)} accepted points (red in overlay).")
        print("   - If red points are on the ball, the tolerance in _reject_ball_outliers is too tight.")
    if cleaned and len(kept_real) < 0.85 * len(cleaned):
        print(f"  FRAGMENT FILTER dropped {len(cleaned) - len(kept_real)} points in runs shorter than 5 frames.")
    if fire >= 0.3 and len(kept_real) >= 0.6 * n_fire:
        print("  Pipeline looks healthy. Verify positions visually in overlay.mp4.")


def write_csv(out_dir, frames, rows, accepted, cleaned, final):
    path = os.path.join(out_dir, "frames.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "peak", "n_cands", "best_x", "best_y", "best_conf",
                     "tracker", "after_outlier", "final_x", "final_y", "final_kind"])
        for f in frames:
            r = rows[f]
            b = r["cands"][0] if r["cands"] else None
            fin = final.get(f)
            wr.writerow([
                f, f"{r['peak']:.4f}", len(r["cands"]),
                f"{b[0]:.4f}" if b else "", f"{b[1]:.4f}" if b else "", f"{b[2]:.3f}" if b else "",
                int(f in accepted), int(f in cleaned),
                f"{fin['pos'][0]:.4f}" if fin else "", f"{fin['pos'][1]:.4f}" if fin else "",
                ("interp" if fin["interpolated"] else "detect") if fin else "",
            ])


def write_timeline(out_dir, frames, peaks, accepted, cleaned, final, thr):
    """Heatmap peak per frame plus one lane per pipeline stage."""
    n = len(frames)
    if n == 0:
        return
    W, plot_h, lane_h = max(800, min(n, 3000)), 200, 18
    H = plot_h + 4 * lane_h + 30
    img = np.full((H, W, 3), 255, np.uint8)
    peak_of = dict(zip(frames, peaks))
    xs = lambda i: int(i * (W - 1) / max(n - 1, 1))  # noqa: E731
    ty = plot_h - int(thr * plot_h)
    cv2.line(img, (0, ty), (W, ty), (0, 0, 200), 1)
    cv2.putText(img, f"threshold {thr}", (5, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    pts = np.array([[xs(i), plot_h - int(min(p, 1.0) * plot_h)] for i, p in enumerate(peaks)], np.int32)
    cv2.polylines(img, [pts], False, (60, 60, 60), 1)
    lanes = [
        ("raw", lambda f: peak_of[f] > thr, C_RAW),
        ("tracker", lambda f: f in accepted, C_TRACK),
        ("cleaned", lambda f: f in cleaned, (0, 160, 0)),
        ("final", lambda f: f in final, None),
    ]
    for li, (name, fn, col) in enumerate(lanes):
        y0 = plot_h + 5 + li * lane_h
        for i, f in enumerate(frames):
            if fn(f):
                c = col or (C_INTERP if final[f]["interpolated"] else C_TRACK)
                cv2.line(img, (xs(i), y0), (xs(i), y0 + lane_h - 4), c, 1)
        cv2.putText(img, name, (3, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, f"frames {frames[0]}-{frames[-1]}  (top: heatmap peak 0..1)",
                (5, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(out_dir, "timeline.png"), img)


def write_overlay(out_dir, video, fps, frames, rows, heatmaps, model_inputs,
                  accepted, cleaned, final, w, h):
    print("\n== Writing overlay.mp4")
    disp_h = 540
    sc = disp_h / h
    disp_w = int(w * sc)
    side_w = int(bt.INPUT_W * disp_h / 2 / bt.INPUT_H)
    out_w = disp_w + side_w
    vw = cv2.VideoWriter(os.path.join(out_dir, "overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, disp_h))
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    frame_set = set(frames)
    trail = []
    fi = frames[0]
    while fi <= frames[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if fi not in frame_set:
            fi += 1
            continue
        r = rows[fi]
        img = cv2.resize(frame, (disp_w, disp_h))
        P = lambda x, y: (int(x * disp_w), int(y * disp_h))  # noqa: E731

        fin = final.get(fi)
        if fin:
            trail.append(P(*fin["pos"]))
        else:
            trail.append(None)
        trail = trail[-20:]
        for a, b in zip(trail, trail[1:]):
            if a and b:
                cv2.line(img, a, b, (255, 255, 255), 1)
        for c in r["cands"]:
            cv2.circle(img, P(c[0], c[1]), 12, C_RAW, 1)
            cv2.putText(img, f"{c[2]:.2f}", (P(c[0], c[1])[0] + 14, P(c[0], c[1])[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_RAW, 1)
        if fi in accepted:
            a = accepted[fi]
            cv2.circle(img, P(a[0], a[1]), 6, C_TRACK if fi in cleaned else C_REJECT, 2)
        if fin and fin["interpolated"]:
            cv2.circle(img, P(*fin["pos"]), 6, C_INTERP, 2)

        stage = ("DETECTED" if fin and not fin["interpolated"] else
                 "INTERPOLATED" if fin else
                 "filtered out" if fi in accepted else
                 "tracker rejected" if r["cands"] else
                 "no candidate")
        cv2.rectangle(img, (0, 0), (disp_w, 22), (0, 0, 0), -1)
        cv2.putText(img, f"frame {fi}  peak {r['peak']:.2f}  cands {len(r['cands'])}  -> {stage}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Right column: what the model saw, and its heatmap on top of it.
        mi = model_inputs[fi]
        hm = heatmaps[fi].astype(np.float32)
        hm_col = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(mi, 0.5, hm_col, 0.5, 0)
        half = disp_h // 2
        side = np.vstack([cv2.resize(mi, (side_w, half)), cv2.resize(blend, (side_w, disp_h - half))])
        cv2.putText(side, "model input", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(side, "heatmap", (5, half + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        vw.write(np.hstack([img, side]))
        fi += 1
    cap.release()
    vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--frames", type=int, default=600, help="frames to process (0 = all)")
    ap.add_argument("--threshold", type=float, default=bt.SCORE_THRESHOLD)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (2 for 60 fps)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true", help="skip overlay.mp4")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
