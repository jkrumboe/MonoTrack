#!/usr/bin/env python3
"""
Fine-tune the TrackNet ball detector on hand-labelled frames of your own
footage (python/label_ball.py), and measure it.

    python python/train_ball.py                 # train, compare, save if better
    python python/train_ball.py --eval-only     # just score the current weights
    python python/train_ball.py --epochs 40 --clips usa_jap_r3,pol_ita_r7

The published weights (WASB-SBDT) were trained on other broadcasts and miss
fast, motion-blurred balls here. Training starts from them, so a few hundred
labelled frames go a long way.

Held out: the last 20% of every labelled clip is never trained on; the
original and the retrained model are both scored on it (precision, recall,
F1 of the ball within 4 heatmap px, ~15 px at 1080p). The new weights are
saved only if they score better, to python/models/tracknetv2_volleyball_finetuned.pth.tar,
which the analyzer then uses instead of the published ones (delete the file
to go back).
"""

import argparse
import glob
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ball_tracknet import (FRAMES_IN, IMAGENET_MEAN, IMAGENET_STD, INPUT_H, INPUT_W, SCORE_THRESHOLD,  # noqa: E402
                           TrackNetV2, _detect_blobs, default_weights_path, finetuned_weights_path,
                           letterbox_affine, published_weights_path)

ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
LABEL_DIR = os.path.join(ROOT, "eval", "ball_labels")
SIGMA = 2.5                  # target blob radius (heatmap px)
TOL = 4.0                    # a detection this close (heatmap px) to the label counts
VAL_FRAC = 0.2


def load_clip(path):
    """Labelled frames of one clip, letterboxed to the model input, plus targets."""
    lab = json.load(open(path, encoding="utf-8"))
    labels = {int(k): v for k, v in lab["frames"].items()}
    video = os.path.join(ROOT, lab["video"])
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    A = letterbox_affine(w, h)
    # Every window of 3 frames holding a label needs its other frames as input too.
    need = {g for f in labels for g in range(f - FRAMES_IN + 1, f + FRAMES_IN) if 0 <= g < n}
    imgs, i = {}, 0
    while i <= max(need):
        ok, f = cap.read()
        if not ok:
            break
        if i in need:
            imgs[i] = cv2.cvtColor(cv2.warpAffine(f, A, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
        i += 1
    # Label positions in heatmap pixels (None = no ball visible).
    pts = {f: (None if v is None else (A[0, 0] * v[0] * w + A[0, 2], A[1, 1] * v[1] * h + A[1, 2]))
           for f, v in labels.items()}
    return {"name": lab["clip"], "imgs": imgs, "pts": pts, "n": n}


def target(pt):
    if pt is None:
        return np.zeros((INPUT_H, INPUT_W), np.float32)
    ys, xs = np.mgrid[0:INPUT_H, 0:INPUT_W]
    return np.exp(-((xs - pt[0]) ** 2 + (ys - pt[1]) ** 2) / (2 * SIGMA ** 2)).astype(np.float32)


def windows(clip, frames):
    """Window starts whose 3 frames are decoded and at least one is labelled among `frames`."""
    out = []
    for s in range(clip["n"] - FRAMES_IN + 1):
        win = range(s, s + FRAMES_IN)
        if all(g in clip["imgs"] for g in win) and any(g in frames for g in win):
            out.append(s)
    return out


def batch(clip, starts, frames, augment):
    """Input tensor (B, 9, H, W), targets (B, 3, H, W), loss mask (B, 3) for unlabelled frames."""
    xs, ys, ms = [], [], []
    for s in starts:
        win = list(range(s, s + FRAMES_IN))
        img = np.stack([clip["imgs"][g] for g in win]).astype(np.float32) / 255.0   # (3, H, W, 3)
        tg = np.stack([target(clip["pts"].get(g)) for g in win])
        mask = np.array([g in frames for g in win], np.float32)
        if augment:
            if random.random() < 0.5:
                img, tg = img[:, :, ::-1], tg[:, :, ::-1]
            img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        xs.append(img.transpose(0, 3, 1, 2).reshape(3 * FRAMES_IN, INPUT_H, INPUT_W))
        ys.append(tg)
        ms.append(mask)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a)))
    return t(xs), t(ys), t(ms)


def wbce(logits, y, mask):
    """TrackNetV2's weighted BCE: hard pixels count more, easy background less."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    loss = -((1 - p) ** 2 * y * torch.log(p) + p ** 2 * (1 - y) * torch.log(1 - p))
    return (loss.mean(dim=(2, 3)) * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def score(model, clips, split, device):
    """Precision / recall / F1 on the given frames per clip, analyzer-style windows (starts at 3k)."""
    model.eval()
    tp = fp = fn = 0
    for c in clips:
        frames = split[c["name"]]
        starts = sorted({f - f % FRAMES_IN for f in frames})
        for s in starts:
            if not all(g in c["imgs"] for g in range(s, s + FRAMES_IN)):
                continue
            x, _, _ = batch(c, [s], frames, augment=False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hm = torch.sigmoid(model(x.to(device)).float())[0].cpu().numpy()
            for j, g in enumerate(range(s, s + FRAMES_IN)):
                if g not in frames:
                    continue
                blobs = _detect_blobs(hm[j], SCORE_THRESHOLD)
                truth = c["pts"][g]
                if blobs:
                    d = math.inf if truth is None else math.hypot(blobs[0][0] - truth[0], blobs[0][1] - truth[1])
                    if d <= TOL:
                        tp += 1
                    else:
                        fp += 1
                        fn += truth is not None
                elif truth is not None:
                    fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec), "frames": tp + fn}


def load_model(path, device):
    model = TrackNetV2()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--clips", help="comma-separated clip names (default: every labelled clip)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--labels", default=LABEL_DIR, help=f"label folder (default {LABEL_DIR})")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.labels, "*.json")))
    if args.clips:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in args.clips.split(",")]
    if not paths:
        sys.exit(f"No ball labels in {args.labels}: label some with python/label_ball.py first")
    clips = [load_clip(p) for p in paths]
    train, val = {}, {}
    for c in clips:
        fs = sorted(c["pts"])
        cut = fs[int(len(fs) * (1 - VAL_FRAC))] if len(fs) >= 5 else fs[-1] + 1
        train[c["name"]] = {f for f in fs if f < cut}
        val[c["name"]] = {f for f in fs if f >= cut}
        print(f"{c['name']}: {len(fs)} labelled ({sum(1 for f in fs if c['pts'][f])} with ball), "
              f"{len(train[c['name']])} train / {len(val[c['name']])} held out")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current = default_weights_path()
    model = load_model(current, device)
    base = score(model, clips, val, device)
    print(f"current weights ({os.path.basename(current)}) on held-out frames: "
          f"P {base['precision']:.2f}  R {base['recall']:.2f}  F1 {base['f1']:.2f}  ({base['frames']} ball frames)")
    if args.eval_only:
        return

    # Always fine-tune from the published weights, so repeated runs don't drift.
    model = load_model(published_weights_path(), device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    items = [(c, s) for c in clips for s in windows(c, train[c["name"]])]
    if not items:
        sys.exit("No complete 3-frame windows among the training labels")
    best, best_state = base["f1"], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(items)
        total = 0.0
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            xs, ys, ms = zip(*(batch(c, [s], train[c["name"]], augment=True) for c, s in chunk))
            x, y, m = torch.cat(xs).to(device), torch.cat(ys).to(device), torch.cat(ms).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = wbce(logits.float(), y, m)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(chunk)
        s = score(model, clips, val, device)
        mark = ""
        if s["f1"] > best:
            best, best_state = s["f1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best"
        print(f"epoch {epoch:3d}  loss {total / len(items):.4f}  held-out P {s['precision']:.2f} "
              f"R {s['recall']:.2f} F1 {s['f1']:.2f}{mark}")

    if best_state is None:
        print("No improvement over the current weights; nothing saved")
        return
    out = finetuned_weights_path()
    torch.save({"model_state_dict": best_state}, out)
    print(f"Saved {out} (held-out F1 {base['f1']:.2f} -> {best:.2f}); the analyzer uses it from now on")


if __name__ == "__main__":
    main()
