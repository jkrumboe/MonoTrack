#!/usr/bin/env python3
"""
Train your own player detector from your own footage, with no hand labelling.

A large, slow detector (the teacher) labels frames of your videos; a small,
fast one (the student) is trained on those labels. The student then sees only
volleyball from your camera position, so it can be smaller than a general
detector and still find the players, and it learns to leave out the crowd.

    # 1. Label: every 5th frame of the videos, by the teacher
    python python/train_players.py label "eval/videos/*.mp4" "my_recordings/*.mp4"

    # 2. Train the student on them (about an hour on a laptop GPU)
    python python/train_players.py train

    # 3. Compare with the current detector on the labelled eval clips
    python python/bench_players.py mine python/models/players.pt --stride 2
    EVAL_FIXTURES_DIR=runs/fx_players_mine npm run eval

The analyzer uses python/models/players.pt when it exists (delete the file to
go back to the stock detector); PLAYER_MODEL=<weights> picks another.

label keeps only people on or near the court when the video has an eval
fixture with a court (eval/fixtures/<clip>.json): spectators, coaches and
line judges far from play are then background the student learns to skip.
Videos without a fixture keep everyone the teacher finds.

The clips held out in EVAL.md (pol_ita_r3, pol_ita_r6, usa_jap_r2) and any
video named with --val go to the validation set and are never trained on.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
DATASET = os.path.join(ROOT, "runs", "players_ds")
OUT_WEIGHTS = os.path.join(HERE, "models", "players.pt")
HOLDOUT = {"pol_ita_r3", "pol_ita_r6", "usa_jap_r2"}
# In play: the court (9 x 18 m) plus the free zone and a server's run-up, as
# the rally solver bounds it (ballReconstruction.ts).
IN_PLAY_X = 4.5 + 6.0
IN_PLAY_Z = 9.0 + 9.0
_COURT = np.array([[-4.5, -9.0], [4.5, -9.0], [4.5, 9.0], [-4.5, 9.0]], np.float64)


def court_filter(clip):
    """frame index, box -> is this person on or near the court; None without a fixture court."""
    path = os.path.join(ROOT, "eval", "fixtures", clip + ".json")
    if not os.path.exists(path):
        return None
    fx = json.load(open(path, encoding="utf-8"))
    if not fx.get("courtCorners"):
        return None
    to_court, _ = cv2.findHomography(np.array(fx["courtCorners"], np.float64), _COURT)
    ref_from = {f["frame"]: np.array(f["ref_from_frame"], np.float64).reshape(3, 3)
                for f in fx["frames"] if f.get("ref_from_frame")}
    moving = bool(fx.get("cameraMoving"))

    def on_court(fi, box):
        feet = np.array([(box[0] + box[2]) / 2, box[3], 1.0])
        if moving:
            H = ref_from.get(fi)
            if H is None:
                return True   # no trusted camera motion on this frame: keep
            feet = H @ feet
        p = to_court @ (feet / feet[2])
        if abs(p[2]) < 1e-9:
            return False
        x, z = p[0] / p[2], p[1] / p[2]
        return abs(x) <= IN_PLAY_X and abs(z) <= IN_PLAY_Z

    return on_court


def cmd_label(args):
    from ultralytics import YOLO
    videos = sorted({v for pattern in args.videos for v in glob.glob(pattern)})
    if not videos:
        sys.exit("No videos match")
    if args.fresh and os.path.isdir(DATASET):
        shutil.rmtree(DATASET)
    teacher = YOLO(args.teacher)
    val_names = HOLDOUT | set(args.val)
    counts = {"train": 0, "val": 0}
    for video in videos:
        clip = os.path.splitext(os.path.basename(video))[0]
        split = "val" if clip in val_names else "train"
        keep = None if args.everyone else court_filter(clip)
        img_dir = os.path.join(DATASET, "images", split)
        lab_dir = os.path.join(DATASET, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        cap = cv2.VideoCapture(video)
        fi = n = n_boxes = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % args.every == 0:
                h, w = frame.shape[:2]
                r = teacher.predict(frame, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False,
                                    augment=args.tta)[0]
                lines = []
                for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                    box = [x1 / w, y1 / h, x2 / w, y2 / h]
                    if keep is not None and not keep(fi, box):
                        continue
                    lines.append(f"0 {(box[0] + box[2]) / 2:.6f} {(box[1] + box[3]) / 2:.6f} "
                                 f"{box[2] - box[0]:.6f} {box[3] - box[1]:.6f}")
                stem = f"{clip}_{fi:06d}"
                cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines))
                n += 1
                n_boxes += len(lines)
            fi += 1
        cap.release()
        counts[split] += n
        print(f"{clip}: {n} frames, {n_boxes / max(n, 1):.1f} players a frame ({split}"
              f"{', court only' if keep else ', everyone'})", flush=True)
    with open(os.path.join(DATASET, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {DATASET.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n")
    print(f"{counts['train']} training and {counts['val']} validation frames in {DATASET}")
    if not counts["val"]:
        print("No validation frames: name a video with --val, or include the eval holdout clips.")


def cmd_train(args):
    from ultralytics import YOLO
    data = os.path.join(DATASET, "data.yaml")
    if not os.path.exists(data):
        sys.exit("No dataset yet: run the label step first")
    if not glob.glob(os.path.join(DATASET, "images", "val", "*.jpg")):
        sys.exit("The dataset has no validation frames (see the label step)")
    student = YOLO(args.student)
    # rect: frames keep their 16:9 shape, as at analysis time. No mosaic in
    # the last epochs and no vertical flips: players are upright and small.
    student.train(data=data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, rect=True,
                  single_cls=True, project=os.path.join(ROOT, "runs", "players_train"), name="student",
                  exist_ok=True, workers=args.workers, patience=10, flipud=0.0, verbose=False)
    best = os.path.join(ROOT, "runs", "players_train", "student", "weights", "best.pt")
    if not os.path.exists(best):
        sys.exit("Training produced no weights")
    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
    shutil.copyfile(best, OUT_WEIGHTS)
    print(f"Saved {os.path.normpath(OUT_WEIGHTS)}. The analyzer now uses it; check it with\n"
          f"  python python/bench_players.py mine {os.path.relpath(OUT_WEIGHTS, ROOT)} --stride 2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label", help="label frames of videos with the teacher")
    lab.add_argument("videos", nargs="+", help="video files or patterns")
    lab.add_argument("--teacher", default="yolov8x.pt", help="detector that makes the labels (downloaded if missing)")
    lab.add_argument("--every", type=int, default=5, help="label every Nth frame")
    lab.add_argument("--imgsz", type=int, default=1280)
    lab.add_argument("--conf", type=float, default=0.35)
    lab.add_argument("--tta", action="store_true", help="test-time augmentation: slower, a few more far players")
    lab.add_argument("--val", nargs="*", default=[], help="clip names (file name without extension) for validation")
    lab.add_argument("--everyone", action="store_true", help="keep people off the court too")
    lab.add_argument("--fresh", action="store_true", help="delete the existing dataset first")
    tr = sub.add_parser("train", help="train the student on the labelled frames")
    tr.add_argument("--student", default="yolov8s.pt", help="weights to start from")
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--imgsz", type=int, default=1280)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    (cmd_label if args.command == "label" else cmd_train)(args)


if __name__ == "__main__":
    main()
