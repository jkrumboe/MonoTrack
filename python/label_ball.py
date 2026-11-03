#!/usr/bin/env python3
"""
Label the ball's position, frame by frame, to retrain the ball detector
(python/train_ball.py).

    python python/label_ball.py eval/videos/clip.mp4

TrackNet misses fast, motion-blurred balls (a spike is a streak several ball
widths long per frame), so labels matter most exactly where it fails. The
tool pre-marks suggestions so most frames take one key:
  yellow ring    the analyzer's ball (eval/fixtures/<clip>.json); grey when
                 it was only gap-filled
  magenta 1-9    moving yellow blobs: neighbouring frames aligned with the
                 camera motion and differenced. They find fast balls TrackNet
                 misses, but also shirts and hands, so check them.
  green dot      your label
Saves eval/ball_labels/<clip>.json.

Keys
  Enter          accept the yellow suggestion, go to the next frame
  1-9            accept magenta candidate n, go to the next frame
  click          the ball is exactly here (click its centre), next frame
  n              no ball visible (out of the picture, hidden), next frame
  x              delete this frame's label
  g              jump to the next unlabelled frame where the detector lost
                 the ball between sightings (the frames that matter most)
  z              zoom 2x around the suggestion, for small far balls
  ← / →  , / .   previous / next frame       < / >   10 frames
  space          play / pause
  w              save          q   save and quit          Esc  quit without saving

Label stretches of consecutive frames rather than scattered ones: the
detector sees three frames at a time and learns from the motion. Also label
some easy frames and "no ball" frames, or the model forgets them.
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "eval"))
KEY_LEFT, KEY_RIGHT = (2424832, 65361), (2555904, 65363)  # Windows, Linux (waitKeyEx)
KEY_ENTER = (13, 10)
VIEW_W = 1280
GAP_MAX = 25                 # frames: longer detector gaps are breaks in play, not misses

# ── Motion/colour candidates ────────────────────────────────────────────────
MOTION_SCALE = 0.5           # work at half resolution
MOTION_DIFF = 30             # grey levels of change that count as moving
MOTION_STEP = 2              # compare with frames i-2 and i+2
# The ball's yellow panels (FIVB Mikasa), as in ball_tracknet.measure_ball_diameter.
YELLOW = ((18, 90, 110), (38, 255, 255))


def motion_candidates(prev2, cur, next2, warp_prev, warp_next, limit=9):
    """Moving yellow blobs in `cur`, as normalised (x, y), biggest first.

    warp_prev/warp_next map prev2/next2 onto cur at MOTION_SCALE. A blob must
    differ from both neighbours (so it is where the ball is now, not where it
    was) and also change in the raw frames, which drops on-screen graphics:
    they do not move with the camera, so after alignment they would.
    """
    small = [cv2.resize(f, None, fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)
             for f in (prev2, cur, next2)]
    h, w = small[1].shape[:2]
    aligned = [cv2.warpPerspective(small[0], warp_prev, (w, h)),
               cv2.warpPerspective(small[2], warp_next, (w, h))]
    moved = np.minimum(cv2.absdiff(small[1], aligned[0]).max(axis=2),
                       cv2.absdiff(small[1], aligned[1]).max(axis=2)) > MOTION_DIFF
    moved_raw = np.minimum(cv2.absdiff(small[1], small[0]).max(axis=2),
                           cv2.absdiff(small[1], small[2]).max(axis=2)) > MOTION_DIFF
    yellow = cv2.inRange(cv2.cvtColor(small[1], cv2.COLOR_BGR2HSV), *YELLOW) > 0
    mask = cv2.dilate((moved & moved_raw & yellow).astype(np.uint8), np.ones((3, 3), np.uint8))
    n, _lbl, st, cen = cv2.connectedComponentsWithStats(mask)
    blobs = [(st[k, cv2.CC_STAT_AREA], cen[k][0] / w, cen[k][1] / h) for k in range(1, n)
             if 4 <= st[k, cv2.CC_STAT_AREA] <= 400]
    return [(x, y) for _a, x, y in sorted(blobs, reverse=True)[:limit]]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    video = sys.argv[1]
    clip = os.path.splitext(os.path.basename(video))[0]
    fx_path = os.path.join(EVAL_DIR, "fixtures", clip + ".json")
    out_path = os.path.join(EVAL_DIR, "ball_labels", clip + ".json")
    fixture = {}
    if os.path.isfile(fx_path):
        with open(fx_path, encoding="utf-8") as fh:
            fixture = {f["frame"]: f for f in json.load(fh)["frames"]}
    else:
        print(f"No fixture at {fx_path}: no detector suggestions (run export_fixture.py first)")
    labels = {}
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as fh:
            labels = {int(k): v for k, v in json.load(fh)["frames"].items()}
        print(f"Loaded {len(labels)} labelled frames from {out_path}")

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = VIEW_W / W
    state = {"frame": 0, "playing": False, "zoom": False, "view": None}
    decoded = {}

    def frame(i):
        """Decoded frame i (small cache: the candidates need i-2 and i+2 too)."""
        if i not in decoded:
            if len(decoded) > 12:
                decoded.pop(next(iter(decoded)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            decoded[i] = img if ok else None
        return decoded[i]

    moving = any(f.get("ref_from_frame") for f in fixture.values())

    def ref_from(i):
        h = fixture.get(i, {}).get("ref_from_frame")
        if h:
            return np.array(h, float).reshape(3, 3)
        return None if moving else np.eye(3)

    def candidates(i):
        if i - MOTION_STEP < 0 or i + MOTION_STEP >= n:
            return []
        imgs = [frame(i - MOTION_STEP), frame(i), frame(i + MOTION_STEP)]
        hs = [ref_from(i - MOTION_STEP), ref_from(i), ref_from(i + MOTION_STEP)]
        if any(im is None for im in imgs) or any(h is None for h in hs):
            return []
        N = np.diag([W * MOTION_SCALE, H * MOTION_SCALE, 1.0])
        warp = lambda a: N @ np.linalg.inv(hs[1]) @ a @ np.linalg.inv(N)
        return motion_candidates(imgs[0], imgs[1], imgs[2], warp(hs[0]), warp(hs[2]))

    detected = sorted(f for f, d in fixture.items() if d.get("ball_img_pos") and not d.get("ball_interpolated"))

    def next_hard(after):
        """Next unlabelled frame inside a short detector gap (the ball lost during play)."""
        for a, b in zip(detected, detected[1:]):
            if 1 < b - a <= GAP_MAX:
                for f in range(max(a + 1, after + 1), b):
                    if f not in labels:
                        return f
        return None

    def label(i, pos):
        labels[i] = pos
        state["frame"] = min(n - 1, i + 1)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or state["view"] is None:
            return
        x0, y0, s = state["view"]
        label(state["frame"], [round((x0 + x / s) / W, 5), round((y0 + y / s) / H, 5)])

    win = f"label ball - {clip}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    def save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"clip": clip, "video": os.path.relpath(video, os.path.join(EVAL_DIR, os.pardir)),
                       "width": W, "height": H,
                       "frames": {str(k): labels[k] for k in sorted(labels)}}, fh, indent=1)
        seen = sum(1 for v in labels.values() if v)
        print(f"Saved {len(labels)} frames ({seen} with the ball, {len(labels) - seen} without) to {out_path}")

    def draw_text(img, text, org, color, size=0.7, thick=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

    shown = {"frame": None, "cands": []}
    while True:
        fi = state["frame"]
        raw = frame(fi)
        if raw is None:
            state.update(frame=max(0, fi - 1), playing=False)
            continue
        if shown["frame"] != fi:
            shown.update(frame=fi, cands=[] if state["playing"] else candidates(fi))
        img = raw.copy()
        fd = fixture.get(fi, {})
        sugg = fd.get("ball_img_pos")
        if sugg:
            col = (160, 160, 160) if fd.get("ball_interpolated") else (0, 220, 255)
            cv2.circle(img, (int(sugg[0] * W), int(sugg[1] * H)), 20, col, 2)
        for k, (cx, cy) in enumerate(shown["cands"]):
            p = (int(cx * W), int(cy * H))
            cv2.circle(img, p, 14, (255, 0, 255), 2)
            draw_text(img, str(k + 1), (p[0] + 14, p[1] - 10), (255, 0, 255), 0.8)
        lab = labels.get(fi, "unset")
        if lab and lab != "unset":
            cv2.circle(img, (int(lab[0] * W), int(lab[1] * H)), 5, (0, 255, 0), -1)

        # 2x zoom around the label / suggestion / first candidate.
        centre = (lab if lab and lab != "unset" else None) or sugg or (shown["cands"][0] if shown["cands"] else None)
        if state["zoom"] and centre:
            cw, ch = W // 2, H // 2
            x0 = int(min(max(centre[0] * W - cw / 2, 0), W - cw))
            y0 = int(min(max(centre[1] * H - ch / 2, 0), H - ch))
            img = img[y0:y0 + ch, x0:x0 + cw]
            s = VIEW_W / cw
        else:
            x0 = y0 = 0
            s = scale
        state["view"] = (x0, y0, s)
        view = cv2.resize(img, (VIEW_W, int(img.shape[0] * s)))

        status = "no ball" if lab is None else ("unset" if lab == "unset" else "ball")
        seen = sum(1 for v in labels.values() if v)
        cv2.rectangle(view, (0, 0), (VIEW_W, 30), (0, 0, 0), -1)
        draw_text(view, f"frame {fi}/{n - 1}  label: {status}   labelled {len(labels)} "
                        f"({seen} ball)   {'ZOOM ' if state['zoom'] else ''}{'PLAYING' if state['playing'] else ''}",
                  (8, 21), (255, 255, 255), 0.6, 1)
        cv2.rectangle(view, (0, view.shape[0] - 26), (VIEW_W, view.shape[0]), (0, 0, 0), -1)
        draw_text(view, "Enter accept yellow | 1-9 magenta | click ball | n no ball | x delete | "
                        "g next hard frame | z zoom | w save | q quit",
                  (8, view.shape[0] - 8), (200, 200, 200), 0.5, 1)
        cv2.imshow(win, view)

        key = cv2.waitKeyEx(40 if state["playing"] else 30)
        if state["playing"]:
            state["frame"] = min(n - 1, fi + 1)
        if key == -1:
            continue
        k = key & 0xFF if key < 256 else key
        if key in KEY_RIGHT or k == ord("."):
            state["frame"] = min(n - 1, fi + 1)
        elif key in KEY_LEFT or k == ord(","):
            state["frame"] = max(0, fi - 1)
        elif k == ord(">"):
            state["frame"] = min(n - 1, fi + 10)
        elif k == ord("<"):
            state["frame"] = max(0, fi - 10)
        elif k == ord(" "):
            state["playing"] = not state["playing"]
        elif key in KEY_ENTER and sugg:
            label(fi, [round(sugg[0], 5), round(sugg[1], 5)])
        elif ord("1") <= k <= ord("9") and k - ord("1") < len(shown["cands"]):
            cx, cy = shown["cands"][k - ord("1")]
            label(fi, [round(cx, 5), round(cy, 5)])
        elif k == ord("n"):
            label(fi, None)
        elif k == ord("x"):
            labels.pop(fi, None)
        elif k == ord("g"):
            nxt = next_hard(fi)
            if nxt is None:
                print("No unlabelled frames left in detector gaps")
            else:
                state["frame"] = nxt
        elif k == ord("z"):
            state["zoom"] = not state["zoom"]
        elif k == ord("w"):
            save()
        elif k == ord("q"):
            save()
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
