#!/usr/bin/env python3
"""
Label ball touches on a clip, for the 3D eval (npm run eval / npm test).

    python python/label_touches.py match.mp4

Shows the video with the analyzer's ball (yellow, grey = gap-filled) and the
player boxes near it, from eval/fixtures/<clip>.json (make it first with
python/export_fixture.py). Saves eval/labels/<clip>.json.

Keys
  ← / →  or  , / .     previous / next frame
  < / >                10 frames back / forward
  space                play / pause
  t s r e a b d        mark a touch at this frame: toss, serve, reception,
                       set (e), attack, block, dig
  (then click)         the toucher's feet; for a jumping player click where
                       they took off. Press Enter to skip the feet.
  x                    delete the touch at this frame
  w                    save          q   save and quit          Esc  quit without saving

Court (only when the analyzer's court is wrong)
  k                    court mode on/off. A numbered list of court points
                       appears (corners, and where the 3 m lines and the
                       centre line meet the outer lines), starting at point 1.
  click                place the highlighted point; the next one is selected
  n                    skip the highlighted point (not visible), go to the next
  1-9, 0               jump to a point directly
                       Any 4+ points will do, in any order and on any frames:
                       when the camera pans, place each where it is visible.
                       With 4 or more the fitted court is drawn on every
                       frame; check it lines up.
  v                    camera view: from the side / from behind a baseline
  Backspace            remove the chosen point

Mark the frame where the ball is in contact (or the closest visible frame).
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "eval"))
TYPES = {ord("t"): "toss", ord("s"): "serve", ord("r"): "reception", ord("e"): "set",
         ord("a"): "attack", ord("b"): "block", ord("d"): "dig"}
COLORS = {"toss": (230, 179, 127), "serve": (214, 134, 47), "reception": (80, 175, 76), "set": (0, 184, 224),
          "attack": (63, 70, 226), "block": (181, 66, 155), "dig": (0, 143, 255)}
KEY_LEFT, KEY_RIGHT = (2424832, 65361), (2555904, 65363)  # Windows, Linux (waitKeyEx)
KEY_BACKSPACE = 8
VIEW_W = 1280

# Court points by camera view, as seen in the picture, with their court
# coordinates (x across, z along; net at z = 0). The world frame must not be
# mirrored relative to the picture or no camera can explain it:
#   side view:   camera at -x, so the picture's left end is z = -9 and the far
#                sideline is x = +4.5
#   behind view: camera at +z, so the far baseline is z = -9 and the picture's
#                left sideline is x = -4.5
NET_TAPE_Y = 2.43            # top of the net tape, men
ANTENNA_TOP_Y = NET_TAPE_Y + 0.8
COURT_POINTS = {
    "side": [
        ("far sideline, left end", (4.5, -9)),
        ("far sideline, left 3 m line", (4.5, -3)),
        ("far sideline, centre line", (4.5, 0)),
        ("far sideline, right 3 m line", (4.5, 3)),
        ("far sideline, right end", (4.5, 9)),
        ("near sideline, left end", (-4.5, -9)),
        ("near sideline, left 3 m line", (-4.5, -3)),
        ("near sideline, centre line", (-4.5, 0)),
        ("near sideline, right 3 m line", (-4.5, 3)),
        ("near sideline, right end", (-4.5, 9)),
        # Above the floor (x, z, height): the antennas stand on the sidelines,
        # from the net tape (2.43 m, men) up 0.8 m. Known heights fix the
        # camera's focal length and height scale.
        ("far antenna, top", (4.5, 0, ANTENNA_TOP_Y)),
        ("far antenna, at the net tape", (4.5, 0, NET_TAPE_Y)),
        ("near antenna, top", (-4.5, 0, ANTENNA_TOP_Y)),
        ("near antenna, at the net tape", (-4.5, 0, NET_TAPE_Y)),
    ],
    "behind": [
        ("far baseline, left end", (-4.5, -9)),
        ("far baseline, right end", (4.5, -9)),
        ("near baseline, right end", (4.5, 9)),
        ("near baseline, left end", (-4.5, 9)),
        ("left sideline, far 3 m line", (-4.5, -3)),
        ("left sideline, centre line", (-4.5, 0)),
        ("left sideline, near 3 m line", (-4.5, 3)),
        ("right sideline, far 3 m line", (4.5, -3)),
        ("right sideline, centre line", (4.5, 0)),
        ("right sideline, near 3 m line", (4.5, 3)),
        ("left antenna, top", (-4.5, 0, ANTENNA_TOP_Y)),
        ("left antenna, at the net tape", (-4.5, 0, NET_TAPE_Y)),
        ("right antenna, top", (4.5, 0, ANTENNA_TOP_Y)),
        ("right antenna, at the net tape", (4.5, 0, NET_TAPE_Y)),
    ],
}
NUM_KEYS = [ord(c) for c in "1234567890"]
# Court lines to draw from a fitted court (world x, z).
LINES = [((-4.5, -9), (4.5, -9)), ((4.5, -9), (4.5, 9)), ((4.5, 9), (-4.5, 9)), ((-4.5, 9), (-4.5, -9)),
         ((-4.5, 0), (4.5, 0)), ((-4.5, -3), (4.5, -3)), ((-4.5, 3), (4.5, 3))]


def legacy_court(court):
    """Old format (4 corners on one frame) -> point list."""
    if not court or "points" in court:
        return court
    view = court.get("view", "side")
    order = [0, 4, 9, 5] if view == "side" else [0, 1, 2, 3]   # far-left, far-right, near-right, near-left
    pts = []
    for idx, pt in zip(order, court["corners"]):
        name, world = COURT_POINTS[view][idx]
        pts.append({"id": idx, "name": name, "world": list(world), "frame": court["frame"], "pt": pt})
    return {"view": view, "points": pts}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    video = sys.argv[1]
    clip = os.path.splitext(os.path.basename(video))[0]
    fx_path = os.path.join(EVAL_DIR, "fixtures", clip + ".json")
    lb_path = os.path.join(EVAL_DIR, "labels", clip + ".json")
    frames = {}
    if os.path.isfile(fx_path):
        with open(fx_path, encoding="utf-8") as fh:
            frames = {f["frame"]: f for f in json.load(fh)["frames"]}
    else:
        print(f"No fixture at {fx_path}: no ball/player overlay (run export_fixture.py first)")
    touches = []
    court = None
    if os.path.isfile(lb_path):
        with open(lb_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        touches = saved.get("touches", [])
        court = legacy_court(saved.get("court"))
        print(f"Loaded {len(touches)} touches from {lb_path}")
    court = court or {"view": "side", "points": []}

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = VIEW_W / W
    state = {"frame": 0, "playing": False, "await_feet": None, "court_mode": False, "court_pick": None}

    def ref_from(fi):
        """Frame -> reference homography (normalised) from the fixture; identity if static."""
        h = frames.get(fi, {}).get("ref_from_frame")
        return np.array(h, float).reshape(3, 3) if h else np.eye(3)

    def fitted_court():
        """World (x,z) -> reference image homography from the placed points, or None."""
        # Floor points only: a homography maps the floor plane.
        pts = [p for p in court["points"] if len(p["world"]) == 2]
        if len(pts) < 4:
            return None
        world = np.float32([p["world"] for p in pts])
        ref = np.float32([cv2.perspectiveTransform(np.float32([[p["pt"]]]), ref_from(p["frame"]))[0, 0] for p in pts])
        Hc, _ = cv2.findHomography(world, ref, 0)
        return Hc

    def next_point(after):
        """The next point after `after` that is not placed yet (wraps), or None."""
        total = len(COURT_POINTS[court["view"]])
        placed = {p["id"] for p in court["points"]}
        for step in range(1, total + 1):
            j = (after + step) % total
            if j not in placed:
                return j
        return None

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pt = [round(x / scale / W, 5), round(y / scale / H, 5)]
        if state["court_mode"] and state["court_pick"] is not None:
            idx = state["court_pick"]
            name, world = COURT_POINTS[court["view"]][idx]
            court["points"] = [p for p in court["points"] if p["id"] != idx]
            court["points"].append({"id": idx, "name": name, "world": list(world), "frame": state["frame"], "pt": pt})
            state["court_pick"] = next_point(idx)
        elif state["await_feet"] is not None:
            state["await_feet"]["feet"] = pt
            state["await_feet"] = None

    win = f"label touches - {clip}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    def save():
        os.makedirs(os.path.dirname(lb_path), exist_ok=True)
        with open(lb_path, "w", encoding="utf-8") as fh:
            data = {"clip": clip, "touches": sorted(touches, key=lambda t: t["frame"])}
            if len(court["points"]) >= 4:
                data["court"] = court
            json.dump(data, fh, indent=2)
        msg = f"Saved {len(touches)} touches"
        if len(court["points"]) >= 4:
            msg += f" and a court from {len(court['points'])} points"
        elif court["points"]:
            msg += f" (court not saved: {len(court['points'])} of at least 4 points)"
        print(f"{msg} to {lb_path}")

    def draw_text(img, text, org, color, size=0.9, thick=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

    cached = {"frame": None, "img": None}
    while True:
        fi = state["frame"]
        # Decode only when the frame changes; the window is redrawn every
        # ~30 ms so clicks (feet, court points) show up immediately.
        if cached["frame"] != fi:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, raw = cap.read()
            if not ok:
                state["frame"] = max(0, fi - 1)
                state["playing"] = False
                continue
            cached.update(frame=fi, img=raw)
        img = cached["img"].copy()
        fd = frames.get(fi, {})
        for x1, y1, x2, y2, *_ in fd.get("players_near_ball", []):
            cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (90, 200, 90), 1)
        if fd.get("ball_img_pos"):
            bx, by = fd["ball_img_pos"]
            col = (160, 160, 160) if fd.get("ball_interpolated") else (0, 220, 255)
            cv2.circle(img, (int(bx * W), int(by * H)), 18, col, 2)
        for t in touches:
            if t.get("feet") and abs(t["frame"] - fi) <= 15:
                fx_, fy_ = t["feet"]
                cv2.drawMarker(img, (int(fx_ * W), int(fy_ * H)), COLORS[t["type"]], cv2.MARKER_TILTED_CROSS, 22, 3)

        # Fitted court, mapped onto this frame (follows the camera when it pans).
        Hc = fitted_court()
        if Hc is not None:
            frame_from_world = np.linalg.inv(ref_from(fi)) @ Hc
            for a, b in LINES:
                seg = cv2.perspectiveTransform(np.float32([[a, b]]), frame_from_world)[0]
                p1, p2 = (seg[0] * [W, H]).astype(int), (seg[1] * [W, H]).astype(int)
                cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 2, cv2.LINE_AA)
        # Placed points: on their own frame as crosses, elsewhere mapped through the motion.
        if state["court_mode"] or Hc is None:
            for p in court["points"]:
                q = cv2.perspectiveTransform(np.float32([[p["pt"]]]),
                                             np.linalg.inv(ref_from(fi)) @ ref_from(p["frame"]))[0, 0]
                pos = (int(q[0] * W), int(q[1] * H))
                here = p["frame"] == fi
                cv2.drawMarker(img, pos, (0, 0, 255) if here else (0, 140, 255), cv2.MARKER_CROSS, 34 if here else 22, 4 if here else 2)
                draw_text(img, str(p["id"] + 1), (pos[0] + 12, pos[1] - 10), (0, 0, 255) if here else (0, 140, 255))

        view = cv2.resize(img, (VIEW_W, int(H * scale)))
        if state["court_mode"]:
            # Point list: chosen point highlighted, placed points ticked.
            placed = {p["id"] for p in court["points"]}
            y0 = 60
            draw_text(view, f"COURT ({court['view']} view, v to switch) - click the highlighted point, n = skip it, 1-0 = jump:",
                      (12, y0), (255, 255, 255), 0.6, 1)
            for j, (name, _) in enumerate(COURT_POINTS[court["view"]]):
                mark = "x" if j in placed else " "
                color = (0, 255, 255) if state["court_pick"] == j else ((120, 255, 120) if j in placed else (230, 230, 230))
                key_hint = (j + 1) % 10 if j < 10 else "n"
                draw_text(view, f"[{mark}] {key_hint}  {name}", (12, y0 + 26 * (j + 1)), color, 0.6, 1)
        here = [t for t in touches if t["frame"] == fi]
        status = f"frame {fi}/{n - 1}"
        n_pts = sum(1 for p in court["points"] if len(p["world"]) == 2)   # floor points
        if state["court_mode"]:
            status += f"   court: {n_pts} point{'s' if n_pts != 1 else ''}"
            status += "   (need 4+)" if n_pts < 4 else "   check the red lines, k to finish, w to save"
            if state["court_pick"] is not None:
                status += f"   click: {COURT_POINTS[court['view']][state['court_pick']][0]}  (n = not visible, skip)"
        elif n_pts >= 4:
            status += "   court placed"
        if here:
            status += f"   TOUCH: {here[0]['type']}"
        if state["await_feet"] is not None:
            status += "   click the toucher's feet (Enter to skip)"
        cv2.rectangle(view, (0, 0), (VIEW_W, 30), (0, 0, 0), -1)
        cv2.putText(view, status, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        listing = "  ".join(f"{t['type'][:4]}@{t['frame']}" for t in sorted(touches, key=lambda t: t["frame"]))
        cv2.rectangle(view, (0, view.shape[0] - 26), (VIEW_W, view.shape[0]), (0, 0, 0), -1)
        cv2.putText(view, listing[-150:], (10, view.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(win, view)

        key = cv2.waitKeyEx(33 if state["playing"] else 30)
        if key == -1:
            if state["playing"]:
                state["frame"] = min(n - 1, fi + 1)
            continue
        k = key & 0xFF
        if state["await_feet"] is not None and key in (13, 10):
            state["await_feet"] = None
        elif key in KEY_RIGHT or k == ord("."):
            state["frame"] = min(n - 1, fi + 1)
        elif key in KEY_LEFT or k == ord(","):
            state["frame"] = max(0, fi - 1)
        elif k == ord(">"):
            state["frame"] = min(n - 1, fi + 10)
        elif k == ord("<"):
            state["frame"] = max(0, fi - 10)
        elif k == ord(" "):
            state["playing"] = not state["playing"]
        elif k == ord("k"):
            state["court_mode"] = not state["court_mode"]
            state["court_pick"] = next_point(-1) if state["court_mode"] else None
            state["playing"] = False
        elif state["court_mode"] and k == ord("n"):
            cur = state["court_pick"] if state["court_pick"] is not None else -1
            state["court_pick"] = (cur + 1) % len(COURT_POINTS[court["view"]])
        elif state["court_mode"] and k in NUM_KEYS:
            state["court_pick"] = NUM_KEYS.index(k)
        elif state["court_mode"] and key == KEY_BACKSPACE and state["court_pick"] is not None:
            court["points"] = [p for p in court["points"] if p["id"] != state["court_pick"]]
        elif k == ord("v"):
            # Same numbered points, re-read in the other view.
            court["view"] = "behind" if court["view"] == "side" else "side"
            for p in court["points"]:
                p["name"], world = COURT_POINTS[court["view"]][p["id"]]
                p["world"] = list(world)
        elif not state["court_mode"] and key in TYPES:
            touches[:] = [t for t in touches if t["frame"] != fi]
            touch = {"frame": fi, "type": TYPES[key]}
            touches.append(touch)
            state["await_feet"] = touch
            state["playing"] = False
        elif k == ord("x"):
            touches[:] = [t for t in touches if t["frame"] != fi]
        elif k == ord("w"):
            save()
        elif k == ord("q"):
            save()
            break
        elif k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
