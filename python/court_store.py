"""
Courts placed by hand in the app, reused for later clips of the same match.

Automatic court detection does not handle side views, but every clip of one
broadcast comes from the same camera standing in the same spot; only its pan
and zoom differ. Such views are related by one homography, exactly, so a
court placed once can be carried to any later clip by matching the venue
around it (stands, boards), as python/propagate_court.py does for the eval
clips.

Each saved court is a snapshot of the frame it belongs to plus the image
positions of the four court corners on it - the same corners the court
detector reports - so a court found here goes through the rest of the
pipeline exactly like a detected one.

Stored in ~/.tennis-3d/courts (TENNIS3D_COURTS overrides).
"""

import json
import os
import sys
import time

import cv2
import numpy as np

from propagate_court import MATCH_W, Matcher, to_norm_h

STORE = os.environ.get("TENNIS3D_COURTS") or os.path.join(os.path.expanduser("~"), ".tennis-3d", "courts")
# Newest first; older ones are rarely the same match.
MAX_TRIED = 40
# Inliers needed to trust a match. propagate_court.py uses 80 at 960 px.
MIN_INLIERS = 80


def _read_frame(video_path, index):
    cap = cv2.VideoCapture(video_path)
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def save_court(video_path, frame, corners):
    """Store a court: the frame it belongs to and its 4 corners (normalised)."""
    img = _read_frame(video_path, frame)
    if img is None:
        raise RuntimeError(f"cannot read frame {frame} of {video_path}")
    os.makedirs(STORE, exist_ok=True)
    key = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    h, w = img.shape[:2]
    cv2.imwrite(os.path.join(STORE, key + ".jpg"),
                cv2.resize(img, (MATCH_W, int(round(h * MATCH_W / w)))), [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(STORE, key + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"video": video_path, "frame": frame, "corners": corners,
                   "width": w, "height": h, "saved": time.time()}, fh)
    return key


def find_court(frame_bgr):
    """Corners of a stored court mapped onto this frame, or (None, reason).

    Tries the stored courts newest first and takes the first whose venue
    matches with enough inliers.
    """
    if not os.path.isdir(STORE):
        return None, "no saved courts"
    keys = sorted((f[:-5] for f in os.listdir(STORE) if f.endswith(".json")), reverse=True)[:MAX_TRIED]
    if not keys:
        return None, "no saved courts"
    m = Matcher()
    h, w = frame_bgr.shape[:2]
    feat = m.features(frame_bgr)
    best = (0, None, None)
    for key in keys:
        img = cv2.imread(os.path.join(STORE, key + ".jpg"))
        if img is None:
            continue
        with open(os.path.join(STORE, key + ".json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        # The snapshot is stored at MATCH_W; features() resizes to the same.
        H, inliers = m.homography(m.features(img), feat)
        if H is None or inliers <= best[0]:
            continue
        Hn = to_norm_h(H, w, h)   # stored frame (normalised) -> this frame
        pts = cv2.perspectiveTransform(np.float32(meta["corners"]).reshape(-1, 1, 2), Hn).reshape(-1, 2)
        best = (inliers, [[round(float(x), 6), round(float(y), 6)] for x, y in pts], key)
        if inliers >= 3 * MIN_INLIERS:
            break   # unmistakable: no need to try older courts
    if best[0] < MIN_INLIERS:
        return None, f"no saved court matches this venue (best {best[0]} inliers)"
    return best[1], {"saved": best[2], "inliers": best[0]}


if __name__ == "__main__":
    # python court_store.py VIDEO [FRAME]: which saved court matches, if any.
    v = sys.argv[1]
    f = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(find_court(_read_frame(v, f)))
