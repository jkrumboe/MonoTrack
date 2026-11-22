"""
Camera motion for a broadcast camera that pans and zooms from a fixed spot.

Such a camera maps every frame to every other by one homography, exactly, for
all points (not just the floor): it only rotates and zooms. So measuring each
frame's homography to one reference frame lets everything downstream work in
the reference image, as if the camera never moved ("virtual static camera").

    motion = CameraMotion()
    for frame in video:
        motion.update(frame)
    motion.set_reference(ref_index)              # e.g. the court calibration frame
    H = motion.ref_from_frame(i)                 # 3x3, normalised [0,1] coords, or None

How: background corners are tracked frame to frame (Lucas-Kanade) and a
homography is fitted with RANSAC, so moving players and the ball are rejected
as outliers. Chaining frame-to-frame steps drifts, so every KEYFRAME_EVERY
frames the current frame is also matched directly against the previous
keyframe (ORB), which cuts the number of chained steps by that factor.
Frames whose fit is weak are marked untrusted rather than guessed.
"""

import cv2
import numpy as np

TRACK_W = 640                 # tracking resolution; homographies are rescaled
KEYFRAME_EVERY = 15
MIN_INLIERS = 40
RANSAC_PX = 1.0               # at TRACK_W
# Lucas-Kanade pyramid levels above the base image. 2 instead of 3: ~15% less
# time a frame, homographies within 1.4 px (at 1920) of 3 on the moving clips.
LK_LEVELS = 2
# A camera that never moves more than this (px at 1920 wide, corner drift over
# the whole clip) is treated as static: no ref fields are emitted.
STATIC_MAX_DRIFT_PX = 2.0


def _gray(frame):
    h, w = frame.shape[:2]
    scale = TRACK_W / w
    small = cv2.resize(frame, (TRACK_W, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale


def _fit(src, dst):
    """Homography dst <- src with RANSAC; (H, inliers) or (None, 0)."""
    if src is None or len(src) < MIN_INLIERS:
        return None, 0
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX, maxIters=2000, confidence=0.999)
    if H is None:
        return None, 0
    return H, int(mask.sum())


class CameraMotion:
    def __init__(self):
        self.prev = None              # previous small gray frame
        self.keyframe = None          # (index, small gray, H_first_from_key)
        self.first_from = []          # per frame: H (small px) mapping frame -> frame 0, or None
        self.quality = []             # per frame: inlier count of the step that placed it
        self.size = None              # (W, H) full resolution
        self.scale = 1.0
        self.ref = 0
        self.orb = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.key_orb = None           # (keyframe gray, its ORB keypoints, descriptors)

    # ── per-frame update ──────────────────────────────────────────────────
    def update(self, frame):
        g, self.scale = _gray(frame)
        self.size = (frame.shape[1], frame.shape[0])
        i = len(self.first_from)
        if i == 0:
            self.first_from.append(np.eye(3))
            self.quality.append(10_000)
            self.prev = g
            self.keyframe = (0, g, np.eye(3))
            return

        H_prev = self.first_from[-1]
        step, inl = self._lk_step(self.prev, g)          # prev <- cur
        H = H_prev @ step if (step is not None and H_prev is not None) else None
        quality = inl

        if i % KEYFRAME_EVERY == 0:
            k_idx, k_gray, H_key = self.keyframe
            direct, inl_k = self._orb_step(k_gray, g)    # key <- cur
            if direct is not None and H_key is not None and inl_k >= MIN_INLIERS:
                H = H_key @ direct
                quality = max(quality, inl_k)
            if H is not None:
                self.keyframe = (i, g, H)

        self.first_from.append(H)
        self.quality.append(quality if H is not None else 0)
        self.prev = g

    def _lk_step(self, prev, cur):
        p0 = cv2.goodFeaturesToTrack(prev, 600, 0.01, 7, blockSize=7)
        if p0 is None:
            return None, 0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=LK_LEVELS)
        ok = st.reshape(-1) == 1
        H, inl = _fit(p1[ok].reshape(-1, 2), p0[ok].reshape(-1, 2))
        return H, inl

    def _orb_step(self, key, cur):
        # A keyframe is matched against later frames until one replaces it:
        # its features are found once.
        if self.key_orb is None or self.key_orb[0] is not key:
            self.key_orb = (key, *self.orb.detectAndCompute(key, None))
        k1, d1 = self.key_orb[1], self.key_orb[2]
        k2, d2 = self.orb.detectAndCompute(cur, None)
        if d1 is None or d2 is None:
            return None, 0
        m = self.matcher.match(d2, d1)
        if len(m) < MIN_INLIERS:
            return None, 0
        src = np.float32([k2[x.queryIdx].pt for x in m])
        dst = np.float32([k1[x.trainIdx].pt for x in m])
        return _fit(src, dst)

    # ── results ───────────────────────────────────────────────────────────
    def set_reference(self, index):
        self.ref = index

    def _ref_from_small(self, i):
        Hf, Hr = self.first_from[i], self.first_from[self.ref]
        if Hf is None or Hr is None:
            return None
        return np.linalg.inv(Hr) @ Hf

    def ref_from_frame(self, i):
        """3x3 homography frame i -> reference, in normalised [0,1] image coords, or None."""
        H = self._ref_from_small(i)
        if H is None or self.quality[i] < MIN_INLIERS:
            return None
        W, Hh = self.size
        s = self.scale
        # normalised -> small px: diag(W*s, H*s); and back.
        N = np.diag([W * s, Hh * s, 1.0])
        Hn = np.linalg.inv(N) @ H @ N
        return Hn / Hn[2, 2]

    def is_static(self):
        """True when the camera never moved enough to matter."""
        W, Hh = self.size or (1, 1)
        corners = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float).T
        worst = 0.0
        for i in range(len(self.first_from)):
            Hn = self.ref_from_frame(i)
            if Hn is None:
                continue
            q = Hn @ corners
            q = q[:2] / q[2]
            worst = max(worst, float(np.max(np.hypot((q[0] - corners[0]) * W, (q[1] - corners[1]) * Hh))))
        return worst < STATIC_MAX_DRIFT_PX

    def zoom(self, i):
        """Local scale of frame i relative to the reference (>1 = zoomed out vs reference)."""
        Hn = self.ref_from_frame(i)
        if Hn is None:
            return None
        J = Hn[:2, :2] / Hn[2, 2]
        return float(np.sqrt(abs(np.linalg.det(J))))


def apply_h(Hn, pt):
    """Map a normalised point through a normalised homography."""
    v = Hn @ np.array([pt[0], pt[1], 1.0])
    return [float(v[0] / v[2]), float(v[1] / v[2])]


def local_scale(Hn, pt):
    """How much H magnifies lengths around pt (for the ball's apparent diameter)."""
    eps = 1e-3
    p0 = np.array(apply_h(Hn, pt))
    px = np.array(apply_h(Hn, [pt[0] + eps, pt[1]]))
    py = np.array(apply_h(Hn, [pt[0], pt[1] + eps]))
    # Geometric mean of the two axis stretches; the aspect cancels for a
    # rotation+zoom camera.
    return float(np.sqrt(np.linalg.norm(px - p0) / eps * np.linalg.norm(py - p0) / eps))


def map_box(Hn, box):
    """Axis-aligned box [x1,y1,x2,y2,conf] through H (bounding box of its corners)."""
    x1, y1, x2, y2 = box[:4]
    pts = [apply_h(Hn, p) for p in ([x1, y1], [x2, y1], [x1, y2], [x2, y2])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5)] + list(box[4:])
