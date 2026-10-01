"""
Volleyball court detection by colour segmentation.

The Hough-line detector this replaces was unusable on broadcast footage: its
corners drifted by tens of percent between frames, frequently fell outside the
image, and implied cameras 3 m high at 0.5 m distance.

An indoor volleyball court is a large, saturated, roughly uniform region
(wood/taraflex) surrounded by a contrasting free zone, so segmenting it by
colour is far more stable than chasing individual white lines through a crowd,
players and advertising boards. Players punch holes in the region, but the
court is convex, so the convex hull of the largest region recovers the true
boundary.

A camera behind one baseline may see the whole court or only one half, and
fitting an 18 m model to a visible 9 m half wrecks the calibration. Which it is
cannot be assumed, so the detector reads the horizontal lines inside the region
— rectified, the full court shows three at 1/3, 1/2, 2/3 (attack, net, attack)
while a half shows a single one — and picks the hypothesis that explains them.
"""

import sys

import cv2
import numpy as np

COURT_W = 9.0
COURT_L = 18.0
NET_Y = COURT_L / 2.0     # 9.0
ATTACK = 3.0

# Which physical rectangle the visible region might be. A broadcast camera
# behind one baseline may see the whole court or only one half, and fitting an
# 18 m model to a visible 9 m half throws the calibration out completely.
#
# The hypotheses are told apart by the horizontal lines *inside* the region.
# Rectified so the region is a unit square, the interior lines fall at:
#   full court (18 m): attack 6 m, net 9 m, attack 12 m -> 1/3, 1/2, 2/3
#   one half    (9 m): a single attack line 3 m from the net -> 1/3 or 2/3
# Those lines are not used to fit the quad, so matching them is an independent
# test of which hypothesis is right.
_HYPOTHESES = [
    ("full-court", [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]],
     [1 / 3.0, 1 / 2.0, 2 / 3.0]),
    ("near-half", [[0, NET_Y], [COURT_W, NET_Y], [COURT_W, COURT_L], [0, COURT_L]],
     [1 / 3.0]),
    ("far-half", [[0, 0], [COURT_W, 0], [COURT_W, NET_Y], [0, NET_Y]],
     [2 / 3.0]),
]


def _court_mask(frame):
    """Binary mask of the playing surface."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    # Wood / warm taraflex: orange-yellow hue, moderate saturation, well lit.
    m = ((h >= 8) & (h <= 32) & (s >= 40) & (s <= 200) & (v >= 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return m


def _largest_quad(mask, min_area_frac=0.05):
    """Convex-hull quadrilateral of the largest region, in pixels, or None."""
    ih, iw = mask.shape[:2]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_area_frac * ih * iw:
        return None

    region = (lbl == idx).astype(np.uint8)
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Players stand on the court and cut notches out of the region; the court
    # itself is convex, so the hull restores the true outline.
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)

    quad = None
    for eps in np.arange(0.010, 0.120, 0.005):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(-1, 2).astype(np.float64)
            break
    if quad is None:
        return None

    # Order as top-left, top-right, bottom-right, bottom-left.
    ssum = quad.sum(axis=1)
    sdiff = (quad[:, 0] - quad[:, 1])
    tl = quad[int(np.argmin(ssum))]
    br = quad[int(np.argmax(ssum))]
    tr = quad[int(np.argmax(sdiff))]
    bl = quad[int(np.argmin(sdiff))]
    ordered = np.array([tl, tr, br, bl])
    # Reject degenerate shapes.
    if len({tuple(np.round(p, 3)) for p in ordered}) != 4:
        return None
    return ordered


def _interior_line_fractions(frame, quad, size=600):
    """Find horizontal court lines inside the quad, as fractions of its height.

    The quad is rectified to a square and scanned for thin bright ridges. A
    plain "is it white" test does not work here: the floor itself is pale, so
    broad bright areas and players' kit swamp the painted lines. Requiring a
    row to be brighter than rows above *and* below isolates real lines.
    """
    M = cv2.getPerspectiveTransform(
        quad.astype(np.float32),
        np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32),
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = cv2.warpPerspective(gray, M, (size, size)).astype(np.float32)

    k = max(4, size // 75)
    ridge = rect - 0.5 * (np.roll(rect, k, axis=0) + np.roll(rect, -k, axis=0))
    ridge[:k] = 0
    ridge[-k:] = 0
    np.clip(ridge, 0, None, out=ridge)

    prof = ridge[:, int(size * 0.2):int(size * 0.8)].mean(axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 0).ravel()
    if prof.max() <= 1e-6:
        return []

    thresh = prof.max() * 0.45
    peaks = []
    min_sep = max(4, size // 24)
    for i in np.argsort(-prof):
        if prof[i] < thresh:
            break
        if all(abs(int(i) - p) >= min_sep for p in peaks):
            peaks.append(int(i))
    fracs = sorted(p / float(size) for p in peaks)
    # Ignore anything hugging the edges: those are the boundary lines.
    return [f for f in fracs if 0.08 < f < 0.92]


def _score_hypothesis(expected, found, tol=0.05):
    """Match predicted interior-line positions against detected ones."""
    if not expected:
        return 0.0
    hit = 0.0
    for e in expected:
        if found:
            d = min(abs(e - f) for f in found)
            if d <= tol:
                hit += 1.0 - d / tol
    recall = hit / len(expected)
    # Penalise detected lines the hypothesis cannot account for, so a model
    # explaining one of three lines does not beat one explaining all three.
    unexplained = sum(1 for f in found if all(abs(e - f) > tol for e in expected))
    penalty = unexplained / max(1, len(found))
    return recall * (1.0 - 0.6 * penalty)


def detect_court_corners(frame):
    """Detect the court and return the 4 FULL-court corners, normalised [0,1].

    Order: far-left, far-right, near-right, near-left — matching
    _CORNERS_PHYSICAL in analyzer.py, so callers need no changes even when only
    part of the court is visible.

    Returns (corners, info) or (None, reason).
    """
    ih, iw = frame.shape[:2]
    quad = _largest_quad(_court_mask(frame))
    if quad is None:
        return None, "no court-coloured region"

    found = _interior_line_fractions(frame, quad)
    if not found:
        return None, "no interior court lines found"

    best = None
    for name, src, expected in _HYPOTHESES:
        score = _score_hypothesis(expected, found)
        if best is None or score > best[0]:
            best = (score, name, src)

    score, name, src = best
    if score < 0.5:
        return None, (f"no hypothesis fits the interior lines "
                      f"(best {name} {score:.2f}, lines at "
                      f"{[round(f, 3) for f in found]})")

    H, _ = cv2.findHomography(np.array(src, np.float32), quad.astype(np.float32))
    if H is None:
        return None, "homography failed"

    full = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]],
                    np.float32).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(full, H).reshape(-1, 2)
    out = [[round(float(p[0]) / iw, 6), round(float(p[1]) / ih, 6)] for p in corners]
    return out, {"hypothesis": name,
                 "score": round(float(score), 3),
                 "lines": [round(f, 3) for f in found]}


def detect_court_corners_verbose(frame, tag=""):
    corners, info = detect_court_corners(frame)
    if corners is None:
        print(f"[court]{tag} rejected: {info}", file=sys.stderr)
    return corners, info
