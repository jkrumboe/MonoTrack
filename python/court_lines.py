"""
Volleyball court detection by fitting the court's painted lines.

The colour detector (court_detect.py) needs the whole playing surface as one
convex region with its four corners in view. Broadcast side cameras break
both: they are zoomed in, so one or both baselines are off-screen, and the
pale court sits inside a blue free zone inside more pale floor, so the colour
region is not the court. Its quads were off by hundreds of pixels on every
real clip.

The painted lines survive both. This detector:
  1. finds thin white ridges and groups their segments into straight lines,
     split into a shallow (near-horizontal) and a steep family;
  2. hypothesises which lines are the two sidelines and which two are known
     transverse lines (baseline, 3 m line or centre line), for both a side
     camera and one behind a baseline, giving a court->image homography from
     four line intersections;
  3. keeps hypotheses a real camera could produce and scores each by how much
     of the whole court model lands on white pixels;
  4. re-ranks the best of those by the colour change across the court
     boundary, both line by line and court against surroundings: the rules
     require the court and free zone to differ in colour, and a wrong fit,
     which puts a boundary somewhere on the playing surface, sees the same
     colour on both sides of it. Line support alone is fooled by advertising
     boards and squashed whole-court readings;
  5. refines the winner as a physical camera fitted to every court line in
     view.

Corners are returned in the analyzer's order even when they lie outside the
image: they only encode the homography.
"""

import itertools
import sys

import cv2
import numpy as np

HALF_W = 4.5
HALF_L = 9.0
# Transverse lines along the court: baselines, 3 m lines, centre line.
TRANSVERSE_Z = [-9.0, -3.0, 0.0, 3.0, 9.0]

# Court corners in the analyzer's order (court-local metres: x 0..9, z 0..18),
# i.e. centred (x, z) = (-4.5,-9), (4.5,-9), (4.5,9), (-4.5,9).
_CORNERS_CENTRED = np.array([[-HALF_W, -HALF_L], [HALF_W, -HALF_L],
                             [HALF_W, HALF_L], [-HALF_W, HALF_L]], np.float64)

WORK_W = 960          # detection runs at this width
MAX_SHALLOW = 7       # line candidates kept per family
MAX_STEEP = 10
SHALLOW_DEG = 25.0    # |angle from horizontal| below this = shallow family


def _model_lines():
    """Court lines as (x1, z1, x2, z2), centred metres."""
    lines = [(-HALF_W, -HALF_L, -HALF_W, HALF_L), (HALF_W, -HALF_L, HALF_W, HALF_L)]
    lines += [(-HALF_W, z, HALF_W, z) for z in TRANSVERSE_Z]
    return lines


def _model_samples(step=0.15):
    pts = []
    for x1, z1, x2, z2 in _model_lines():
        n = max(2, int(np.hypot(x2 - x1, z2 - z1) / step) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts.append(np.stack([x1 + (x2 - x1) * t, z1 + (z2 - z1) * t], axis=1))
    return np.concatenate(pts)


_SAMPLES = _model_samples()
_SAMPLES_H = np.concatenate([_SAMPLES, np.ones((len(_SAMPLES), 1))], axis=1).T


def _region_samples():
    """Points on the playing surface, and in a band of free zone around it."""
    gx, gz = np.meshgrid(np.arange(-4.0, 4.01, 0.5), np.arange(-8.5, 8.51, 0.5))
    inside = np.stack([gx.ravel(), gz.ravel()], axis=1)
    out = []
    for off in np.arange(0.7, 2.6, 0.6):
        zz = np.arange(-HALF_L - off, HALF_L + off + 0.01, 0.5)
        xx = np.arange(-HALF_W - off, HALF_W + off + 0.01, 0.5)
        out += [[-HALF_W - off, z] for z in zz] + [[HALF_W + off, z] for z in zz]
        out += [[x, -HALF_L - off] for x in xx] + [[x, HALF_L + off] for x in xx]
    return inside, np.array(out)


_INSIDE, _OUTSIDE = _region_samples()


def _side_samples(step=0.4):
    """Per boundary line, points just inside the court and just outside it."""
    sides = []
    for axis, sgn in (("x", -1), ("x", 1), ("z", -1), ("z", 1)):
        half = HALF_W if axis == "x" else HALF_L
        along = np.arange(-HALF_L + 0.2, HALF_L - 0.19, step) if axis == "x"             else np.arange(-HALF_W + 0.2, HALF_W - 0.19, step)
        pair = []
        for offs in (np.linspace(-1.0, -0.3, 3), np.linspace(0.6, 1.6, 3)):
            pts = [[sgn * (half + o), a] if axis == "x" else [a, sgn * (half + o)]
                   for o in offs for a in along]
            pair.append(np.array(pts))
        sides.append(tuple(pair))
    return sides


_SIDES = _side_samples()
RERANK_TOP = 60


def white_line_mask(frame, out_size=None):
    """Painted lines: thin ridges brighter than both sides, close to white.

    Measured at full resolution, where a broadcast line is 3-5 px wide; at
    half size it blurs into the blue free zone and fails a whiteness test,
    while the wider pale strips between advertising letters survive. The
    result is max-pooled down to out_size (w, h).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    r = max(2, int(round(frame.shape[1] / 480)))   # 4 px at 1920 wide
    ridge = np.zeros_like(L)
    for dx, dy in ((r, 0), (0, r), (r, r), (r, -r)):
        a = np.roll(L, (dy, dx), (0, 1))
        b = np.roll(L, (-dy, -dx), (0, 1))
        np.maximum(ridge, np.minimum(L - a, L - b), out=ridge)
    m = ((ridge > 10) & (chroma < 25) & (L > 150)).astype(np.uint8) * 255
    if out_size is not None and (m.shape[1], m.shape[0]) != tuple(out_size):
        m = cv2.resize(m, tuple(out_size), interpolation=cv2.INTER_AREA)
        m = np.where(m > 40, 255, 0).astype(np.uint8)
    return m


def _segment_lines(mask):
    """Straight-line candidates as (a, b, c) with a^2 + b^2 = 1, plus support."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, 40, minLineLength=40, maxLineGap=8)
    if segs is None:
        return [], []
    segs = segs.reshape(-1, 4).astype(np.float64)
    out = []
    for x1, y1, x2, y2 in segs:
        L = np.hypot(x2 - x1, y2 - y1)
        a, b = (y2 - y1) / L, -(x2 - x1) / L
        c = -(a * x1 + b * y1)
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ang = (ang + 90.0) % 180.0 - 90.0
        out.append((a, b, c, L, ang, (x1 + x2) / 2, (y1 + y2) / 2))

    # Greedy merge: a segment joins a line when it is parallel and on it.
    out.sort(key=lambda s: -s[3])
    merged = []   # [a, b, c, support, angle]
    for a, b, c, L, ang, mx, my in out:
        for m in merged:
            d_ang = abs((ang - m[4] + 90.0) % 180.0 - 90.0)
            if d_ang < 3.0 and abs(m[0] * mx + m[1] * my + m[2]) < 4.0:
                m[3] += L
                break
        else:
            merged.append([a, b, c, L, ang])

    shallow = [m for m in merged if abs(m[4]) < SHALLOW_DEG]
    steep = [m for m in merged if abs(m[4]) >= SHALLOW_DEG]
    shallow.sort(key=lambda m: -m[3])
    steep.sort(key=lambda m: -m[3])
    return shallow[:MAX_SHALLOW], steep[:MAX_STEEP]


def _x_at(line, y):
    a, b, c = line[:3]
    return -(b * y + c) / a if abs(a) > 1e-9 else np.inf


def _y_at(line, x):
    a, b, c = line[:3]
    return -(a * x + c) / b if abs(b) > 1e-9 else np.inf


def _meet(l1, l2):
    p = np.cross(np.array(l1[:3]), np.array(l2[:3]))
    if abs(p[2]) < 1e-12:
        return None
    return p[:2] / p[2]


def camera_from_homography(H, img_w, img_h):
    """(focal px, camera centre (x, y, z) in centred court metres, rotation
    world->camera) or None.

    H maps centred court (x, z, 1) to image pixels. Square pixels and a centred
    principal point leave the focal length as the one intrinsic unknown; the
    two orthonormality constraints each give an estimate of it, and a view no
    real camera could produce makes them negative or disagree.
    """
    # Scalar arithmetic first: this runs on thousands of hypotheses per frame
    # and rejects most of them before any matrix work.
    cx, cy = img_w / 2.0, img_h / 2.0
    h31, h32 = float(H[2, 0]), float(H[2, 1])
    h11, h12 = float(H[0, 0]) - cx * h31, float(H[0, 1]) - cx * h32
    h21, h22 = float(H[1, 0]) - cy * h31, float(H[1, 1]) - cy * h32
    cands = []
    den = h31 * h32
    if abs(den) > 1e-18:
        cands.append(-(h11 * h12 + h21 * h22) / den)
    den = h32 * h32 - h31 * h31
    if abs(den) > 1e-18:
        cands.append(((h11 ** 2 + h21 ** 2) - (h12 ** 2 + h22 ** 2)) / den)
    cands = [c for c in cands if np.isfinite(c) and c > 1.0]
    if not cands:
        return None
    f = float(np.sqrt(np.mean(cands)))
    n1 = np.sqrt((h11 / f) ** 2 + (h21 / f) ** 2 + h31 ** 2)
    n2 = np.sqrt((h12 / f) ** 2 + (h22 / f) ** 2 + h32 ** 2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    # Unequal column norms = the two focal estimates disagree.
    if max(n1, n2) / min(n1, n2) > 1.25:
        return None
    h = H.astype(np.float64).copy()
    h[0, :] -= cx * h[2, :]
    h[1, :] -= cy * h[2, :]
    M = np.diag([1.0 / f, 1.0 / f, 1.0]) @ h
    lam = 2.0 / (n1 + n2)
    r1, r3, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    if t[2] < 0:
        r1, r3, t = -r1, -r3, -t
    r2 = np.array([r3[1] * r1[2] - r3[2] * r1[1],
                   r3[2] * r1[0] - r3[0] * r1[2],
                   r3[0] * r1[1] - r3[1] * r1[0]])
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    if not np.all(np.isfinite(C)):
        return None
    return f, C, R


def _plausible(H, img_w, img_h):
    cam = camera_from_homography(H, img_w, img_h)
    if cam is None:
        return None
    f, C, _R = cam
    fov_y = 2 * np.degrees(np.arctan(img_h / (2 * f)))
    if not (3.0 <= fov_y <= 90.0):
        return None
    # Above the floor, outside the court, not absurdly far.
    if not (2.0 <= C[1] <= 45.0):
        return None
    horiz = np.hypot(C[0], C[2])
    if not (8.0 <= horiz <= 160.0):
        return None
    if abs(C[0]) < HALF_W + 1.0 and abs(C[2]) < HALF_L + 1.0:
        return None
    return cam


def _score(H, dist, img_w, img_h, sigma=2.5):
    """How well the court model lands on white pixels (higher is better)."""
    p = H @ _SAMPLES_H
    w = p[2]
    ok = w > 1e-9
    u = np.where(ok, p[0] / np.where(ok, w, 1), -1)
    v = np.where(ok, p[1] / np.where(ok, w, 1), -1)
    vis = ok & (u >= 2) & (u < img_w - 2) & (v >= 2) & (v < img_h - 2)
    if vis.sum() < 40:
        return -1e9, 0
    d = dist[v[vis].astype(int), u[vis].astype(int)]
    hit = np.exp(-(d * d) / (2 * sigma * sigma))
    # Visible model line with nothing under it is evidence against: players
    # occlude some, so it costs less than a hit earns.
    return float(hit.sum() - 0.35 * (hit < 0.2).sum()), int(vis.sum())


def _project_pixels(H, pts, w, h):
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    ok = p[:, 2] > 1e-9
    u = p[:, 0] / np.where(ok, p[:, 2], 1)
    v = p[:, 1] / np.where(ok, p[:, 2], 1)
    vis = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[vis].astype(int), v[vis].astype(int)


def _boundary_contrast(H, lab):
    """Lab colour distance between the court and the free zone around it."""
    h, w = lab.shape[:2]
    ui, vi = _project_pixels(H, _INSIDE, w, h)
    uo, vo = _project_pixels(H, _OUTSIDE, w, h)
    if len(ui) < 20 or len(uo) < 20:
        return 0.0
    # Medians shrug off the players standing on the court.
    a = np.median(lab[vi, ui].astype(np.float32), axis=0)
    b = np.median(lab[vo, uo].astype(np.float32), axis=0)
    return float(np.linalg.norm(a - b))


def _side_contrast(H, lab):
    """Colour step across each visible boundary line, averaged by visibility.

    Measured line by line rather than court against surroundings, so a 3 m
    line mistaken for a boundary (front zone against back zone, often only a
    shade apart) does not borrow contrast from a real boundary elsewhere.
    """
    h, w = lab.shape[:2]
    vals, wts = [], []
    for inside, outside in _SIDES:
        ui, vi = _project_pixels(H, inside, w, h)
        uo, vo = _project_pixels(H, outside, w, h)
        if len(ui) < 8 or len(uo) < 8:
            continue
        a = np.median(lab[vi, ui].astype(np.float32), axis=0)
        b = np.median(lab[vo, uo].astype(np.float32), axis=0)
        vals.append(float(np.linalg.norm(a - b)))
        wts.append(min(len(ui), len(uo)))
    return float(np.average(vals, weights=wts)) if vals else 0.0


def _rank(score, H, lab):
    return max(score, 1.0) * _side_contrast(H, lab) ** 2 * _boundary_contrast(H, lab)


def _hypotheses(shallow, steep, img_w, img_h):
    """Yield (homography, view) for every line assignment worth scoring."""
    mid_x, mid_y = img_w / 2.0, img_h / 2.0
    z_pairs = list(itertools.combinations(TRANSVERSE_Z, 2))

    # Side camera at -x: far sideline (x=+4.5) is the upper shallow line,
    # z grows to the right.
    for s1, s2 in itertools.combinations(shallow, 2):
        top, bot = (s1, s2) if _y_at(s1, mid_x) < _y_at(s2, mid_x) else (s2, s1)
        if abs(_y_at(top, mid_x) - _y_at(bot, mid_x)) < 0.06 * img_h:
            continue
        for t1, t2 in itertools.combinations(steep, 2):
            left, right = (t1, t2) if _x_at(t1, mid_y) < _x_at(t2, mid_y) else (t2, t1)
            if abs(_x_at(left, mid_y) - _x_at(right, mid_y)) < 0.04 * img_w:
                continue
            pts = [_meet(top, left), _meet(top, right), _meet(bot, right), _meet(bot, left)]
            if any(p is None for p in pts):
                continue
            img = np.array(pts, np.float32)
            for za, zb in z_pairs:
                world = np.array([[HALF_W, za], [HALF_W, zb], [-HALF_W, zb], [-HALF_W, za]], np.float32)
                yield cv2.getPerspectiveTransform(world, img), "side"

    # Camera behind the +z baseline: sidelines are the steep lines (x=-4.5 on
    # the left), z grows towards the bottom of the picture.
    for t1, t2 in itertools.combinations(steep, 2):
        left, right = (t1, t2) if _x_at(t1, mid_y) < _x_at(t2, mid_y) else (t2, t1)
        if abs(_x_at(left, mid_y) - _x_at(right, mid_y)) < 0.06 * img_w:
            continue
        for s1, s2 in itertools.combinations(shallow, 2):
            top, bot = (s1, s2) if _y_at(s1, mid_x) < _y_at(s2, mid_x) else (s2, s1)
            if abs(_y_at(top, mid_x) - _y_at(bot, mid_x)) < 0.03 * img_h:
                continue
            pts = [_meet(top, left), _meet(top, right), _meet(bot, right), _meet(bot, left)]
            if any(p is None for p in pts):
                continue
            img = np.array(pts, np.float32)
            for za, zb in z_pairs:
                world = np.array([[-HALF_W, za], [HALF_W, za], [HALF_W, zb], [-HALF_W, zb]], np.float32)
                yield cv2.getPerspectiveTransform(world, img), "end"


def _homography_from_camera(f, R, C, img_w, img_h):
    """Court (x, z, 1) -> pixels for a camera with a centred principal point."""
    K = np.array([[f, 0, img_w / 2.0], [0, f, img_h / 2.0], [0, 0, 1.0]])
    return K @ np.stack([R[:, 0], R[:, 2], -R @ C], axis=1)


def _bilinear(img, u, v):
    h, w = img.shape
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = u.astype(int), v.astype(int)
    du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def refine_camera(H, dist, img_w, img_h, caps=(12.0, 5.0)):
    """Fit a physical camera (focal length, rotation, position) to the line
    pixels, starting from H. Returns the refined homography, or H when it has
    no camera reading.

    A full homography has 8 free parameters and, with only a couple of
    transverse lines in view, snaps to whichever stray line is nearest. A
    camera with square pixels and a centred principal point has 7, all
    physical, which is what keeps the fit on the court.
    """
    from scipy.optimize import least_squares

    cam = camera_from_homography(H, img_w, img_h)
    if cam is None:
        return H
    f, C, R = cam
    rvec, _ = cv2.Rodrigues(R)
    x0 = np.concatenate([rvec.ravel(), C, [np.log(f)]])

    def unpack(x):
        Rm, _ = cv2.Rodrigues(x[:3].reshape(3, 1))
        return _homography_from_camera(np.exp(x[6]), Rm, x[3:6], img_w, img_h)

    for cap in caps:
        def resid(x, cap=cap):
            p = unpack(x) @ _SAMPLES_H
            w = p[2]
            ok = w > 1e-9
            u = np.where(ok, p[0] / np.where(ok, w, 1), -1e6)
            v = np.where(ok, p[1] / np.where(ok, w, 1), -1e6)
            vis = (u >= 1) & (u < img_w - 2) & (v >= 1) & (v < img_h - 2)
            r = np.full(len(u), cap)
            r[vis] = np.minimum(_bilinear(dist, u[vis], v[vis]), cap)
            return r
        try:
            sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0,
                                diff_step=1e-4, max_nfev=60)
            x0 = sol.x
        except Exception:
            return H
    return unpack(x0)


def detect_court_lines(frame, debug=False):
    """Detect the court from its lines.

    Returns (corners, info) with corners as four [x, y] normalised to the
    frame, in the analyzer's order (centred court (-4.5,-9), (4.5,-9),
    (4.5,9), (-4.5,9)), or (None, reason).
    """
    ih, iw = frame.shape[:2]
    s = WORK_W / float(iw)
    img = cv2.resize(frame, (WORK_W, int(round(ih * s)))) if abs(s - 1) > 1e-6 else frame
    h, w = img.shape[:2]
    mask = white_line_mask(frame, (w, h))
    shallow, steep = _segment_lines(mask)
    if len(shallow) < 2 or len(steep) < 2:
        return None, f"too few lines (shallow {len(shallow)}, steep {len(steep)})"
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

    scored = []
    n_tried = 0
    for H, view in _hypotheses(shallow, steep, w, h):
        n_tried += 1
        if _plausible(H, w, h) is None:
            continue
        sc, _nvis = _score(H, dist, w, h)
        scored.append((sc, H, view))
    if not scored:
        return None, f"no plausible camera among {n_tried} line assignments"
    n_ok = len(scored)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    scored.sort(key=lambda r: -r[0])
    best = max(scored[:RERANK_TOP], key=lambda r: _rank(r[0], r[1], lab))
    _sc, H, view = best
    H = refine_camera(H, dist, w, h)
    cam = _plausible(H, w, h)
    if cam is None:
        return None, "refined court implies no plausible camera"
    sc, nvis = _score(H, dist, w, h)
    contrast = _boundary_contrast(H, lab)
    # Fraction of the visible model that sits on a painted line.
    quality = sc / max(1, nvis)
    if quality < 0.35:
        return None, f"court lines poorly supported (quality {quality:.2f})"

    S = np.diag([1.0 / w, 1.0 / h, 1.0])
    Hn = S @ H
    c = Hn @ np.concatenate([_CORNERS_CENTRED.T, np.ones((1, 4))])
    corners = [[round(float(c[0, i] / c[2, i]), 6), round(float(c[1, i] / c[2, i]), 6)] for i in range(4)]
    f, C, _R = cam
    info = {
        "view": view,
        "quality": round(float(quality), 3),
        "contrast": round(contrast, 1),
        "fov_y": round(float(2 * np.degrees(np.arctan(h / (2 * f)))), 2),
        "camera": [round(float(v), 2) for v in C],
        "hypotheses": n_ok,
    }
    if debug:
        info["mask"] = mask
        info["H"] = H
    return corners, info


def detect_court_lines_verbose(frame, tag=""):
    corners, info = detect_court_lines(frame)
    if corners is None:
        print(f"[court-lines]{tag} rejected: {info}", file=sys.stderr)
    return corners, info
