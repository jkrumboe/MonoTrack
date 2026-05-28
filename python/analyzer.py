#!/usr/bin/env python3
"""
Volleyball match video analyzer sidecar process.

Communicates with the Tauri shell via stdin/stdout as JSON messages.
Each line of stdout is a JSON object.

Protocol:
  Request:  { "action": "analyze", "video_path": "/path/to/match.mp4" }
  Response: per-frame results streamed as JSON lines, ending with a "complete" action.
"""

import json
import sys
import os
import time
import math
import subprocess
import base64

# Optional imports — gracefully degrade if CV libs aren't installed yet
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision, BaseOptions as MpBaseOptions
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Lazy-loaded CNN court detector (persists across calls within one process)
_court_detector = None
# Lazy-loaded YOLOv8 model for sports ball detection
_yolo_model = None
# Lazy-loaded Depth Anything V2 depth estimation pipeline
_depth_model = None
# Temporal smoothing state for court corners
_prev_corners = None
_smoothing_alpha = 0.4  # EMA weight for new detection (0=ignore new, 1=no smoothing)
# Cached video capture to avoid re-opening the same file
_cached_cap = None
_cached_cap_path = None
# Last successfully reprojected court keypoints (reused between detections)
_last_court_keypoints = None

def _get_court_detector():
    """CNN court detector is disabled — the trained model is tennis-specific and not used for volleyball.
    Volleyball court detection uses the Hough-based pipeline only.
    """
    return None


def _get_depth_model():
    """Lazy-load Depth Anything V2 Small model for monocular depth estimation.

    Returns a HuggingFace depth-estimation pipeline, or None if transformers/torch
    are not installed.  The model is downloaded on first call (~100 MB, cached by HF).
    Uses GPU if available, otherwise CPU.
    """
    global _depth_model
    if _depth_model is not None:
        return _depth_model
    if not HAS_TRANSFORMERS or not HAS_TORCH:
        print("[depth] transformers or torch not available — depth estimation disabled", file=sys.stderr)
        return None
    try:
        device = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
        print(f"[depth] Loading Depth-Anything-V2-Small on {'GPU' if device == 0 else 'CPU'}...", file=sys.stderr)
        _depth_model = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        print("[depth] Depth model loaded", file=sys.stderr)
        return _depth_model
    except Exception as e:
        print(f"[depth] Failed to load depth model: {e}", file=sys.stderr)
        return None


def _get_yolo_model():
    """Lazy-load YOLOv8 model for sports ball detection. Returns model or None."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO

        # Prefer explicit, normalized absolute candidates first.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_model = os.path.abspath(os.path.join(script_dir, os.pardir, "yolov8m.pt"))
        cwd_model = os.path.abspath("yolov8m.pt")
        candidates = [project_root_model, cwd_model, "yolov8m.pt"]

        model_path = None
        for candidate in candidates:
            if os.path.isfile(candidate):
                model_path = candidate
                break

        print(f"[yolo] Script dir: {script_dir}", file=sys.stderr)
        print(f"[yolo] Model candidates: {candidates}", file=sys.stderr)
        if model_path is None:
            print("[yolo] yolov8m.pt not found — ball detection disabled", file=sys.stderr)
            return None

        print(f"[yolo] Loading model from {model_path}...", file=sys.stderr)
        _yolo_model = YOLO(model_path)
        print(f"[yolo] Loaded model from {model_path}", file=sys.stderr)
        return _yolo_model
    except Exception as e:
        print(f"[yolo] Failed to load YOLO model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


# COCO class index for "sports ball" (includes volleyball)
_SPORTS_BALL_CLASS = 32


def _estimate_depth_at_ball(frame_bgr, cx_norm: float, cy_norm: float) -> float | None:
    """Run Depth Anything V2 on a single frame and return the relative depth at (cx_norm, cy_norm).

    The returned value is a *relative* depth (larger = farther from camera) normalised
    to [0, 1] within the frame.  It is scale-ambiguous; callers should align it to
    metric distance using a known reference (e.g. detected court corners).

    Returns None if the depth model is unavailable or inference fails.
    """
    try:
        from PIL import Image as PILImage
        depth_pipe = _get_depth_model()
        if depth_pipe is None:
            return None

        # Convert BGR → RGB PIL image
        rgb = frame_bgr[:, :, ::-1]
        img_h, img_w = rgb.shape[:2]
        pil_img = PILImage.fromarray(rgb)

        result = depth_pipe(pil_img)
        depth_map = np.array(result["depth"], dtype=np.float32)  # shape (H, W)

        # Normalise to [0, 1]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-5:
            return None
        depth_norm = (depth_map - d_min) / (d_max - d_min)

        # Sample with bilinear interpolation at ball centre
        px = cx_norm * (img_w - 1)
        py = cy_norm * (img_h - 1)
        x0, y0 = int(px), int(py)
        x1, y1 = min(x0 + 1, img_w - 1), min(y0 + 1, img_h - 1)
        fx, fy = px - x0, py - y0
        d = (depth_norm[y0, x0] * (1 - fx) * (1 - fy) +
             depth_norm[y0, x1] * fx * (1 - fy) +
             depth_norm[y1, x0] * (1 - fx) * fy +
             depth_norm[y1, x1] * fx * fy)
        return round(float(d), 5)
    except Exception as e:
        print(f"[depth] Inference error: {e}", file=sys.stderr)
        return None


def _estimate_depth_batch(frames_and_balls: list[tuple]) -> dict[int, float]:
    """Run depth estimation on a batch of (frame_idx, frame_bgr, cx_norm, cy_norm) tuples.

    Returns {frame_idx: relative_depth_0_to_1}.
    Skips gracefully if depth model is unavailable.
    """
    try:
        from PIL import Image as PILImage
        depth_pipe = _get_depth_model()
        if depth_pipe is None:
            return {}
    except ImportError:
        return {}

    results = {}
    total = len(frames_and_balls)
    for i, (frame_idx, frame_bgr, cx_norm, cy_norm) in enumerate(frames_and_balls):
        if i % 20 == 0:
            print(f"[depth] Estimating depth: {i}/{total}", file=sys.stderr)
        d = _estimate_depth_at_ball(frame_bgr, cx_norm, cy_norm)
        if d is not None:
            results[frame_idx] = d
    return results


def _detect_ball_yolo(frame, model, conf_thresh=0.20):
    """Detect sports ball in a frame using YOLOv8.

    Returns [cx_norm, cy_norm, conf, x1_norm, y1_norm, x2_norm, y2_norm]
    (all normalised 0–1) or None if no ball found.
    If multiple balls are detected the highest-confidence one is returned.
    """
    try:
        results = model.predict(frame, classes=[_SPORTS_BALL_CLASS], conf=conf_thresh,
                                verbose=False, imgsz=640)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx])
        h, w = frame.shape[:2]
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        return [round(cx, 6), round(cy, 6), round(conf, 4),
                round(x1 / w, 6), round(y1 / h, 6), round(x2 / w, 6), round(y2 / h, 6)]
    except Exception as e:
        print(f"[yolo-predict] Exception: {e}", file=sys.stderr)
        raise


def _interpolate_ball_path(raw_detections):
    """Fill in ball positions for frames where YOLO missed a detection.

    Uses cubic spline interpolation (scipy) when ≥4 anchors are available,
    otherwise falls back to linear interpolation between consecutive detections.
    Gaps larger than MAX_GAP frames are left empty (ball likely out of frame
    or mid-rally break).

    raw_detections: dict {frame_idx: [cx_norm, cy_norm, conf]}

    Returns dict {frame_idx: {'pos': [cx, cy], 'conf': float, 'interpolated': bool}}
    """
    MAX_GAP = 60  # ~2 s at 30 fps — don't interpolate across a rally break

    if not raw_detections:
        return {}

    result = {}
    frames_with_ball = sorted(raw_detections.keys())

    # Seed result with real detections
    for f, det in raw_detections.items():
        result[f] = {'pos': [det[0], det[1]], 'conf': det[2],
                     'bbox': [det[3], det[4], det[5], det[6]] if len(det) >= 7 else None,
                     'interpolated': False}

    # Build spline over all detected frames if scipy is available
    spline_done = False
    if len(frames_with_ball) >= 4:
        try:
            from scipy.interpolate import CubicSpline
            fi = [float(f) for f in frames_with_ball]
            xs = [raw_detections[f][0] for f in frames_with_ball]
            ys = [raw_detections[f][1] for f in frames_with_ball]
            cs_x = CubicSpline(fi, xs, extrapolate=False)
            cs_y = CubicSpline(fi, ys, extrapolate=False)

            for i in range(len(frames_with_ball) - 1):
                f_start = frames_with_ball[i]
                f_end = frames_with_ball[i + 1]
                gap = f_end - f_start
                if gap <= 1 or gap > MAX_GAP:
                    continue
                for f in range(f_start + 1, f_end):
                    cx = float(cs_x(f))
                    cy = float(cs_y(f))
                    result[f] = {
                        'pos': [round(max(0.0, min(1.0, cx)), 6),
                                round(max(0.0, min(1.0, cy)), 6)],
                        'conf': 0.0,
                        'bbox': None,
                        'interpolated': True,
                    }
            spline_done = True
        except Exception:
            pass

    if not spline_done:
        # Linear interpolation between consecutive anchor frames
        for i in range(len(frames_with_ball) - 1):
            f_start = frames_with_ball[i]
            f_end = frames_with_ball[i + 1]
            gap = f_end - f_start
            if gap <= 1 or gap > MAX_GAP:
                continue
            p_s = raw_detections[f_start]
            p_e = raw_detections[f_end]
            for f in range(f_start + 1, f_end):
                t = (f - f_start) / gap
                cx = p_s[0] + t * (p_e[0] - p_s[0])
                cy = p_s[1] + t * (p_e[1] - p_s[1])
                result[f] = {
                    'pos': [round(cx, 6), round(cy, 6)],
                    'conf': 0.0,
                    'bbox': None,
                    'interpolated': True,
                }

    return result


# ─── Volleyball Court Model (FIVB Standard) ─────────────────────────────
COURT_LENGTH = 18.0            # full court length baseline-to-baseline (m)
COURT_WIDTH = 9.0              # court width sideline-to-sideline (m)
NET_Y = COURT_LENGTH / 2      # net centre: 9.0 m
ATTACK_DIST = 3.0              # attack line distance from net (m)
NET_HEIGHT = 2.43              # standard men's net height (m)

# Internal court lines for Hough-based homography scoring.
# Including baselines improves scoring so the algorithm prefers the full court
# over attack-line-only hypotheses.
COURT_MODEL_LINES = [
    # Outer boundaries
    ((0, 0), (COURT_WIDTH, 0)),                                # far baseline
    ((0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),          # near baseline
    # Attack lines (3 m from net)
    ((0, NET_Y - ATTACK_DIST), (COURT_WIDTH, NET_Y - ATTACK_DIST)),
    ((0, NET_Y + ATTACK_DIST), (COURT_WIDTH, NET_Y + ATTACK_DIST)),
    # Sidelines
    ((0, 0), (0, COURT_LENGTH)),
    ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)),
    # Centre line under net
    ((0, NET_Y), (COURT_WIDTH, NET_Y)),
]

# 10 court keypoints in physical court coordinates (meters).
# Origin: far-left corner; x = rightward, y = toward near baseline.
COURT_KPS_PHYSICAL = [
    [0, 0],                                    # 0: far-left corner
    [COURT_WIDTH, 0],                          # 1: far-right corner
    [0, COURT_LENGTH],                         # 2: near-left corner
    [COURT_WIDTH, COURT_LENGTH],               # 3: near-right corner
    [0, NET_Y - ATTACK_DIST],                  # 4: far attack line left
    [COURT_WIDTH, NET_Y - ATTACK_DIST],        # 5: far attack line right
    [0, NET_Y + ATTACK_DIST],                  # 6: near attack line left
    [COURT_WIDTH, NET_Y + ATTACK_DIST],        # 7: near attack line right
    [0, NET_Y],                                # 8: net left
    [COURT_WIDTH, NET_Y],                      # 9: net right
]

# 4 court corners in the order returned by corner detection:
# far-left, far-right, near-right, near-left.
_CORNERS_PHYSICAL = [
    [0, 0],
    [COURT_WIDTH, 0],
    [COURT_WIDTH, COURT_LENGTH],
    [0, COURT_LENGTH],
]


def _reproject_keypoints_from_corners(corners_norm, img_h, img_w):
    """Reproject all 8 court keypoints from 4 smoothed corners via homography.

    corners_norm: list of 4 [x_norm, y_norm] in order [far-left, far-right,
                  near-right, near-left] (normalized to [0,1]).
    Returns list of 8 [x_norm, y_norm] or None on failure.
    """
    src_pts = np.array(_CORNERS_PHYSICAL, dtype=np.float32)
    dst_pts = np.array(
        [[c[0] * img_w, c[1] * img_h] for c in corners_norm],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None
    kps_phys = np.array(COURT_KPS_PHYSICAL, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(kps_phys, H).reshape(-1, 2)
    return [[round(float(p[0]) / img_w, 6), round(float(p[1]) / img_h, 6)] for p in projected]


def _get_player_img_pos(pose_lms):
    """Extract normalised [0,1] 2-D image position from a pose landmark list.

    pose_lms: list of [x, y, z] where x,y are MediaPipe-normalised image coords.
    Returns [x, y] or None.
    """
    if not pose_lms or len(pose_lms) < 25:
        return None
    # Prefer average ankle position (landmarks 27, 28)
    if len(pose_lms) > 28:
        ax = (pose_lms[27][0] + pose_lms[28][0]) / 2
        ay = (pose_lms[27][1] + pose_lms[28][1]) / 2
        if 0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0:
            return [round(ax, 4), round(ay, 4)]
    # Fall back to hip midpoint (landmarks 23, 24)
    ax = (pose_lms[23][0] + pose_lms[24][0]) / 2
    ay = (pose_lms[23][1] + pose_lms[24][1]) / 2
    if 0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0:
        return [round(ax, 4), round(ay, 4)]
    return None


def _is_in_court_quad(px_norm, py_norm, corners_norm, img_w, img_h, margin_norm=0.12):
    """Return True if the normalised image point is inside or near the court quad.

    corners_norm: 4×2 list [[x,y], ...] normalised [0,1],
                  order: far-left, far-right, near-right, near-left.
    margin_norm:  extra tolerance as a fraction of min(img_w, img_h).
    """
    pts = np.array([[c[0] * img_w, c[1] * img_h] for c in corners_norm], dtype=np.float32)
    test_pt = (float(px_norm * img_w), float(py_norm * img_h))
    dist = cv2.pointPolygonTest(pts.reshape(-1, 1, 2), test_pt, measureDist=True)
    margin_px = margin_norm * min(img_w, img_h)
    return float(dist) >= -margin_px


def _assign_players_from_court(results, corners_norm, img_w, img_h):
    """Filter MediaPipe results to poses whose feet are inside the court quad.

    Players are sorted by ankle image-y so the nearest-camera player
    (highest y in image) becomes P1 and the far-baseline player (lowest y)
    becomes P2.  This excludes ball boys, line judges, and spectators.

    Returns (pose_p1, world_p1, pose_p2, world_p2) — any value may be None
    if fewer than the expected number of in-court players are detected.
    """
    if not results or not results.pose_landmarks:
        return None, None, None, None

    candidates = []
    for i, lms in enumerate(results.pose_landmarks):
        n = len(lms)
        # Use ankle midpoint as the ground-contact reference point
        if n > 28:
            ax = (lms[27].x + lms[28].x) / 2
            ay = (lms[27].y + lms[28].y) / 2
        elif n > 24:
            ax = (lms[23].x + lms[24].x) / 2
            ay = (lms[23].y + lms[24].y) / 2
        else:
            continue

        if not _is_in_court_quad(ax, ay, corners_norm, img_w, img_h):
            continue

        wlms = None
        if results.pose_world_landmarks and i < len(results.pose_world_landmarks):
            wlms = results.pose_world_landmarks[i]
        candidates.append({'ay': ay, 'lms': lms, 'wlms': wlms})

    # Highest image-y = near baseline = P1; lowest = far baseline = P2
    candidates.sort(key=lambda c: c['ay'], reverse=True)

    def _extract(c):
        if c is None:
            return None, None
        pose = [[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)] for lm in c['lms']]
        world = ([[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)] for lm in c['wlms']]
                 if c['wlms'] else None)
        return pose, world

    p1_pose, p1_world = _extract(candidates[0] if len(candidates) > 0 else None)
    p2_pose, p2_world = _extract(candidates[1] if len(candidates) > 1 else None)
    return p1_pose, p1_world, p2_pose, p2_world


def send(obj: dict):
    """Send a JSON message to stdout (Tauri reads this)."""
    print(json.dumps(obj), flush=True)


def send_error(message: str):
    send({"action": "error", "message": message})


def send_progress(percent: int, message: str):
    send({"action": "progress", "percent": percent, "message": message})


# ─── Court Detection Functions ──────────────────────────────────────────

def _detect_court_surface(hsv):
    """Detect court surface via HSV color segmentation. Returns dilated binary mask."""
    masks = {
        'green': cv2.inRange(hsv, (30, 25, 40), (85, 255, 255)),
        'teal':  cv2.inRange(hsv, (78, 35, 50), (100, 255, 255)),   # FIVB teal/cyan courts
        'blue':  cv2.inRange(hsv, (95, 30, 40), (130, 255, 255)),
        'clay':  cv2.inRange(hsv, (5, 50, 80), (25, 255, 255)),
    }
    mask = max(masks.values(), key=cv2.countNonZero)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels >= 2:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    return cv2.dilate(mask, dk)


def _detect_white_lines(hsv, court_mask):
    """Detect white court lines within the court region."""
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    white = cv2.bitwise_and(white, court_mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(white, cv2.MORPH_OPEN, k)


def _merge_segments(segments, angle_thresh=8, dist_thresh=20, gap_thresh=50):
    """Merge nearby parallel line segments into representative lines.
    
    Only merges segments that are:
    - Within angle_thresh degrees of each other
    - Within dist_thresh pixels perpendicular distance
    - Overlapping or within gap_thresh pixels along the line direction
    """
    if not segments:
        return []

    infos = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        infos.append((seg, angle, length, mx, my))

    infos.sort(key=lambda x: x[2], reverse=True)
    used = [False] * len(infos)
    merged = []

    for i in range(len(infos)):
        if used[i]:
            continue
        used[i] = True
        group = [infos[i]]

        # Direction vector of the anchor segment
        seg_i = infos[i][0]
        dxi, dyi = seg_i[2] - seg_i[0], seg_i[3] - seg_i[1]
        li = max(infos[i][2], 1e-6)
        # Unit direction and normal
        ux, uy = dxi / li, dyi / li
        nx, ny = -uy, ux

        # Projection range of anchor segment along its direction
        proj_i_1 = seg_i[0] * ux + seg_i[1] * uy
        proj_i_2 = seg_i[2] * ux + seg_i[3] * uy
        grp_min = min(proj_i_1, proj_i_2)
        grp_max = max(proj_i_1, proj_i_2)

        for j in range(i + 1, len(infos)):
            if used[j]:
                continue
            da = abs(infos[i][1] - infos[j][1])
            if da > 90:
                da = 180 - da
            if da > angle_thresh:
                continue

            # Perpendicular distance between midpoints
            perp = abs(nx * (infos[j][3] - infos[i][3]) + ny * (infos[j][4] - infos[i][4]))
            if perp > dist_thresh:
                continue

            # Parallel gap check: project segment j onto anchor direction
            seg_j = infos[j][0]
            proj_j_1 = seg_j[0] * ux + seg_j[1] * uy
            proj_j_2 = seg_j[2] * ux + seg_j[3] * uy
            j_min = min(proj_j_1, proj_j_2)
            j_max = max(proj_j_1, proj_j_2)

            # Gap = distance between nearest endpoints along the direction
            gap = max(0, max(j_min - grp_max, grp_min - j_max))
            if gap > gap_thresh:
                continue

            group.append(infos[j])
            used[j] = True
            # Expand the group's projection range
            grp_min = min(grp_min, j_min)
            grp_max = max(grp_max, j_max)

        # Build merged line from group's extreme projected points
        angle_rad = math.radians(group[0][1])
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        all_pts = []
        for g in group:
            s = g[0]
            all_pts.append((s[0], s[1]))
            all_pts.append((s[2], s[3]))
        projs = sorted(all_pts, key=lambda p: p[0] * dx + p[1] * dy)
        p1, p2 = projs[0], projs[-1]
        merged.append([p1[0], p1[1], p2[0], p2[1]])

    return merged


def _classify_lines(lines):
    """Split lines into horizontal-ish and vertical-ish groups."""
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        if angle < 45:
            h_lines.append(line)
        else:
            v_lines.append(line)
    return h_lines, v_lines


def _line_intersection(l1, l2):
    """Intersection of two infinite lines defined by segment endpoints."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    a1, b1 = y2 - y1, x1 - x2
    c1 = x2 * y1 - x1 * y2
    a2, b2 = y4 - y3, x3 - x4
    c2 = x4 * y3 - x3 * y4
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    return [(b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det]


def _score_hypothesis(corners, white_mask, img_h, img_w):
    """Score a court model hypothesis by projecting internal lines onto white pixels."""
    model_pts = np.array([
        [0, 0], [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]
    ], dtype=np.float32)
    img_pts = np.array(corners, dtype=np.float32)

    H, _ = cv2.findHomography(model_pts, img_pts)
    if H is None:
        return 0.0

    total, hits = 0, 0
    for (sx, sy), (ex, ey) in COURT_MODEL_LINES:
        p1 = cv2.perspectiveTransform(
            np.array([[[sx, sy]]], dtype=np.float32), H)[0][0]
        p2 = cv2.perspectiveTransform(
            np.array([[[ex, ey]]], dtype=np.float32), H)[0][0]
        for t in np.linspace(0, 1, 12):
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < img_w and 0 <= py < img_h:
                total += 1
                r0, r1 = max(0, py - 2), min(img_h, py + 3)
                c0, c1 = max(0, px - 2), min(img_w, px + 3)
                if white_mask[r0:r1, c0:c1].any():
                    hits += 1

    if total == 0:
        return 0.0

    hit_rate = hits / total
    area = cv2.contourArea(img_pts)
    # Uncapped area weight: larger quads score strictly higher than smaller ones.
    # This prevents attack-zone hypotheses (covering ~40% of frame) from beating
    # the full-court hypothesis (~60% of frame) even when hit_rate is similar.
    area_ratio = area / (img_h * img_w)
    area_weight = (area_ratio / 0.35) ** 2  # no cap
    return hit_rate * area_weight


def _find_best_court(h_lines, v_lines, white_mask, img_h, img_w):
    """Try combinations of 2 horizontal + 2 vertical lines, return best court corners."""
    from itertools import combinations

    def seg_len(l):
        return math.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2)

    h_top = sorted(h_lines, key=seg_len, reverse=True)[:7]
    v_top = sorted(v_lines, key=seg_len, reverse=True)[:7]
    best_score, best_corners = 0, None

    for h1, h2 in combinations(h_top, 2):
        y1 = (h1[1] + h1[3]) / 2
        y2 = (h2[1] + h2[3]) / 2
        if abs(y1 - y2) < img_h * 0.35:
            continue
        top, bot = (h1, h2) if y1 < y2 else (h2, h1)

        # In broadcast view the far (top) baseline is in the upper ~35% of
        # the frame. Reject any "top" line below that — it's an attack line.
        if (top[1] + top[3]) / 2 > img_h * 0.38:
            continue

        for v1, v2 in combinations(v_top, 2):
            tl = _line_intersection(top, v1)
            tr = _line_intersection(top, v2)
            bl = _line_intersection(bot, v1)
            br = _line_intersection(bot, v2)

            if any(p is None for p in (tl, tr, bl, br)):
                continue

            if tl[0] > tr[0]:
                tl, tr = tr, tl
                bl, br = br, bl

            corners = [tl, tr, br, bl]

            # Allow corners off-screen, especially the near (bottom) baseline
            # which is frequently cut off in broadcast views.
            margin_x = img_w * 0.20
            margin_top = img_h * 0.15
            margin_bot = img_h * 0.55
            skip = False
            for c in corners:
                if c[0] < -margin_x or c[0] > img_w + margin_x:
                    skip = True; break
                if c[1] < -margin_top:
                    skip = True; break
                if c[1] > img_h + margin_bot:
                    skip = True; break
            if skip:
                continue

            contour = np.array(corners, dtype=np.float32)
            if cv2.contourArea(contour) < 100:
                continue

            # Perspective check: in broadcast view, top (far) edge should
            # be narrower than bottom (near) edge
            top_width = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
            bot_width = math.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
            if top_width > bot_width * 1.1:  # allow small tolerance
                continue

            score = _score_hypothesis(corners, white_mask, img_h, img_w)
            if score > best_score:
                best_score = score
                best_corners = corners

    return best_corners, best_score


def _refine_corners(corners, white_mask, img_h, img_w):
    """Refine raw line-intersection corners using homography reprojection.
    
    The line intersections can drift when segment angles are slightly off.
    Fitting a homography and reprojecting the model corners gives 
    geometrically consistent results.
    """
    model_pts = np.array([
        [0, 0], [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]
    ], dtype=np.float32)
    img_pts = np.array(corners, dtype=np.float32)

    H, _ = cv2.findHomography(model_pts, img_pts)
    if H is None:
        return corners

    # Reproject the 4 model corners through the homography
    refined = []
    for pt in model_pts:
        proj = cv2.perspectiveTransform(
            np.array([[[pt[0], pt[1]]]], dtype=np.float32), H)[0][0]
        refined.append([float(proj[0]), float(proj[1])])

    return refined


def _img_to_base64(img):
    """Encode a BGR or grayscale image to a base64 data URL (JPEG for speed)."""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode('ascii')


def _draw_lines_on(img, lines, color=(0, 255, 0), thickness=2):
    """Draw line segments on an image copy."""
    vis = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return vis


def _draw_corners_on(img, corners, labels=None):
    """Draw corner points and optional labels on an image copy."""
    vis = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, c in enumerate(corners):
        pt = (int(c[0]), int(c[1]))
        cv2.circle(vis, pt, 8, colors[i % 4], -1)
        cv2.circle(vis, pt, 8, (255, 255, 255), 2)
        if labels and i < len(labels):
            cv2.putText(vis, labels[i], (pt[0] + 12, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def _draw_model_overlay(img, corners):
    """Draw projected court model lines over the image."""
    model_pts = np.array([
        [0, 0], [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]
    ], dtype=np.float32)
    img_pts = np.array(corners, dtype=np.float32)
    H, _ = cv2.findHomography(model_pts, img_pts)
    if H is None:
        return img.copy()

    vis = img.copy()

    # Draw outer boundary
    boundary = [(0, 0), (COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH)]
    for i in range(4):
        p1 = cv2.perspectiveTransform(np.array([[[boundary[i][0], boundary[i][1]]]], dtype=np.float32), H)[0][0]
        p2 = cv2.perspectiveTransform(np.array([[[boundary[(i + 1) % 4][0], boundary[(i + 1) % 4][1]]]], dtype=np.float32), H)[0][0]
        cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)

    # Draw internal lines
    line_colors = [(0, 255, 255), (0, 255, 255), (255, 0, 255), (255, 128, 0), (255, 128, 0)]
    line_labels = ["Far service", "Near service", "Center", "L singles", "R singles"]
    for idx, ((sx, sy), (ex, ey)) in enumerate(COURT_MODEL_LINES):
        p1 = cv2.perspectiveTransform(np.array([[[sx, sy]]], dtype=np.float32), H)[0][0]
        p2 = cv2.perspectiveTransform(np.array([[[ex, ey]]], dtype=np.float32), H)[0][0]
        color = line_colors[idx % len(line_colors)]
        cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
        mx, my = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
        cv2.putText(vis, line_labels[idx], (mx - 30, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Net line
    net_l = cv2.perspectiveTransform(np.array([[[0, NET_Y]]], dtype=np.float32), H)[0][0]
    net_r = cv2.perspectiveTransform(np.array([[[COURT_WIDTH, NET_Y]]], dtype=np.float32), H)[0][0]
    cv2.line(vis, (int(net_l[0]), int(net_l[1])), (int(net_r[0]), int(net_r[1])), (0, 0, 255), 2)
    cv2.putText(vis, "Net", (int((net_l[0] + net_r[0]) / 2) - 15, int((net_l[1] + net_r[1]) / 2) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return vis


def detect_court_corners_debug(frame):
    """
    Detect volleyball court corners with full debug output.
    Returns (corners_normalized, debug_steps, log_lines).
    corners_normalized: [[x, y], ...] in [0,1] or None.
    debug_steps: [{"label": str, "info": str, "image": base64_str}, ...]
    log_lines: [str, ...]
    """
    t0 = time.time()
    img_h, img_w = frame.shape[:2]
    steps = []
    log = []

    # Step 1: Input frame
    steps.append({
        "label": "1. Input Frame",
        "info": f"{img_w}×{img_h}",
        "image": _img_to_base64(frame),
    })

    # Step 2: HSV + court surface mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    court_mask = _detect_court_surface(hsv)
    court_px = cv2.countNonZero(court_mask)
    court_pct = court_px / (img_h * img_w) * 100
    log.append(f"Court mask: {court_px} px ({court_pct:.1f}%)")

    mask_vis = cv2.bitwise_and(frame, frame, mask=court_mask)
    steps.append({
        "label": "2. Court Region",
        "info": f"{court_px} px ({court_pct:.1f}% of frame)",
        "image": _img_to_base64(mask_vis),
    })

    if court_px < img_h * img_w * 0.05:
        log.append("FAIL: Court region too small (<5% of frame)")
        return None, steps, log

    # Step 3: White line mask
    white_mask = _detect_white_lines(hsv, court_mask)
    white_px = cv2.countNonZero(white_mask)
    log.append(f"White line pixels: {white_px}")

    white_vis = np.zeros_like(frame)
    white_vis[white_mask > 0] = (255, 255, 255)
    steps.append({
        "label": "3. White Line Pixels",
        "info": f"{white_px} pixels",
        "image": _img_to_base64(white_vis),
    })

    if white_px < 100:
        log.append("FAIL: Too few white pixels (<100)")
        return None, steps, log

    # Step 4: Hough line segments
    lines = cv2.HoughLinesP(
        white_mask, rho=1, theta=np.pi / 180,
        threshold=50,
        minLineLength=int(min(img_w, img_h) * 0.04),
        maxLineGap=int(min(img_w, img_h) * 0.03),
    )
    if lines is None:
        log.append("FAIL: HoughLinesP returned no lines")
        steps.append({"label": "4. Hough Lines", "info": "0 segments", "image": _img_to_base64(frame)})
        return None, steps, log

    segments = [l[0].tolist() for l in lines]
    log.append(f"Hough segments: {len(segments)}")

    hough_vis = _draw_lines_on(frame, segments, color=(0, 255, 0), thickness=1)
    steps.append({
        "label": "4. Hough Lines",
        "info": f"{len(segments)} raw segments",
        "image": _img_to_base64(hough_vis),
    })

    if len(segments) < 4:
        log.append(f"FAIL: Too few Hough segments ({len(segments)} < 4)")
        return None, steps, log

    # Step 5: Merged lines
    merged = _merge_segments(segments, angle_thresh=8,
                             dist_thresh=max(img_w, img_h) * 0.02,
                             gap_thresh=max(img_w, img_h) * 0.05)
    log.append(f"Merged lines: {len(merged)}")

    merged_vis = _draw_lines_on(frame, merged, color=(0, 200, 255), thickness=2)
    steps.append({
        "label": "5. Merged Lines",
        "info": f"{len(merged)} lines",
        "image": _img_to_base64(merged_vis),
    })

    if len(merged) < 4:
        log.append(f"FAIL: Too few merged lines ({len(merged)} < 4)")
        return None, steps, log

    # Step 6: Classified lines (H vs V)
    h_lines, v_lines = _classify_lines(merged)
    log.append(f"Classified: {len(h_lines)} horizontal, {len(v_lines)} vertical")

    class_vis = frame.copy()
    for x1, y1, x2, y2 in h_lines:
        cv2.line(class_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for x1, y1, x2, y2 in v_lines:
        cv2.line(class_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    steps.append({
        "label": "6. Classified (H=red, V=blue)",
        "info": f"{len(h_lines)}H + {len(v_lines)}V",
        "image": _img_to_base64(class_vis),
    })

    if len(h_lines) < 2 or len(v_lines) < 2:
        log.append(f"FAIL: Need ≥2 H and ≥2 V lines (got {len(h_lines)}H, {len(v_lines)}V)")
        return None, steps, log

    # Step 7: Best court model
    corners, score = _find_best_court(h_lines, v_lines, white_mask, img_h, img_w)
    log.append(f"Best Hough court score: {score:.4f}")
    method = "hough"

    if corners is None or score < 0.1:
        log.append(f"Hough failed (score={score:.4f}) — trying mask convex hull fallback...")
        corners = _detect_court_from_mask(court_mask, img_h, img_w)
        method = "mask"
        if corners is not None:
            log.append("Mask convex hull succeeded")
        else:
            log.append("FAIL: Both Hough and mask methods found no court")
            steps.append({"label": "7. Selected Court", "info": "No valid court", "image": _img_to_base64(frame)})
            return None, steps, log

    corner_labels = ["TL(far-L)", "TR(far-R)", "BR(near-R)", "BL(near-L)"]
    corners_vis = _draw_corners_on(frame, corners, corner_labels)
    # Draw quadrilateral
    pts = np.array([[int(c[0]), int(c[1])] for c in corners], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(corners_vis, [pts], True, (0, 255, 255), 2)
    steps.append({
        "label": f"7. Raw Corners ({method})",
        "info": f"method={method}, score={score:.4f}",
        "image": _img_to_base64(corners_vis),
    })

    for i, c in enumerate(corners):
        log.append(f"  Raw {corner_labels[i]}: ({c[0]:.1f}, {c[1]:.1f})")

    # Step 8: Refine corners via homography reprojection
    corners = _refine_corners(corners, white_mask, img_h, img_w)
    log.append("Refined corners via homography")

    refined_vis = _draw_corners_on(frame, corners, corner_labels)
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(refined_vis, [pts], True, (0, 255, 255), 2)
    steps.append({
        "label": "8. Refined Corners",
        "info": "homography-corrected",
        "image": _img_to_base64(refined_vis),
    })

    for i, c in enumerate(corners):
        log.append(f"  Refined {corner_labels[i]}: ({c[0]:.1f}, {c[1]:.1f})")

    # Step 9: Model overlay
    overlay_vis = _draw_model_overlay(frame, corners)
    overlay_vis = _draw_corners_on(overlay_vis, corners, corner_labels)
    steps.append({
        "label": "9. Court Model Overlay",
        "info": f"score={score:.4f}",
        "image": _img_to_base64(overlay_vis),
    })

    elapsed = (time.time() - t0) * 1000
    log.append(f"SUCCESS: Detection complete in {elapsed:.0f}ms")

    normalized = [[round(c[0] / img_w, 6), round(c[1] / img_h, 6)] for c in corners]
    return normalized, steps, log


def _smooth_corners(new_corners):
    """Apply EMA smoothing to detected corners for temporal stability."""
    global _prev_corners
    if _prev_corners is None:
        _prev_corners = new_corners
        return new_corners

    smoothed = []
    for (nx, ny), (px, py) in zip(new_corners, _prev_corners):
        sx = _smoothing_alpha * nx + (1 - _smoothing_alpha) * px
        sy = _smoothing_alpha * ny + (1 - _smoothing_alpha) * py
        smoothed.append([round(sx, 6), round(sy, 6)])
    _prev_corners = smoothed
    return smoothed


def _detect_court_from_mask(court_mask, img_h, img_w):
    """Estimate court corners from the convex hull of the court surface mask.

    Works even when the far baseline is obscured by the net/players, because
    the teal/green court colour is still visible behind the net.

    Returns 4 corners [[x,y], ...] in pixel coords (far-L, far-R, near-R, near-L)
    or None on failure.
    """
    cnts, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < img_h * img_w * 0.08:
        return None

    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)

    # Select 4 extreme corners using coordinate sums/differences
    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])].tolist()   # min x+y  → far-left
    tr = hull[np.argmax(hull[:, 0] - hull[:, 1])].tolist()   # max x-y  → far-right
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])].tolist()   # max x+y  → near-right
    bl = hull[np.argmin(hull[:, 0] - hull[:, 1])].tolist()   # min x-y  → near-left

    corners = [tl, tr, br, bl]

    # Validate: minimum area and perspective (top edge narrower than bottom)
    pts = np.array(corners, dtype=np.float32)
    if cv2.contourArea(pts) < img_h * img_w * 0.08:
        return None
    top_w = math.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    bot_w = math.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    if top_w > bot_w * 1.20:   # top edge must not be wider than bottom
        return None

    return corners


def _detect_court_hough_fast(frame):
    """Court detection: Hough first, falls back to surface-mask convex hull.

    The Hough path requires the far baseline to be detected as a line
    (it must be in the upper 48% of the frame). If that fails the mask
    convex-hull approach is used — it works as long as the teal/green
    court surface is cleanly segmented.

    Returns corners_normalized [[x,y]×4] in [0,1] (far-L, far-R, near-R,
    near-L) or None.
    """
    img_h, img_w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    court_mask = _detect_court_surface(hsv)
    if cv2.countNonZero(court_mask) < img_h * img_w * 0.05:
        return None

    white_mask = _detect_white_lines(hsv, court_mask)
    corners = None

    # --- Attempt 1: Hough line detection ---
    if cv2.countNonZero(white_mask) >= 100:
        lines = cv2.HoughLinesP(
            white_mask, rho=1, theta=np.pi / 180,
            threshold=50,
            minLineLength=int(min(img_w, img_h) * 0.04),
            maxLineGap=int(min(img_w, img_h) * 0.03),
        )
        if lines is not None and len(lines) >= 4:
            segments = [l[0].tolist() for l in lines]
            merged = _merge_segments(
                segments, angle_thresh=8,
                dist_thresh=max(img_w, img_h) * 0.02,
                gap_thresh=max(img_w, img_h) * 0.05,
            )
            if len(merged) >= 4:
                h_lines, v_lines = _classify_lines(merged)
                if len(h_lines) >= 2 and len(v_lines) >= 2:
                    c, score = _find_best_court(h_lines, v_lines, white_mask, img_h, img_w)
                    if c is not None and score >= 0.08:
                        corners = c

    # --- Attempt 2: court surface mask convex hull ---
    if corners is None:
        corners = _detect_court_from_mask(court_mask, img_h, img_w)

    if corners is None:
        return None

    corners = _refine_corners(corners, white_mask, img_h, img_w)
    corners = _smooth_corners(corners)
    return [[round(c[0] / img_w, 6), round(c[1] / img_h, 6)] for c in corners]


def _detect_court_cnn(frame):
    """Try CNN-based court detection. Returns (corners_normalized, log) or (None, log)."""
    log = []
    detector = _get_court_detector()
    if detector is None:
        log.append("CNN detector not available (torch not installed or model missing)")
        return None, log

    t0 = time.time()
    try:
        keypoints, corners = detector.detect(frame, use_homography=True)
        elapsed = (time.time() - t0) * 1000

        n_detected = sum(1 for p in keypoints if p[0] is not None)
        log.append(f"CNN detected {n_detected}/14 keypoints in {elapsed:.0f}ms")

        if corners is not None:
            corners = _smooth_corners(corners)
            log.append(f"CNN court corners (smoothed): {corners}")
            return corners, log
        else:
            log.append("CNN: not enough keypoints for court corners")
            return None, log
    except Exception as e:
        log.append(f"CNN detection error: {e}")
        return None, log


def _get_video_capture(video_path):
    """Get a cached VideoCapture for the given path, or create a new one."""
    global _cached_cap, _cached_cap_path
    if _cached_cap is not None and _cached_cap_path == video_path and _cached_cap.isOpened():
        return _cached_cap
    if _cached_cap is not None:
        _cached_cap.release()
    _cached_cap = cv2.VideoCapture(video_path)
    _cached_cap_path = video_path
    return _cached_cap


def handle_detect_court(video_path, frame_time=0.0):
    """Handle detect_court action: read a frame and run court detection."""
    if not os.path.isfile(video_path):
        send({"action": "error", "message": f"File not found: {video_path}"})
        return

    t_total = time.time()

    cap = _get_video_capture(video_path)
    if not cap.isOpened():
        send({"action": "error", "message": f"Cannot open video: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = int(frame_time * fps)
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    t_seek = time.time()
    ret, frame = cap.read()

    if not ret:
        send({"action": "error", "message": f"Cannot read frame at time {frame_time:.2f}s"})
        return

    t_read = time.time()
    read_ms = (t_read - t_seek) * 1000

    # Try CNN-based detection first (much more robust)
    corners, cnn_log = _detect_court_cnn(frame)
    t_detect = time.time()
    detect_ms = (t_detect - t_read) * 1000
    total_ms = (t_detect - t_total) * 1000

    if corners is not None:
        send({
            "action": "court_detected",
            "corners": corners,
            "steps": [],
            "log": cnn_log,
            "method": "cnn",
            "timing": {
                "read_ms": round(read_ms, 1),
                "detect_ms": round(detect_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        })
        return

    # Fall back to Hough-based detection
    corners, steps, log = detect_court_corners_debug(frame)
    t_hough = time.time()
    hough_ms = (t_hough - t_detect) * 1000
    total_ms = (t_hough - t_total) * 1000

    log = cnn_log + ["Falling back to Hough-based detection..."] + log
    if corners is not None:
        send({
            "action": "court_detected",
            "corners": corners,
            "steps": steps,
            "log": log,
            "method": "hough",
            "timing": {
                "read_ms": round(read_ms, 1),
                "detect_ms": round(hough_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        })
    else:
        send({"action": "court_failed", "steps": steps, "log": log,
              "message": "Court detection failed - see debug steps"})


def get_video_info(video_path: str):
    """
    Get video frame count and FPS.
    Tries cv2 first, then ffprobe, returns (total_frames, fps) or (None, None).
    """
    # Method 1: OpenCV
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                if total > 0:
                    return total, fps
        except Exception:
            pass

    # Method 2: ffprobe
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', video_path],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    fps_str = stream.get('r_frame_rate', '30/1')
                    parts = fps_str.split('/')
                    fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])
                    nb_frames = stream.get('nb_frames')
                    if nb_frames and nb_frames != 'N/A':
                        return int(nb_frames), fps
                    # Estimate from duration
                    duration = stream.get('duration')
                    if not duration or duration == 'N/A':
                        duration = data.get('format', {}).get('duration')
                    if duration and duration != 'N/A':
                        return int(float(duration) * fps), fps
    except (FileNotFoundError, Exception):
        pass

    return None, None


def generate_dummy_pose(frame_idx: int, court_x: float, court_z: float):
    """
    Generate a realistic 33-landmark skeleton (MediaPipe layout) in world coords.
    Landmarks are in meters: X = sideline axis, Y = up, Z = baseline axis.

    MediaPipe landmark indices:
      0:nose 1-4:eyes 5-8:ears 9-10:mouth
      11:L_shoulder 12:R_shoulder 13:L_elbow 14:R_elbow
      15:L_wrist 16:R_wrist 17-22:hands
      23:L_hip 24:R_hip 25:L_knee 26:R_knee
      27:L_ankle 28:R_ankle 29-32:feet
    """
    t = frame_idx * 0.08  # animation time

    # Base position on court + slight lateral sway
    bx = court_x + math.sin(t * 0.7) * 0.4
    bz = court_z + math.cos(t * 0.5) * 0.3

    # Body dimensions (meters)
    shoulder_w = 0.4   # half shoulder width
    hip_w = 0.2        # half hip width
    upper_arm = 0.30
    forearm = 0.28
    thigh = 0.45
    shin = 0.45

    # Animated arm swing and knee bend
    arm_swing = math.sin(t * 2.0) * 0.5
    arm_raise = abs(math.sin(t * 1.5)) * 0.4  # racket arm goes up sometimes
    knee_bend = abs(math.sin(t * 1.2)) * 0.1

    # Heights
    ankle_y = 0.05
    knee_y = ankle_y + shin - knee_bend
    hip_y = knee_y + thigh
    shoulder_y = hip_y + 0.50
    head_y = shoulder_y + 0.25

    # Head / face landmarks (0-10)
    nose      = [bx, head_y, bz]
    l_eye_in  = [bx - 0.03, head_y + 0.02, bz - 0.02]
    l_eye     = [bx - 0.05, head_y + 0.02, bz - 0.02]
    l_eye_out = [bx - 0.07, head_y + 0.02, bz - 0.02]
    r_eye_in  = [bx + 0.03, head_y + 0.02, bz - 0.02]
    r_eye     = [bx + 0.05, head_y + 0.02, bz - 0.02]
    r_eye_out = [bx + 0.07, head_y + 0.02, bz - 0.02]
    l_ear     = [bx - 0.10, head_y, bz]
    r_ear     = [bx + 0.10, head_y, bz]
    l_mouth   = [bx - 0.03, head_y - 0.05, bz - 0.01]
    r_mouth   = [bx + 0.03, head_y - 0.05, bz - 0.01]

    # Shoulders (11-12)
    l_shoulder = [bx - shoulder_w, shoulder_y, bz]
    r_shoulder = [bx + shoulder_w, shoulder_y, bz]

    # Left arm — swings with movement
    l_elbow = [bx - shoulder_w - 0.05, shoulder_y - upper_arm + arm_swing * 0.15, bz + arm_swing * 0.1]
    l_wrist = [bx - shoulder_w - 0.08, shoulder_y - upper_arm - forearm + arm_swing * 0.2, bz + arm_swing * 0.15]

    # Right arm — racket arm, raises more
    r_elbow = [bx + shoulder_w + 0.05, shoulder_y - upper_arm * 0.5 + arm_raise * 0.3, bz - 0.1 + arm_raise * 0.2]
    r_wrist = [bx + shoulder_w + 0.10, shoulder_y - upper_arm * 0.2 + arm_raise * 0.6, bz - 0.15 + arm_raise * 0.3]

    # Hands (17-22) — close to wrists
    l_pinky  = [l_wrist[0] - 0.02, l_wrist[1] - 0.03, l_wrist[2]]
    l_index  = [l_wrist[0] + 0.02, l_wrist[1] - 0.03, l_wrist[2]]
    l_thumb  = [l_wrist[0] + 0.03, l_wrist[1] - 0.01, l_wrist[2] - 0.02]
    r_pinky  = [r_wrist[0] + 0.02, r_wrist[1] - 0.03, r_wrist[2]]
    r_index  = [r_wrist[0] - 0.02, r_wrist[1] - 0.03, r_wrist[2]]
    r_thumb  = [r_wrist[0] - 0.03, r_wrist[1] - 0.01, r_wrist[2] - 0.02]

    # Hips (23-24)
    l_hip = [bx - hip_w, hip_y, bz]
    r_hip = [bx + hip_w, hip_y, bz]

    # Legs — alternating step
    step = math.sin(t * 2.0) * 0.15
    l_knee  = [bx - hip_w, knee_y, bz + step]
    r_knee  = [bx + hip_w, knee_y, bz - step]
    l_ankle = [bx - hip_w, ankle_y, bz + step * 1.2]
    r_ankle = [bx + hip_w, ankle_y, bz - step * 1.2]

    # Feet (29-32)
    l_heel  = [l_ankle[0], 0.0, l_ankle[2] - 0.05]
    l_toe   = [l_ankle[0], 0.0, l_ankle[2] + 0.15]
    r_heel  = [r_ankle[0], 0.0, r_ankle[2] - 0.05]
    r_toe   = [r_ankle[0], 0.0, r_ankle[2] + 0.15]

    # Assemble in MediaPipe order (indices 0-32)
    landmarks = [
        nose, l_eye_in, l_eye, l_eye_out,       # 0-3
        r_eye_in, r_eye, r_eye_out, l_ear,       # 4-7
        r_ear, l_mouth, r_mouth,                  # 8-10
        l_shoulder, r_shoulder,                    # 11-12
        l_elbow, r_elbow,                          # 13-14
        l_wrist, r_wrist,                          # 15-16
        l_pinky, r_pinky, l_index, r_index,       # 17-20
        l_thumb, r_thumb,                          # 21-22
        l_hip, r_hip,                              # 23-24
        l_knee, r_knee,                            # 25-26
        l_ankle, r_ankle,                          # 27-28
        l_heel, r_heel, l_toe, r_toe,            # 29-32
    ]

    return [[round(c, 4) for c in lm] for lm in landmarks]


def generate_dummy_ball(frame_idx: int, total_frames: int):
    """Generate a volleyball that rallies back and forth over the net (z=0)."""
    # Volleyball rallies are faster; ~50 frames to cross the court
    rally_period = 50
    t_in_rally = (frame_idx % rally_period) / rally_period
    going_forward = (frame_idx // rally_period) % 2 == 0

    if going_forward:
        z = -8.0 + t_in_rally * 16.0   # near baseline to far baseline
    else:
        z = 8.0 - t_in_rally * 16.0

    x = math.sin(frame_idx * 0.1) * 2.5  # lateral drift (narrower court)
    # Higher arc for volleyball — peaks at ~4m over the net
    z_norm = abs(z) / 8.0  # 0 at net, 1 at baseline
    y = max(0.1, 4.0 * (1.0 - z_norm * z_norm))

    detected = (frame_idx % 4) != 0  # ~75% detection rate
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
    }, detected


def analyze_video_dummy(video_path: str, total_frames: int = 150, fps: float = 30.0):
    """
    Dummy analysis that generates fake frame data focused on court display.
    Pose and ball are disabled; only frame/timestamp/progress events are sent.
    """
    send_progress(0, f"Starting analysis ({total_frames} frames at {fps:.1f} FPS)...")

    for i in range(total_frames):
        timestamp = round(i / fps, 4)

        send({
            "action": "frame",
            "frame": i,
            "timestamp": timestamp,
        })

        percent = int((i + 1) / total_frames * 100)
        prev_percent = int(i / total_frames * 100)
        if percent > prev_percent:
            send_progress(percent, f"Processed frame {i+1}/{total_frames}")

    send({
        "action": "complete",
        "total_frames": total_frames,
        "fps": fps,
        "score": {"team1": 0, "team2": 0},
        "stats": {},
    })


def _assign_teams_from_court(results, corners_norm, img_w, img_h):
    """Filter detected poses to the court quad and split by team.

    Players whose ankles fall on the near half (y_norm >= 0.5) belong to
    team 1; those on the far half (y_norm < 0.5) to team 2.

    Returns (team1, team2) where each is a list of dicts:
      { 'ay': float, 'ax': float, 'pose': [...], 'world': [...] | None }
    Sorted so the most confident detection (deepest into the frame from
    that side's perspective) comes first.
    """
    if not results or not results.pose_landmarks:
        return [], []

    team1, team2 = [], []
    for i, lms in enumerate(results.pose_landmarks):
        n = len(lms)
        if n > 28:
            ax = (lms[27].x + lms[28].x) / 2
            ay = (lms[27].y + lms[28].y) / 2
        elif n > 24:
            ax = (lms[23].x + lms[24].x) / 2
            ay = (lms[23].y + lms[24].y) / 2
        else:
            continue

        if not _is_in_court_quad(ax, ay, corners_norm, img_w, img_h):
            continue

        wlms = None
        if results.pose_world_landmarks and i < len(results.pose_world_landmarks):
            wlms = results.pose_world_landmarks[i]

        pose = [[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)] for lm in lms]
        world = [[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)] for lm in wlms] if wlms else None
        entry = {'ay': ay, 'ax': ax, 'pose': pose, 'world': world}
        if ay >= 0.5:  # near side → team 1
            team1.append(entry)
        else:          # far side  → team 2
            team2.append(entry)

    # Sort so the "deepest" player on each side comes first
    team1.sort(key=lambda c: c['ay'], reverse=True)
    team2.sort(key=lambda c: c['ay'])
    return team1, team2



def analyze_video_cv(video_path: str):
    """
    Real CV analysis using OpenCV — court detection only.
    Falls back to dummy if video can't be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        send_error(f"Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    send_progress(0, f"Analyzing {total_frames} frames at {fps:.1f} FPS...")

    court_interval = max(int(fps), 1)
    global _prev_corners, _last_court_keypoints
    _prev_corners = None
    _last_court_keypoints = None

    # Load YOLO model once (optional — ball detection gracefully disabled if unavailable)
    yolo_model = _get_yolo_model()
    ball_raw = {}   # frame_idx -> [cx_norm, cy_norm, conf, x1_norm, y1_norm, x2_norm, y2_norm]
    
    if yolo_model is None:
        print("[yolo] No model available — skipping ball detection", file=sys.stderr)
    else:
        print("[yolo] Model loaded, starting frame processing", file=sys.stderr)

    # ── Pass 1: court + ball detection ────────────────────────────────────────────
    all_frame_data = []   # per-frame lightweight dicts (no raw pixels kept)
    court_data = []
    court_detected_once = False  # stop re-running court detection after first success
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / fps, 4)
        fh, fw = frame.shape[:2]

        frame_court_corners = None
        frame_court_keypoints = None
        if not court_detected_once:
            try:
                corners = _detect_court_hough_fast(frame)
                if corners is not None:
                    frame_court_corners = corners
                    court_detected_once = True
                    kps_reproj = _reproject_keypoints_from_corners(corners, fh, fw)
                    if kps_reproj is not None:
                        frame_court_keypoints = kps_reproj
                        _last_court_keypoints = kps_reproj
                    court_data.append({
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "corners": corners,
                        "keypoints": kps_reproj,
                    })
                    print(f"[court] Detected at frame {frame_idx}, stopping re-detection", file=sys.stderr)
            except Exception as e:
                print(f"[court] Frame {frame_idx} error: {e}", file=sys.stderr)

        # ── Ball detection (every frame) ─────────────────────────────────
        if yolo_model is not None:
            try:
                det = _detect_ball_yolo(frame, yolo_model)
                if det is not None:
                    ball_raw[frame_idx] = det
                    if frame_idx < 3 or frame_idx % 50 == 0:
                        print(f"[yolo] Frame {frame_idx}: detected ball at {det[0]:.3f}, {det[1]:.3f} conf={det[2]:.3f}", file=sys.stderr)
            except Exception as e:
                print(f"[yolo] Frame {frame_idx} error: {e}", file=sys.stderr)
                if frame_idx < 5:
                    import traceback
                    traceback.print_exc(file=sys.stderr)

        all_frame_data.append({
            'frame': frame_idx,
            'timestamp': timestamp,
            'corners': frame_court_corners,
            'keypoints': frame_court_keypoints,
        })

        percent = int((frame_idx + 1) / max(total_frames, 1) * 100)
        if percent % 5 == 0 and frame_idx > 0:
            send_progress(
                min(percent // 2, 48),
                f"Detecting: frame {frame_idx + 1}/{total_frames}"
            )
        frame_idx += 1

    cap.release()

    # ── Interpolate ball trajectory for frames with no detection ──────────
    ball_path = _interpolate_ball_path(ball_raw)
    n_ball_detected = len(ball_raw)
    n_ball_total = len(ball_path)
    print(f"[yolo] Ball: {n_ball_detected} raw detections, "
          f"{n_ball_total - n_ball_detected} interpolated, "
          f"{n_ball_total} total", file=sys.stderr)

    # ── Pass 2: Depth Anything V2 on YOLO-detected (non-interpolated) frames ──
    ball_depth = {}   # frame_idx -> relative_depth [0,1]
    if HAS_TRANSFORMERS and HAS_TORCH and n_ball_detected > 0:
        send_progress(50, "Estimating depth (Depth Anything V2)...")
        try:
            depth_pipe = _get_depth_model()
            if depth_pipe is not None:
                from PIL import Image as PILImage
                cap2 = cv2.VideoCapture(video_path)
                frames_to_process = sorted(ball_raw.keys())
                total_depth = len(frames_to_process)
                prev_fi = -1
                for i, fi in enumerate(frames_to_process):
                    # Seek only when needed (sequential reads are faster)
                    if fi != prev_fi + 1:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret2, frame2 = cap2.read()
                    prev_fi = fi
                    if not ret2:
                        continue
                    cx, cy = ball_raw[fi][0], ball_raw[fi][1]
                    d = _estimate_depth_at_ball(frame2, cx, cy)
                    if d is not None:
                        ball_depth[fi] = d
                    if i % 30 == 0:
                        pct = 50 + int(i / max(total_depth, 1) * 15)
                        send_progress(pct, f"Depth estimation: {i + 1}/{total_depth} frames")
                cap2.release()
                print(f"[depth] Estimated depth for {len(ball_depth)}/{total_depth} frames", file=sys.stderr)
        except Exception as e:
            print(f"[depth] Depth estimation pass failed: {e}", file=sys.stderr)

    # ── Emit all frame events ──────────────────────────────────────────────
    send_progress(70, "Sending results...")
    n_frames = len(all_frame_data)
    for fd in all_frame_data:
        fi = fd['frame']
        frame_msg = {
            "action": "frame",
            "frame": fi,
            "timestamp": fd['timestamp'],
        }
        if fd['corners'] is not None:
            frame_msg["court_corners"] = fd['corners']
        if fd['keypoints'] is not None:
            frame_msg["court_keypoints"] = fd['keypoints']

        # Ball position (detected or interpolated)
        bd = ball_path.get(fi)
        if bd is not None:
            frame_msg["ball_img_pos"] = bd['pos']
            frame_msg["ball_interpolated"] = bd['interpolated']
            frame_msg["ball_conf"] = bd['conf']
            if bd['bbox'] is not None:
                frame_msg["ball_bbox"] = bd['bbox']
            # Depth estimate available only for YOLO-detected (non-interpolated) frames
            if fi in ball_depth:
                frame_msg["ball_depth_estimate"] = ball_depth[fi]

        send(frame_msg)

        if fi % 200 == 0 and fi > 0:
            pct = 70 + int(fi / max(n_frames, 1) * 28)
            send_progress(pct, f"Sending frame {fi + 1}/{n_frames}")

    send({
        "action": "complete",
        "total_frames": frame_idx,
        "fps": fps,
        "court_data": court_data,
        "score": {"team1": 0, "team2": 0},
        "stats": {},
    })


def analyze_video(video_path: str, duration_hint: float = 0.0):
    """Route to real CV or dummy analysis based on available libraries."""
    if not os.path.isfile(video_path):
        # Non-existent path = demo mode (generates synthetic data)
        send_progress(0, "Demo mode: generating sample data...")
        analyze_video_dummy(video_path)
        return

    if HAS_CV2:
        analyze_video_cv(video_path)
    else:
        # Try to get real video metadata via ffprobe
        total_frames, fps = get_video_info(video_path)
        if total_frames and fps:
            send_progress(0, f"Video detected: {total_frames} frames at {fps:.1f} FPS (no CV libs, using dummy poses)")
            analyze_video_dummy(video_path, total_frames, fps)
        elif duration_hint > 0:
            # Use duration from the frontend
            fps = 30.0
            total_frames = int(duration_hint * fps)
            send_progress(0, f"Using video duration hint: {total_frames} frames")
            analyze_video_dummy(video_path, total_frames, fps)
        else:
            send_progress(0, "Demo mode: cannot detect video metadata, generating sample data")
            analyze_video_dummy(video_path)


def main():
    """Main loop: read JSON requests from stdin, dispatch actions."""
    # Startup diagnostics — helps debug venv/package issues when running from Tauri
    print(f"[analyzer] Python: {sys.executable}", file=sys.stderr)
    print(f"[analyzer] CWD: {os.getcwd()}", file=sys.stderr)
    try:
        import ultralytics
        print(f"[analyzer] ultralytics {ultralytics.__version__} OK", file=sys.stderr)
    except ImportError:
        print("[analyzer] ultralytics NOT available — ball detection disabled", file=sys.stderr)

    send({"action": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            send_error(f"Invalid JSON: {e}")
            continue

        action = msg.get("action")

        if action == "analyze":
            video_path = msg.get("video_path", "")
            if not video_path:
                send_error("Missing video_path")
                continue
            duration_hint = msg.get("duration", 0.0)
            analyze_video(video_path, duration_hint)

        elif action == "detect_court":
            video_path = msg.get("video_path", "")
            if not video_path:
                send_error("Missing video_path")
                continue
            if not HAS_CV2 or not HAS_NUMPY:
                send_error("OpenCV and NumPy required for court detection")
                continue
            frame_time = msg.get("frame_time", 0.0)
            handle_detect_court(video_path, frame_time)

        elif action == "ping":
            send({"action": "pong"})

        elif action == "quit":
            break

        else:
            send_error(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
