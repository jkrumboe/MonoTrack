#!/usr/bin/env python3
"""
Tennis match video analyzer sidecar process.

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
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# ─── Tennis Court Model (ITF Standard) ──────────────────────────────────
COURT_LENGTH = 23.77           # baseline to baseline (m)
COURT_WIDTH = 10.97            # doubles sideline to sideline (m)
SINGLES_WIDTH = 8.23           # singles width (m)
NET_Y = COURT_LENGTH / 2      # net position: 11.885m
SERVICE_DIST = 6.40            # service line distance from net (m)
ALLEY = (COURT_WIDTH - SINGLES_WIDTH) / 2  # 1.37m

# Internal court lines for model scoring: [(start_xy, end_xy), ...]
COURT_MODEL_LINES = [
    # Far service line
    ((ALLEY, NET_Y - SERVICE_DIST), (COURT_WIDTH - ALLEY, NET_Y - SERVICE_DIST)),
    # Near service line
    ((ALLEY, NET_Y + SERVICE_DIST), (COURT_WIDTH - ALLEY, NET_Y + SERVICE_DIST)),
    # Center service line
    ((COURT_WIDTH / 2, NET_Y - SERVICE_DIST), (COURT_WIDTH / 2, NET_Y + SERVICE_DIST)),
    # Left singles sideline
    ((ALLEY, 0), (ALLEY, COURT_LENGTH)),
    # Right singles sideline
    ((COURT_WIDTH - ALLEY, 0), (COURT_WIDTH - ALLEY, COURT_LENGTH)),
]


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
        'blue': cv2.inRange(hsv, (90, 30, 40), (130, 255, 255)),
        'clay': cv2.inRange(hsv, (5, 50, 80), (25, 255, 255)),
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
        for t in np.linspace(0, 1, 30):
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
    # Strongly prefer larger quads — full court is typically 30-50% of frame
    area_ratio = area / (img_h * img_w)
    area_weight = min(1.0, area_ratio / 0.30)
    return hit_rate * (area_weight ** 2)


def _find_best_court(h_lines, v_lines, white_mask, img_h, img_w):
    """Try combinations of 2 horizontal + 2 vertical lines, return best court corners."""
    from itertools import combinations

    def seg_len(l):
        return math.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2)

    h_top = sorted(h_lines, key=seg_len, reverse=True)[:10]
    v_top = sorted(v_lines, key=seg_len, reverse=True)[:10]
    best_score, best_corners = 0, None

    for h1, h2 in combinations(h_top, 2):
        y1 = (h1[1] + h1[3]) / 2
        y2 = (h2[1] + h2[3]) / 2
        if abs(y1 - y2) < img_h * 0.30:
            continue
        top, bot = (h1, h2) if y1 < y2 else (h2, h1)

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

            margin = max(img_w, img_h) * 0.15
            if any(c[0] < -margin or c[0] > img_w + margin or
                   c[1] < -margin or c[1] > img_h + margin for c in corners):
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
    Detect tennis court corners with full debug output.
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
    log.append(f"Best court score: {score:.4f}")

    if corners is None or score < 0.1:
        log.append(f"FAIL: No valid court found (score={score:.4f} < 0.1)")
        steps.append({"label": "7. Selected Court", "info": "No valid court", "image": _img_to_base64(frame)})
        return None, steps, log

    corner_labels = ["TL(far-L)", "TR(far-R)", "BR(near-R)", "BL(near-L)"]
    corners_vis = _draw_corners_on(frame, corners, corner_labels)
    # Draw quadrilateral
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(corners_vis, [pts], True, (0, 255, 255), 2)
    steps.append({
        "label": "7. Raw Line Intersections",
        "info": f"score={score:.4f}",
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


def handle_detect_court(video_path, frame_time=0.0):
    """Handle detect_court action: read a frame and run court detection with debug."""
    if not os.path.isfile(video_path):
        send({"action": "error", "message": f"File not found: {video_path}"})
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        send({"action": "error", "message": f"Cannot open video: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = int(frame_time * fps)
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        send({"action": "error", "message": f"Cannot read frame at time {frame_time:.2f}s"})
        return

    corners, steps, log = detect_court_corners_debug(frame)
    if corners is not None:
        send({"action": "court_detected", "corners": corners, "steps": steps, "log": log})
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
    """Generate a ball that rallies back and forth across the net (z=0)."""
    # Ball takes ~40 frames to cross the court, then comes back
    rally_period = 80
    t_in_rally = (frame_idx % rally_period) / rally_period
    going_forward = (frame_idx // rally_period) % 2 == 0

    if going_forward:
        z = -10.0 + t_in_rally * 20.0   # near baseline to far baseline
    else:
        z = 10.0 - t_in_rally * 20.0

    x = math.sin(frame_idx * 0.08) * 3.0  # lateral drift
    # Parabolic arc peaking at the net (z≈0)
    z_norm = abs(z) / 10.0  # 0 at net, 1 at baseline
    y = max(0.05, 2.5 * (1.0 - z_norm * z_norm))  # peaks at ~2.5m over net

    detected = (frame_idx % 5) != 0  # ~80% detection rate
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
    }, detected


def analyze_video_dummy(video_path: str, total_frames: int = 150, fps: float = 30.0):
    """
    Dummy analysis that generates fake frame data.
    Uses real frame count/FPS if provided, otherwise defaults.
    """
    send_progress(0, f"Starting analysis ({total_frames} frames at {fps:.1f} FPS)...")

    total_shots = 0
    ball_speeds = []

    for i in range(total_frames):
        timestamp = round(i / fps, 4)

        # Players positioned on opposite baselines
        # Player 1: near baseline (z ≈ -9), Player 2: far baseline (z ≈ +9)
        pose_p1 = generate_dummy_pose(i, court_x=0.0, court_z=-9.0)
        pose_p2 = generate_dummy_pose(i, court_x=0.0, court_z=9.0)
        ball, ball_detected = generate_dummy_ball(i, total_frames)

        if ball_detected and i > 0:
            total_shots += 1
            ball_speeds.append(80 + math.sin(i * 0.1) * 40)

        send({
            "action": "frame",
            "frame": i,
            "timestamp": timestamp,
            "pose_player1": pose_p1,
            "pose_player2": pose_p2,
            "ball": ball,
            "ball_detected": ball_detected,
        })

        # Report progress at ~1% intervals
        percent = int((i + 1) / total_frames * 100)
        prev_percent = int(i / total_frames * 100)
        if percent > prev_percent:
            send_progress(percent, f"Processed frame {i+1}/{total_frames}")

    avg_speed = round(sum(ball_speeds) / max(len(ball_speeds), 1), 1)

    send({
        "action": "complete",
        "total_frames": total_frames,
        "fps": fps,
        "score": {"player1": 0, "player2": 0},
        "stats": {
            "avg_ball_speed": avg_speed,
            "total_shots": total_shots,
        },
    })


def analyze_video_cv(video_path: str):
    """
    Real CV analysis using OpenCV + MediaPipe.
    Falls back to dummy if video can't be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        send_error(f"Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    send_progress(0, f"Analyzing {total_frames} frames at {fps:.1f} FPS...")

    # Initialize MediaPipe Pose if available
    pose_detector = None
    if HAS_MEDIAPIPE:
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    frame_idx = 0
    total_shots = 0
    ball_speeds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / fps, 4)

        # Pose estimation
        if pose_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb)
            if results.pose_landmarks:
                pose_p1 = [
                    [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]
                    for lm in results.pose_landmarks.landmark
                ]
            else:
                pose_p1 = generate_dummy_pose(frame_idx, court_x=0.0, court_z=-9.0)
            # Second player would need separate detection/tracking
            pose_p2 = generate_dummy_pose(frame_idx, court_x=0.0, court_z=9.0)
        else:
            pose_p1 = generate_dummy_pose(frame_idx, court_x=0.0, court_z=-9.0)
            pose_p2 = generate_dummy_pose(frame_idx, court_x=0.0, court_z=9.0)

        # Ball tracking (placeholder — TrackNet integration goes here)
        ball, ball_detected = generate_dummy_ball(frame_idx, total_frames)
        if ball_detected:
            total_shots += 1
            ball_speeds.append(80 + math.sin(frame_idx * 0.1) * 40)

        send({
            "action": "frame",
            "frame": frame_idx,
            "timestamp": timestamp,
            "pose_player1": pose_p1,
            "pose_player2": pose_p2,
            "ball": ball,
            "ball_detected": ball_detected,
        })

        percent = int((frame_idx + 1) / max(total_frames, 1) * 100)
        if percent % 10 == 0 and frame_idx > 0:
            send_progress(percent, f"Processed frame {frame_idx+1}/{total_frames}")

        frame_idx += 1

    cap.release()
    if pose_detector:
        pose_detector.close()

    avg_speed = round(sum(ball_speeds) / max(len(ball_speeds), 1), 1)

    send({
        "action": "complete",
        "total_frames": frame_idx,
        "fps": fps,
        "score": {"player1": 0, "player2": 0},
        "stats": {
            "avg_ball_speed": avg_speed,
            "total_shots": total_shots,
        },
    })


def analyze_video(video_path: str, duration_hint: float = 0.0):
    """Route to real CV or dummy analysis based on available libraries."""
    if not os.path.isfile(video_path):
        # For demo mode, allow non-existent paths
        if HAS_CV2:
            send_error(f"File not found: {video_path}")
            return
        else:
            send_progress(0, "Demo mode: CV libraries not installed, generating sample data")
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
