"""
TrackNetV2 ball detection for volleyball.

Architecture and pre/post-processing follow WASB-SBDT (MIT licensed):
  https://github.com/nttcom/WASB-SBDT
which publishes TrackNetV2 weights trained per sport, including volleyball.
The original TrackNet idea is from Huang et al. 2019; the PyTorch reference
implementation this mirrors most closely is yastrebksv/TrackNet.

Why this instead of YOLO: a generic COCO "sports ball" detector looks at one
frame at a time, so a small, motion-blurred volleyball scores barely above
noise (0.09-0.13 on broadcast footage here). TrackNet stacks 3 consecutive
frames into a 9-channel input, which makes the ball's *motion* the signal, and
predicts a heatmap per input frame rather than a box.

Input : 3 frames x RGB = 9 channels at 512x288
Output: 3 heatmaps (one per input frame) at 512x288
"""

import os
import sys

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional at import time
    HAS_TORCH = False
    nn = None


INPUT_W = 512
INPUT_H = 288
FRAMES_IN = 3
SCORE_THRESHOLD = 0.5   # WASB detector.postprocessor.score_threshold
# WASB build_img_transforms: T.Normalize with ImageNet statistics (RGB order).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_affine(w, h):
    """2x3 affine mapping an w x h frame into the 512x288 model input.

    Uniform scale, centred, black padding. For 16:9 footage (what the weights
    were trained on) this is identical to WASB's get_transform(). WASB scales
    by 512/max(w, h), which would crop 4:3 and portrait video top/bottom, so
    here the whole frame is fitted inside instead - never stretched.
    """
    k = min(INPUT_W / float(w), INPUT_H / float(h))
    return np.array([[k, 0.0, INPUT_W / 2.0 - k * w / 2.0],
                     [0.0, k, INPUT_H / 2.0 - k * h / 2.0]], dtype=np.float32)


def heatmap_to_image_norm(x, y, affine, w, h):
    """Map a heatmap pixel back to normalised [0,1] coordinates of the frame."""
    k, tx, ty = (float(v) for v in (affine[0, 0], affine[0, 2], affine[1, 2]))
    # Plain floats: these go straight into json.dumps, which rejects np.float32.
    return (x - tx) / k / w, (y - ty) / k / h


# ─── Architecture (must match the published weights exactly) ────────────────
# Mirrors WASB-SBDT src/models/unet2d.py + unet2d_parts.py. Note bn_first=False:
# these blocks are Conv -> ReLU -> BatchNorm, not the usual Conv -> BN -> ReLU.

if HAS_TORCH:

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch, mid_ch=None):
            super().__init__()
            mid_ch = mid_ch or out_ch
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x):
            return self.double_conv(x)

    class _TripleConv(nn.Module):
        def __init__(self, in_ch, out_ch, mid_ch=None):
            super().__init__()
            mid_ch = mid_ch or out_ch
            self.triple_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x):
            return self.triple_conv(x)

    class _Down(nn.Module):
        def __init__(self, n, in_ch, out_ch):
            super().__init__()
            conv = _DoubleConv(in_ch, out_ch) if n == 2 else _TripleConv(in_ch, out_ch)
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), conv)

        def forward(self, x):
            return self.maxpool_conv(x)

    class _Up(nn.Module):
        def __init__(self, n, in1_ch, in2_ch, out_ch):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            merged = in1_ch + in2_ch
            self.conv = _DoubleConv(merged, out_ch) if n == 2 else _TripleConv(merged, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            return self.conv(torch.cat([x2, x1], dim=1))

    class _OutConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class TrackNetV2(nn.Module):
        """UNet-style TrackNetV2: 9 input channels -> 3 output heatmaps."""

        def __init__(self, n_channels=3 * FRAMES_IN, n_classes=FRAMES_IN):
            super().__init__()
            self.inc = _DoubleConv(n_channels, 64)
            self.down1 = _Down(2, 64, 128)
            self.down2 = _Down(3, 128, 256)
            self.down3 = _Down(3, 256, 512)
            self.up1 = _Up(3, 512, 256, 256)
            self.up2 = _Up(2, 256, 128, 128)
            self.up3 = _Up(2, 128, 64, 64)
            self.outc = _OutConv(64, n_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            return self.outc(x)


def published_weights_path():
    """WASB-SBDT's volleyball weights (downloaded by ensure_weights)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "models", "tracknetv2_volleyball_best.pth.tar")


def finetuned_weights_path():
    """Weights retrained on your own footage by train_ball.py, if any."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "models", "tracknetv2_volleyball_finetuned.pth.tar")


def default_weights_path():
    """The retrained weights when train_ball.py has saved some, else the published ones."""
    tuned = finetuned_weights_path()
    return tuned if os.path.isfile(tuned) else published_weights_path()


# Google Drive id from WASB-SBDT src/setup_scripts/setup_weights.sh
_GDRIVE_ID = "103jOdYp4k20avid4uyB9USCuwiphI4Kz"


def ensure_weights(path=None):
    """Return a path to the volleyball weights, downloading them if missing."""
    path = path or default_weights_path()
    if os.path.isfile(path):
        return path
    try:
        import gdown
    except ImportError:
        print("[tracknet] gdown not installed — cannot fetch weights", file=sys.stderr)
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[tracknet] Downloading volleyball weights to {path}", file=sys.stderr)
    try:
        gdown.download(id=_GDRIVE_ID, output=path, quiet=True)
    except Exception as e:
        print(f"[tracknet] Weight download failed: {e}", file=sys.stderr)
        return None
    return path if os.path.isfile(path) else None


# Pixels of anti-aliased rim lost on the two sides of the ball by the colour
# mask. Checked on broadcast 1080p footage against two known positions (a
# toss above a server at the near baseline, a serve from behind the far
# baseline); without it, small far balls read up to ~10 m too deep.
BALL_RIM_PX = 2.5


def measure_ball_diameter(frame_bgr, cx_norm, cy_norm, half=40):
    """Apparent ball diameter in pixels at a detection, or None.

    TrackNet's blob is the heatmap response, not the ball, so its size says
    nothing about depth. The ball itself (a yellow/blue Mikasa) segments well
    by colour in a small crop:
      1. strict yellow, closed over the panel seams; every piece near the
         detection is kept, because a seam can split the ball in two;
      2. hysteresis into a looser yellow range, only a few px out from those
         pieces, to recover the shaded rim without swallowing a yellowish
         floor or shirt;
      3. the narrow side of the hull's min-area rectangle, since motion blur
         stretches the ball only along its direction of travel.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = int(cx_norm * w), int(cy_norm * h)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    crop = frame_bgr[y0:cy + half, x0:cx + half]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    strict = cv2.morphologyEx(cv2.inRange(hsv, (18, 90, 110), (38, 255, 255)),
                              cv2.MORPH_CLOSE, close)
    loose = cv2.inRange(hsv, (14, 55, 65), (40, 255, 255))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(strict)
    centre = np.array([cx - x0, cy - y0], dtype=np.float64)
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= 15 and np.linalg.norm(cents[i] - centre) < 18]
    if not keep or sum(int(stats[i, cv2.CC_STAT_AREA]) for i in keep) < 40:
        return None
    seed = np.isin(labels, keep).astype(np.uint8) * 255

    reach = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    grown = cv2.morphologyEx(cv2.bitwise_or(seed, cv2.bitwise_and(loose, reach)),
                             cv2.MORPH_CLOSE, close)
    _, labels2 = cv2.connectedComponents(grown)
    ids = np.unique(labels2[seed > 0])
    ids = ids[ids > 0]
    pts = np.column_stack(np.nonzero(np.isin(labels2, ids)))[:, ::-1].astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.convexHull(pts))
    d = float(min(rw, rh))
    return round(d + BALL_RIM_PX, 2) if d >= 6 else None


def _detect_blobs(hm, threshold=SCORE_THRESHOLD):
    """Connected-component blob detection on one sigmoid heatmap.

    Mirrors WASB's TracknetV2Postprocessor._detect_blob_concomp with
    use_hm_weight=True: each blob's position is the heatmap-weighted centroid.
    Returns [(x, y, peak, area)] in heatmap pixel coordinates.
    """
    if hm.max() <= threshold:
        return []
    _, hm_th = cv2.threshold(hm, threshold, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
    out = []
    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        ws = hm[ys, xs]
        total = ws.sum()
        if total <= 0:
            continue
        x = float((xs * ws).sum() / total)
        y = float((ys * ws).sum() / total)
        # Peak probability is a calibrated 0-1 confidence; the weight sum is
        # not, so use the peak downstream and keep area for the bbox.
        out.append((x, y, float(ws.max()), int(xs.shape[0]),
                    int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    out.sort(key=lambda b: -b[2])
    return out


class TrackNetBallDetector:
    """Runs TrackNetV2 over a video in non-overlapping 3-frame windows."""

    def __init__(self, model_path=None, device=None, threshold=SCORE_THRESHOLD):
        if not HAS_TORCH:
            raise RuntimeError("torch is required for TrackNet ball detection")
        path = model_path or ensure_weights()
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError("TrackNetV2 volleyball weights not available")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = TrackNetV2()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # strict=True: a shape or name mismatch means our architecture has
        # drifted from the published weights, which must fail loudly.
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        # Half precision on a GPU: 143 -> 81 ms a window on an RTX 3070 Ti,
        # heatmaps within 1e-4 of full precision.
        self.half = self.device.type == "cuda"
        if self.half:
            torch.backends.cudnn.benchmark = True
        self.threshold = threshold
        self.affine = None
        self.last_heatmaps = None
        self.last_peak = 0.0
        self.max_peak = 0.0
        print(f"[tracknet] Loaded {os.path.basename(path)} on {self.device}", file=sys.stderr)

    def _preprocess(self, frames_bgr):
        """Stack 3 BGR frames into a single 9-channel tensor.

        Must match WASB's test-time pipeline exactly, or the heatmaps collapse:
          - aspect-preserving letterbox into 512x288 (see letterbox_affine),
            not a plain stretch;
          - RGB, ToTensor (/255), then ImageNet mean/std normalisation.
        Feeding raw [0,1] pixels was the main reason the model never fired.
        """
        h, w = frames_bgr[0].shape[:2]
        self.affine = letterbox_affine(w, h)
        chans = []
        for f in frames_bgr:
            img = cv2.warpAffine(f, self.affine, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            chans.append(img.transpose(2, 0, 1))  # HWC -> CHW
        stacked = np.concatenate(chans, axis=0)   # (9, H, W), oldest frame first
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    def heatmaps(self, frames_bgr):
        """Raw sigmoid heatmaps (3, 288, 512) for a 3-frame window, for debugging."""
        with torch.no_grad():
            logits = self.model(self._preprocess(frames_bgr))
        return torch.sigmoid(logits)[0].cpu().numpy()

    @torch.no_grad()
    def detect_window(self, frames_bgr):
        """Detect the ball in a window of 3 consecutive frames.

        Returns a list of per-frame candidate lists, each candidate being
        [cx_norm, cy_norm, conf, x1_norm, y1_norm, x2_norm, y2_norm].
        """
        if len(frames_bgr) != FRAMES_IN:
            raise ValueError(f"expected {FRAMES_IN} frames, got {len(frames_bgr)}")
        h, w = frames_bgr[0].shape[:2]
        inp = self._preprocess(frames_bgr)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            logits = self.model(inp).float()
        hms = torch.sigmoid(logits)[0].cpu().numpy()  # (3, H, W)
        self.last_heatmaps = hms
        # Track the strongest response seen, so a run that detects nothing can
        # be diagnosed as "model never fired" vs "model fired but was gated".
        self.last_peak = float(hms.max())
        self.max_peak = max(getattr(self, "max_peak", 0.0), self.last_peak)

        per_frame = []
        for j in range(hms.shape[0]):
            cands = []
            for (x, y, peak, _area, x1, y1, x2, y2) in _detect_blobs(hms[j], self.threshold):
                # The blob is the ball's blurred response, not its true
                # silhouette, so this bbox is only a rough extent.
                cx, cy = heatmap_to_image_norm(x, y, self.affine, w, h)
                bx1, by1 = heatmap_to_image_norm(x1, y1, self.affine, w, h)
                bx2, by2 = heatmap_to_image_norm(x2 + 1, y2 + 1, self.affine, w, h)
                cands.append([
                    round(cx, 6), round(cy, 6), round(peak, 4),
                    round(bx1, 6), round(by1, 6), round(bx2, 6), round(by2, 6),
                ])
            per_frame.append(cands)
        return per_frame
