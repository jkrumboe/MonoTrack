"""
TrackNetV3 ball detection for volleyball.

Architecture and pre-processing follow qaz812345/TrackNetV3 (MIT licensed):
  https://github.com/qaz812345/TrackNetV3
(Chen & Wang, "TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations
and Trajectory Rectification", 2023). Upstream publishes badminton weights
only; the volleyball weights used here are deadfast/beach-volley-vision-models
on Hugging Face (MIT), fine-tuned from them on beach volleyball rallies.

What V3 adds over V2: it sees 8 frames at once *plus the clip's median
frame*, so the static background (court, net, crowd) is given to it rather
than inferred, and the ball stands out as what differs from it.

Upstream's second stage, InpaintNet, fills gaps in the 2D track. It is left
out: it was not retrained for volleyball, and the analyzer's 3D solver
already bridges gaps with real ballistics.

Input : median + 8 frames x RGB = 27 channels at 512x288, stretched, /255
Output: 8 sigmoid heatmaps at 512x288
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

from ball_tracknet import SCORE_THRESHOLD, candidates_from_heatmap, download_checked, warp_normalised

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional at import time
    HAS_TORCH = False
    nn = None

INPUT_W = 512
INPUT_H = 288
FRAMES_IN = 8
MEDIAN_SAMPLES = 300   # frames spread over the clip for the background

_WEIGHTS_URL = ("https://huggingface.co/deadfast/beach-volley-vision-models/resolve/"
                "b03107eddc5bc32d2f89f70b225e3b67da9c34e7/tracknet_best.pt")
_WEIGHTS_SHA256 = "95f52d4c4e9c30c987f82a97cf1a869549db9916a8eb82d335b57e99d11739eb"


def weights_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                        "tracknetv3_beach_volleyball.pt")


def ensure_weights():
    return download_checked(_WEIGHTS_URL, weights_path(), _WEIGHTS_SHA256)


def _resize_rgb(frame_bgr):
    """BGR frame -> RGB at the model size. PIL's resize, as in training: its
    bicubic filter antialiases a 1080p downscale, cv2's INTER_CUBIC does not."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.asarray(img.resize((INPUT_W, INPUT_H)))


# ─── Architecture (must match the published weights exactly) ────────────────
# Mirrors TrackNetV3 model.py. Unlike V2 these are Conv -> BN -> ReLU, without
# conv bias, and the model ends in its own sigmoid.

if HAS_TORCH:

    class _Conv(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(out_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class _Convs(nn.Module):
        def __init__(self, n, in_dim, out_dim):
            super().__init__()
            self.conv_1 = _Conv(in_dim, out_dim)
            self.conv_2 = _Conv(out_dim, out_dim)
            if n == 3:
                self.conv_3 = _Conv(out_dim, out_dim)

        def forward(self, x):
            x = self.conv_2(self.conv_1(x))
            return self.conv_3(x) if hasattr(self, "conv_3") else x

    class TrackNetV3(nn.Module):
        def __init__(self, in_dim=3 * (FRAMES_IN + 1), out_dim=FRAMES_IN):
            super().__init__()
            self.down_block_1 = _Convs(2, in_dim, 64)
            self.down_block_2 = _Convs(2, 64, 128)
            self.down_block_3 = _Convs(3, 128, 256)
            self.bottleneck = _Convs(3, 256, 512)
            self.up_block_1 = _Convs(3, 768, 256)
            self.up_block_2 = _Convs(2, 384, 128)
            self.up_block_3 = _Convs(2, 192, 64)
            self.predictor = nn.Conv2d(64, out_dim, (1, 1))
            self.pool = nn.MaxPool2d(2, 2)
            self.up = nn.Upsample(scale_factor=2)

        def forward(self, x):
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(self.pool(x1))
            x3 = self.down_block_3(self.pool(x2))
            x = self.bottleneck(self.pool(x3))
            x = self.up_block_1(torch.cat([self.up(x), x3], dim=1))
            x = self.up_block_2(torch.cat([self.up(x), x2], dim=1))
            x = self.up_block_3(torch.cat([self.up(x), x1], dim=1))
            return torch.sigmoid(self.predictor(x))


class TrackNetV3BallDetector:
    """Runs TrackNetV3 over a video in 8-frame windows.

    begin_clip() must be called with the video first: the median background
    is computed from the whole clip before any window is detected.
    """

    name = "tracknetv3"
    window = FRAMES_IN

    def __init__(self, model_path=None, device=None, threshold=SCORE_THRESHOLD):
        if not HAS_TORCH:
            raise RuntimeError("torch is required for TrackNetV3 ball detection")
        path = model_path or ensure_weights()
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError("TrackNetV3 volleyball weights not available")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # weights_only: the checkpoint is a third-party download; tensors and
        # plain dicts are all it needs, so nothing in it gets to run code.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        params = ckpt.get("param_dict", {})
        if params.get("seq_len", FRAMES_IN) != FRAMES_IN or params.get("bg_mode", "concat") != "concat":
            raise ValueError(f"unsupported TrackNetV3 checkpoint: {params}")
        self.model = TrackNetV3()
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device).eval()
        self.half = self.device.type == "cuda"
        self.threshold = threshold
        self.median = None
        self.last_peak = 0.0
        self.max_peak = 0.0
        print(f"[tracknetv3] Loaded {os.path.basename(path)} on {self.device}", file=sys.stderr)

    def begin_clip(self, video_path, motion=None):
        """Median background of the clip, from up to MEDIAN_SAMPLES spread frames.

        With motion (see WindowStream) the frames are warped into the reference
        image first, or a panning camera smears the background it is meant to
        show; set_view() then turns the median to each window's view.
        """
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, n // MEDIAN_SAMPLES)
        imgs = []
        for i in range(0, n, step):
            # Seeking is slow on long-GOP video; read through short steps instead.
            if step > 4 or i == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            else:
                for _ in range(step - 1):
                    cap.grab()
            ok, f = cap.read()
            if not ok:
                break
            img = _resize_rgb(f)
            H = motion(i) if motion is not None else None
            imgs.append(img if H is None else warp_normalised(img, H))
        cap.release()
        if not imgs:
            raise ValueError(f"cannot read frames from {video_path}")
        self.median = self.ref_median = np.median(np.stack(imgs), axis=0).astype(np.uint8)

    def set_view(self, H_ref):
        """Show the median as seen from a frame whose homography to the reference is H_ref."""
        self.median = warp_normalised(self.ref_median, np.linalg.inv(H_ref))

    def _preprocess(self, frames_bgr):
        if self.median is None:
            raise RuntimeError("begin_clip() was not called")
        chans = [self.median] + [_resize_rgb(f) for f in frames_bgr]
        x = np.concatenate([c.transpose(2, 0, 1) for c in chans]).astype(np.float32) / 255.0
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def window_heatmaps(self, frames_bgr):
        """Sigmoid heatmaps (8, 288, 512) for a window of 8 consecutive frames."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            hms = self.model(self._preprocess(frames_bgr)).float()[0].cpu().numpy()
        self.last_peak = float(hms.max())
        self.max_peak = max(self.max_peak, self.last_peak)
        return hms

    def candidates(self, hm, w, h):
        return candidates_from_heatmap(hm, self.threshold, lambda x, y: (x / INPUT_W, y / INPUT_H))

    def detect_window(self, frames_bgr):
        """Per-frame candidate lists for a window of 8 consecutive frames."""
        if len(frames_bgr) != FRAMES_IN:
            raise ValueError(f"expected {FRAMES_IN} frames, got {len(frames_bgr)}")
        h, w = frames_bgr[0].shape[:2]
        return [self.candidates(hm, w, h) for hm in self.window_heatmaps(frames_bgr)]
