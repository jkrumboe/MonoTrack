"""
VballNet ball detection for volleyball (ONNX, runs on the CPU).

Models and pre/post-processing follow asigatchov/fast-volleyball-tracking-inference
(MIT licensed): https://github.com/asigatchov/fast-volleyball-tracking-inference
VballNet is a TrackNetV4-style network trained on volleyball by its author. It
looks at 9 consecutive *grayscale* frames at once, which makes it light enough
for ~150 windows/s on a desktop CPU.

Two output families are published:
  heatmap (VballNetV4c...): 9 heatmaps at 512x288, then 9 radius planes
  grid    (VballNetGrid...): per frame a 48x27 grid of (conf, dx, dy[, radius])
             over a 768x432 input
Both are already sigmoid probabilities.

Input : 9 grayscale frames, stretched (not letterboxed) to the model size, /255
"""

import os
import sys

import cv2
import numpy as np

from ball_tracknet import SCORE_THRESHOLD, candidates_from_heatmap, download_checked

FRAMES_IN = 9
GRID_COLS, GRID_ROWS = 48, 27

# Pinned to an upstream commit and checked by hash, so a retrained model
# published under the same name can't silently change the results here.
_UPSTREAM = ("https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/"
             "dd8d35eef96a0a85b6424a049dc4b560afbf7550/models/")
MODELS = {
    # Best F1 in upstream's own benchmark (0.90).
    "VballNetV4c_seq9_grayscale_20260908_213829.onnx":
        "8eea11cc5c881fa008ec55b0f5732ab79f2e0e1833d7f586e680f3aaa98e4b8c",
    "VballNetGridV2b_seq9_grayscale_20260909_001145.onnx":
        "59529e8f40891eb689e103b97713c4c2c6eaa674fd625618e73c4354c7a8dc27",
}
DEFAULT_MODEL = "VballNetV4c_seq9_grayscale_20260908_213829.onnx"


def ensure_weights(name=DEFAULT_MODEL):
    """Path to a published VballNet model, downloading it if missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", name)
    if name not in MODELS:
        return path if os.path.isfile(path) else None
    return download_checked(_UPSTREAM + name, path, MODELS[name])


def _grid_heatmap(planes, in_w, in_h):
    """One frame's detection grid drawn as a heatmap at input size.

    Each cell that is a local maximum becomes a small blob peaking at its
    confidence, at its regressed sub-cell position, so grid and heatmap models
    share the blob detection and temporal averaging downstream.
    """
    conf = planes[0]
    hm = np.zeros((in_h, in_w), np.float32)
    peaks = (conf > 0.05) & (conf == cv2.dilate(conf, np.ones((3, 3), np.float32)))
    for row, col in zip(*np.nonzero(peaks)):
        x = int(np.clip((col + float(planes[1, row, col])) * in_w / GRID_COLS, 0, in_w - 1))
        y = int(np.clip((row + float(planes[2, row, col])) * in_h / GRID_ROWS, 0, in_h - 1))
        cv2.circle(hm, (x, y), 2, float(conf[row, col]), -1)
    return hm


class VballNetBallDetector:
    """Runs a VballNet ONNX model over a video in 9-frame windows."""

    name = "vballnet"
    window = FRAMES_IN
    # Overlapping windows, averaged: 45 -> 55 on the eval.
    stride = 3

    def __init__(self, model_path=None, threshold=SCORE_THRESHOLD):
        import onnxruntime as ort
        path = model_path or ensure_weights()
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError("VballNet model not available")
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, seq, self.in_h, self.in_w = inp.shape
        if seq != FRAMES_IN:
            raise ValueError(f"{os.path.basename(path)} takes {seq} frames, expected {FRAMES_IN}")
        out_ch = self.session.get_outputs()[0].shape[1]
        self.family = "grid" if "grid" in os.path.basename(path).lower() else "heatmap"
        # Read the planes per frame from the channel count, not the file name.
        self.planes = out_ch // FRAMES_IN
        self.threshold = threshold
        self.last_peak = 0.0
        self.max_peak = 0.0
        print(f"[vballnet] Loaded {os.path.basename(path)} ({self.family}, "
              f"{self.in_w}x{self.in_h}) on {self.session.get_providers()[0]}", file=sys.stderr)

    def _preprocess(self, frames_bgr):
        x = np.stack([
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (self.in_w, self.in_h),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            for f in frames_bgr])
        return x[None]   # (1, 9, H, W), oldest frame first

    def window_heatmaps(self, frames_bgr):
        """Heatmaps (9, H, W) for a window of 9 consecutive frames."""
        out = self.session.run(None, {self.input_name: self._preprocess(frames_bgr)})[0][0]
        if self.family == "grid":
            grid = out.reshape(FRAMES_IN, self.planes, GRID_ROWS, GRID_COLS)
            hms = np.stack([_grid_heatmap(g, self.in_w, self.in_h) for g in grid])
        else:
            hms = out[:FRAMES_IN]   # any radius planes follow the heatmaps
        self.last_peak = float(hms.max())
        self.max_peak = max(self.max_peak, self.last_peak)
        return hms

    def candidates(self, hm, w, h):
        return candidates_from_heatmap(hm, self.threshold, lambda x, y: (x / self.in_w, y / self.in_h))

    def detect_window(self, frames_bgr):
        """Per-frame candidate lists for a window of 9 consecutive frames."""
        if len(frames_bgr) != FRAMES_IN:
            raise ValueError(f"expected {FRAMES_IN} frames, got {len(frames_bgr)}")
        h, w = frames_bgr[0].shape[:2]
        return [self.candidates(hm, w, h) for hm in self.window_heatmaps(frames_bgr)]
