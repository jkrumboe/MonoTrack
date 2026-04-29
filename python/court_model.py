"""
Tennis court keypoint detection using a CNN (TrackNet-style architecture).

Based on TennisCourtDetector by yastrebksv:
  https://github.com/yastrebksv/TennisCourtDetector

The model predicts 14 court keypoints via heatmap regression on a 640x360 image,
then optionally refines them using homography across 12 court configurations.

Keypoint layout (in court reference diagram coordinates):
  0-1:   far baseline left/right
  2-3:   near baseline left/right
  4-5:   left singles sideline top/bottom
  6-7:   right singles sideline top/bottom
  8-9:   far service line left/right
  10-11: near service line left/right
  12-13: center service line top/bottom
"""

import os
import sys
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ─── Model Architecture ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=15):
        super().__init__()
        self.out_channels = out_channels
        # Encoder
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


# ─── Court Reference Keypoints ──────────────────────────────────────────────
# 14 keypoints in the court reference diagram coordinate system (pixels in a
# synthetic diagram). These define the line intersections of a standard court.

COURT_REF_KPS = [
    (286, 561), (1379, 561),       # 0-1: far baseline left, right
    (286, 2935), (1379, 2935),     # 2-3: near baseline left, right
    (423, 561), (423, 2935),       # 4-5: left singles sideline top, bottom
    (1242, 561), (1242, 2935),     # 6-7: right singles sideline top, bottom
    (423, 1110), (1242, 1110),     # 8-9: far service line left, right
    (423, 2386), (1242, 2386),     # 10-11: near service line left, right
    (832, 1110), (832, 2386),      # 12-13: center service line top, bottom
]

# 12 configurations of 4 keypoints for homography estimation.
COURT_CONF_IND = {
    1:  [0, 1, 2, 3],
    2:  [4, 6, 5, 7],
    3:  [4, 1, 5, 3],
    4:  [0, 6, 2, 7],
    5:  [8, 9, 10, 11],
    6:  [8, 9, 5, 7],
    7:  [4, 6, 10, 11],
    8:  [6, 1, 7, 3],
    9:  [0, 4, 2, 5],
    10: [8, 12, 10, 13],
    11: [12, 9, 13, 11],
    12: [10, 11, 5, 7],
}


# ─── Postprocessing ─────────────────────────────────────────────────────────

def _postprocess_heatmap(heatmap, low_thresh=170, min_radius=10, max_radius=25):
    """Extract a keypoint position from a single heatmap channel using HoughCircles."""
    _, binary = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is not None:
        return float(circles[0][0][0]), float(circles[0][0][1])
    return None, None


def _homography_refine(points):
    """
    Refine detected keypoints using homography.

    Tries 12 court configurations, finds the homography that best fits the
    detected points, then reprojects all reference keypoints to get
    geometrically consistent results.
    """
    refer_kps = np.array(COURT_REF_KPS, dtype=np.float32).reshape((-1, 1, 2))

    best_matrix = None
    best_dist = np.inf

    for conf_id, inds in COURT_CONF_IND.items():
        conf_pts = [COURT_REF_KPS[i] for i in inds]
        det_pts = [points[i] for i in inds]

        if any(p[0] is None or p[1] is None for p in det_pts):
            continue

        src = np.array(conf_pts, dtype=np.float32)
        dst = np.array(det_pts, dtype=np.float32)

        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            continue

        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)

        dists = []
        for i in range(14):
            if i not in inds and points[i][0] is not None:
                dists.append(np.linalg.norm(
                    np.array(points[i]) - trans_kps[i][0]
                ))

        if dists:
            dist_mean = np.mean(dists)
            if dist_mean < best_dist:
                best_dist = dist_mean
                best_matrix = matrix

    if best_matrix is not None:
        refined = cv2.perspectiveTransform(refer_kps, best_matrix)
        return [(float(p[0][0]), float(p[0][1])) for p in refined]

    return points


# ─── Detector Class ─────────────────────────────────────────────────────────

INPUT_W, INPUT_H = 640, 360


def _default_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "court_detection_model.pth")


def _default_onnx_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "court_detection_model.onnx")


def _export_onnx(pth_path, onnx_path):
    """Convert PyTorch weights to ONNX for faster inference."""
    if not HAS_TORCH:
        return False
    try:
        model = BallTrackerNet(out_channels=15)
        model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=True))
        model.eval()
        dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        # Use dynamo=False to force the legacy (TorchScript-based) exporter
        # which produces correct full-size ONNX files
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=13,
            dynamo=False,
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[court_model] Exported ONNX model to {onnx_path} ({size_mb:.1f} MB)", file=sys.stderr)
        if size_mb < 1.0:
            print(f"[court_model] WARNING: ONNX model too small ({size_mb:.1f} MB), removing", file=sys.stderr)
            os.remove(onnx_path)
            return False
        return True
    except Exception as e:
        print(f"[court_model] ONNX export failed: {e}", file=sys.stderr)
        if os.path.isfile(onnx_path):
            os.remove(onnx_path)
        return False


class CourtDetector:
    """CNN-based court keypoint detector (tennis-only model). Load once, call detect() per frame.
    
    Prefers ONNX Runtime (3-5x faster on CPU). Falls back to PyTorch.
    Auto-exports ONNX from PyTorch weights on first run.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = _default_model_path()

        onnx_path = _default_onnx_path()
        self._backend = None  # 'ort' or 'torch'
        self._ort_session = None
        self._torch_model = None

        # Try ONNX Runtime first (much faster on CPU)
        if HAS_ORT:
            # Auto-export ONNX if .pth exists but .onnx doesn't
            if not os.path.isfile(onnx_path) and os.path.isfile(model_path):
                _export_onnx(model_path, onnx_path)

            if os.path.isfile(onnx_path):
                try:
                    sess_opts = ort.SessionOptions()
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess_opts.intra_op_num_threads = os.cpu_count() or 4
                    self._ort_session = ort.InferenceSession(
                        onnx_path, sess_opts,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    )
                    self._backend = 'ort'
                    print(f"[court_model] Using ONNX Runtime ({self._ort_session.get_providers()})", file=sys.stderr)
                except Exception as e:
                    print(f"[court_model] ONNX Runtime init failed: {e}", file=sys.stderr)

        # Fall back to PyTorch
        if self._backend is None:
            if not HAS_TORCH:
                raise RuntimeError(
                    "Either onnxruntime or torch is required for CNN court detection. "
                    "Install with: pip install onnxruntime  (or torch)"
                )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Run: python download_model.py   (from the python/ directory)"
                )
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._torch_model = BallTrackerNet(out_channels=15)
            self._torch_model.load_state_dict(torch.load(model_path, map_location=self._device, weights_only=True))
            self._torch_model = self._torch_model.to(self._device)
            self._torch_model.eval()
            self._backend = 'torch'
            print(f"[court_model] Using PyTorch ({self._device})", file=sys.stderr)

    def _infer(self, img_np):
        """Run model inference. Input: HWC float32 [0,1]. Returns: (15, H, W) float32 sigmoid output."""
        inp = np.rollaxis(img_np, 2, 0)[np.newaxis]  # (1, 3, H, W)

        if self._backend == 'ort':
            output = self._ort_session.run(None, {'input': inp.astype(np.float32)})[0][0]
            # Apply sigmoid (ONNX export doesn't include it)
            output = 1.0 / (1.0 + np.exp(-output))
            return output
        else:
            inp_t = torch.from_numpy(inp).float().to(self._device)
            with torch.no_grad():
                out = self._torch_model(inp_t)[0]
                return torch.sigmoid(out).cpu().numpy()

    def detect(self, frame, use_homography=True):
        """
        Detect court keypoints in a BGR frame.

        Returns:
            keypoints: list of 14 (x, y) tuples in original frame coords
                       (None, None) for undetected keypoints
            corners_normalized: list of 4 [x, y] in [0,1] coords
                       order: far-left, far-right, near-right, near-left
                       or None if detection failed
        """
        img_h, img_w = frame.shape[:2]

        # Preprocess: resize to 640x360, normalize to [0,1]
        img = cv2.resize(frame, (INPUT_W, INPUT_H))
        inp = img.astype(np.float32) / 255.0

        # Forward pass
        pred = self._infer(inp)

        # Extract keypoints from the 14 heatmap channels
        points = []
        for i in range(14):
            heatmap = (pred[i] * 255).astype(np.uint8)
            x, y = _postprocess_heatmap(heatmap)
            if x is not None:
                # Scale from 640x360 heatmap coords to original frame coords
                x = x * img_w / INPUT_W
                y = y * img_h / INPUT_H
            points.append((x, y))

        n_detected = sum(1 for p in points if p[0] is not None)
        if n_detected < 4:
            return points, None

        # Homography refinement: use all 12 court configs to find best fit
        if use_homography:
            points = _homography_refine(points)

        # Extract the 4 doubles court corners
        # Order matches CourtCalibration.tsx COURT_CORNERS_3D:
        #   far-left, far-right, near-right, near-left
        corner_indices = [0, 1, 3, 2]
        corners = []
        for idx in corner_indices:
            x, y = points[idx]
            if x is None or y is None:
                return points, None
            corners.append([round(x / img_w, 6), round(y / img_h, 6)])

        return points, corners
