"""Detailed profiling of court detection stages."""
import time, os
import numpy as np
import cv2

pth = os.path.join(os.path.dirname(__file__), 'models', 'court_detection_model.pth')
if not os.path.isfile(pth):
    print('Model not found'); exit()

from court_model import CourtDetector, _postprocess_heatmap, _homography_refine, INPUT_W, INPUT_H

detector = CourtDetector()
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup
detector.detect(frame)

# Profile breakdown
img = cv2.resize(frame, (INPUT_W, INPUT_H))
inp = img.astype(np.float32) / 255.0

# 1. Model inference only
times_infer = []
for _ in range(5):
    t0 = time.time()
    pred = detector._infer(inp)
    times_infer.append((time.time()-t0)*1000)
print(f"Model inference: avg {sum(times_infer)/len(times_infer):.0f}ms  {[f'{t:.0f}' for t in times_infer]}")

# 2. Postprocess (14x HoughCircles) 
pred = detector._infer(inp)
times_post = []
for _ in range(5):
    t0 = time.time()
    points = []
    for i in range(14):
        heatmap = (pred[i] * 255).astype(np.uint8)
        x, y = _postprocess_heatmap(heatmap)
        points.append((x, y))
    times_post.append((time.time()-t0)*1000)
print(f"Postprocess (14x HoughCircles): avg {sum(times_post)/len(times_post):.0f}ms  {[f'{t:.0f}' for t in times_post]}")

# 3. Homography refinement
times_homo = []
for _ in range(5):
    t0 = time.time()
    _homography_refine(points)
    times_homo.append((time.time()-t0)*1000)
print(f"Homography refine: avg {sum(times_homo)/len(times_homo):.1f}ms  {[f'{t:.1f}' for t in times_homo]}")
