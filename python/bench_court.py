"""Quick benchmark for court detection speed."""
import time, os, sys
import numpy as np

pth = os.path.join(os.path.dirname(__file__), 'models', 'court_detection_model.pth')
if not os.path.isfile(pth):
    print(f'Model not found at {pth} - skipping test')
    sys.exit(0)

print(f'Model exists: {os.path.getsize(pth)/(1024*1024):.1f} MB')

t0 = time.time()
from court_model import CourtDetector
detector = CourtDetector()
t1 = time.time()
print(f'Init time: {(t1-t0)*1000:.0f}ms (backend={detector._backend})')

onnx_path = os.path.join(os.path.dirname(__file__), 'models', 'court_detection_model.onnx')
if os.path.isfile(onnx_path):
    print(f'ONNX model: {os.path.getsize(onnx_path)/(1024*1024):.1f} MB')

# Benchmark with a dummy frame
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup
detector.detect(frame)

# Measure
times = []
for i in range(5):
    t0 = time.time()
    detector.detect(frame)
    times.append((time.time()-t0)*1000)

print(f'Detect times: {[f"{t:.0f}ms" for t in times]}')
print(f'Average: {sum(times)/len(times):.0f}ms')
