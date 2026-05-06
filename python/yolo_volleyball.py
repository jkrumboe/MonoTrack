from pathlib import Path

from ultralytics import YOLO

FRAMES_DIR = Path('C:/repos/tennis-3d/notebooks/data/data/images/frames')
SPORTS_BALL = 32  # COCO class index for sports ball

model = YOLO('yolov8m.pt')

frames = sorted(FRAMES_DIR.glob('*.jpg'))
print(f"Running on {len(frames)} frames...")

results = model.predict(
    source=[str(f) for f in frames],
    classes=[SPORTS_BALL],
    save=True,
    stream=True,
)

for i, r in enumerate(results):
    boxes = r.boxes
    if boxes and len(boxes):
        print(f"Frame {i:06d}: ball detected at {boxes.xyxy.tolist()}")