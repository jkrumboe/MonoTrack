from ultralytics import YOLO

SPORTS_BALL = 32  # COCO class index for sports ball

model = YOLO('yolov8m.pt')

results = model.predict(
    source='C:/repos/tennis-3d/notebooks/data/data/videos/clip1.mp4',
    classes=[SPORTS_BALL],
    show=True,
    save=True,
    stream=True,
)

for i, r in enumerate(results):
    if r.boxes and len(r.boxes):
        print(f"Frame {i:06d}: ball at {r.boxes.xyxy.tolist()}")