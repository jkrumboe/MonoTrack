import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, prefix: str = "frame") -> int:
	output_dir.mkdir(parents=True, exist_ok=True)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise ValueError(f"Could not open video file: {video_path}")

	count = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break

		frame_file = output_dir / f"{prefix}_{count:06d}.jpg"
		cv2.imwrite(str(frame_file), frame)
		count += 1

	cap.release()
	return count


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract all frames from an MP4 video.")
	parser.add_argument("input_video", type=Path, help="Path to input .mp4 file")
	parser.add_argument("output_dir", type=Path, help="Directory where frames will be saved")
	parser.add_argument("--prefix", default="frame", help="Frame filename prefix")
	args = parser.parse_args()

	if not args.input_video.exists():
		raise FileNotFoundError(f"Input video not found: {args.input_video}")

	total = extract_frames(args.input_video, args.output_dir, args.prefix)
	print(f"Extracted {total} frames to {args.output_dir}")


if __name__ == "__main__":
	main()
