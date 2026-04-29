#!/usr/bin/env python3
"""
Download the pretrained court detection model weights.

Source: TennisCourtDetector by yastrebksv (tennis-specific, not used for volleyball)
  https://github.com/yastrebksv/TennisCourtDetector

Run this script from the python/ directory:
  python download_model.py
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "court_detection_model.pth")
GDRIVE_FILE_ID = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"


def download():
    if os.path.isfile(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model already exists at {MODEL_PATH} ({size_mb:.1f} MB)")
        resp = input("Re-download? [y/N] ").strip().lower()
        if resp != "y":
            return

    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading model weights to {MODEL_PATH}...")
    gdown.download(url, MODEL_PATH, quiet=False)

    if os.path.isfile(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Done! Model saved ({size_mb:.1f} MB)")
    else:
        print("ERROR: Download failed. Try manually downloading from:")
        print(f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        print(f"  and place the file at: {MODEL_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    download()
