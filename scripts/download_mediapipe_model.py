#!/usr/bin/env python3
"""Download the MediaPipe FaceLandmarker .task model used by the webcam engine."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "face_landmarker.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and DEST.stat().st_size > 0:
        print(f"Already present: {DEST}")
        return 0
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
