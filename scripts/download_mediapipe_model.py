#!/usr/bin/env python3
"""Download and verify the pinned MediaPipe FaceLandmarker model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fovea.webcam.model import (  # noqa: E402
    FACE_LANDMARKER_SHA256,
    FACE_LANDMARKER_URL,
    FACE_LANDMARKER_VERSION,
    download_face_landmarker,
)


def main() -> int:
    print(f"Pinned FaceLandmarker float16 revision {FACE_LANDMARKER_VERSION}")
    print(f"URL: {FACE_LANDMARKER_URL}")
    print(f"SHA-256: {FACE_LANDMARKER_SHA256}")
    path = download_face_landmarker()
    print(f"Verified {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
