"""Pinned MediaPipe FaceLandmarker asset used by the webcam engine.

Updating the model requires changing FACE_LANDMARKER_VERSION, the URL, and
FACE_LANDMARKER_SHA256 together. A checksum mismatch is a hard failure.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

# Tasks Face Landmarker, float16, numbered revision (not "latest").
FACE_LANDMARKER_VERSION = "1"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    f"face_landmarker/face_landmarker/float16/{FACE_LANDMARKER_VERSION}/"
    "face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"

_CHUNK = 1024 * 1024


class ModelChecksumError(RuntimeError):
    """Raised when a FaceLandmarker file does not match the pinned SHA-256."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_face_landmarker(path: Path, expected: str = FACE_LANDMARKER_SHA256) -> None:
    if not path.is_file():
        msg = f"Face landmarker model not found: {path}"
        raise FileNotFoundError(msg)
    actual = sha256_file(path)
    if actual != expected:
        msg = (
            f"Face landmarker checksum mismatch for {path}. "
            f"expected={expected} actual={actual}. "
            "Delete the file and re-run scripts/download_mediapipe_model.py, "
            "or update FACE_LANDMARKER_VERSION and FACE_LANDMARKER_SHA256 together."
        )
        raise ModelChecksumError(msg)


def download_face_landmarker(
    dest: Path | None = None,
    *,
    url: str = FACE_LANDMARKER_URL,
    expected: str = FACE_LANDMARKER_SHA256,
) -> Path:
    """Download the pinned model if needed and verify SHA-256 before returning."""
    path = dest or DEFAULT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        verify_face_landmarker(path, expected)
        return path

    tmp = path.with_suffix(path.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        verify_face_landmarker(tmp, expected)
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            tmp.unlink()
        raise
    return path
