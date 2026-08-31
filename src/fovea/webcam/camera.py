"""OpenCV camera wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np


class CameraError(RuntimeError):
    pass


class Webcam:
    def __init__(self, device_index: int, width: int, height: int, mirror: bool) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._capture: Any = None

    def connect(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.device_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not capture.isOpened():
            capture.release()
            msg = (
                f"Could not open webcam index {self.device_index}. "
                "Check that a camera is connected and permissions are granted."
            )
            raise CameraError(msg)
        self._capture = capture

    def read(self) -> np.ndarray | None:
        import cv2

        if self._capture is None:
            raise CameraError("Webcam is not connected.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if self.mirror:
            return np.asarray(cv2.flip(frame, 1))
        return np.asarray(frame)

    @property
    def is_connected(self) -> bool:
        return self._capture is not None

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
