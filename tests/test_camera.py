from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fovea.webcam.camera import (
    CameraActuals,
    CameraError,
    CameraInfo,
    CameraSelector,
    Webcam,
    _avfoundation_cameras,
    _v4l2_cameras,
    resolve_camera,
)

CAMERAS = (
    CameraInfo(0, "Built-in Camera", "builtin", True),
    CameraInfo(2, "Desk Camera", "desk", False),
    CameraInfo(4, "Desk Camera Wide", "desk-wide", False),
)


def test_resolve_camera_index_passes_through() -> None:
    assert resolve_camera(CameraSelector(index=9), CAMERAS) == (9, None)
    assert resolve_camera(CameraSelector(index=2), CAMERAS) == (2, CAMERAS[1])


def test_resolve_camera_name_is_case_insensitive_substring() -> None:
    assert resolve_camera(CameraSelector(name="BUILT-in"), CAMERAS) == (0, CAMERAS[0])


@pytest.mark.parametrize("name", ["missing", "desk"])
def test_resolve_camera_name_requires_exactly_one_match(name: str) -> None:
    with pytest.raises(CameraError, match="Use --camera-id") as caught:
        resolve_camera(CameraSelector(name=name), CAMERAS)
    assert "Built-in Camera" in str(caught.value)
    assert "desk-wide" in str(caught.value)


def test_resolve_camera_id_is_exact() -> None:
    assert resolve_camera(CameraSelector(unique_id="desk"), CAMERAS) == (2, CAMERAS[1])
    with pytest.raises(CameraError, match="matched no cameras"):
        resolve_camera(CameraSelector(unique_id="DESK"), CAMERAS)


def test_resolve_camera_rejects_duplicate_ids() -> None:
    duplicates = (*CAMERAS, CameraInfo(8, "Duplicate", "desk", False))
    with pytest.raises(CameraError, match="matched multiple cameras"):
        resolve_camera(CameraSelector(unique_id="desk"), duplicates)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"index": 0, "name": "camera"}],
)
def test_camera_selector_requires_exactly_one_selector(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CameraSelector(**kwargs)  # type: ignore[arg-type]


def test_v4l2_cameras_filters_sorts_and_uses_smallest_by_id(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys" / "class" / "video4linux"
    by_id = tmp_path / "dev" / "v4l" / "by-id"
    devices = tmp_path / "dev"
    sysfs.mkdir(parents=True)
    by_id.mkdir(parents=True)
    devices.mkdir(exist_ok=True)
    for index, name in ((10, "Ten"), (2, "Two"), (3, "Metadata"), (0, "Default")):
        node = sysfs / f"video{index}"
        node.mkdir()
        (node / "name").write_text(name + "\n", encoding="utf-8")
        (devices / f"video{index}").touch()
    (by_id / "z-camera").symlink_to(devices / "video2")
    (by_id / "a-camera").symlink_to(devices / "video2")
    (by_id / "default-camera").symlink_to(devices / "video0")

    cameras = _v4l2_cameras(sysfs, by_id, lambda path: path.name != "video3")

    assert cameras == (
        CameraInfo(0, "Default", "default-camera", True),
        CameraInfo(2, "Two", "a-camera", False),
        CameraInfo(10, "Ten", None, False),
    )


def test_avfoundation_cameras_put_video_before_muxed() -> None:
    video = SimpleNamespace(localizedName=lambda: "Video", uniqueID=lambda: "video-id")
    muxed = SimpleNamespace(localizedName="Muxed", uniqueID="muxed-id")
    assert _avfoundation_cameras((video,), (muxed,)) == (
        CameraInfo(0, "Video", "video-id", True),
        CameraInfo(1, "Muxed", "muxed-id", False),
    )


class FakeCapture:
    def __init__(
        self,
        *,
        width: float = 1280.0,
        height: float = 720.0,
        fps: float = 29.97,
        opened: bool = True,
    ) -> None:
        self.values = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
        }
        self.opened = opened
        self.set_calls: list[tuple[int, float]] = []
        self.release_calls = 0
        self.read_error = False

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        return self.values[prop]

    def read(self) -> tuple[bool, np.ndarray]:
        if self.read_error:
            raise cv2.error("read failed")
        return True, np.zeros((2, 3, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_calls += 1


def test_webcam_pins_backend_and_reports_actuals(monkeypatch) -> None:
    capture = FakeCapture()
    opened_with: list[tuple[int, int]] = []

    def open_capture(index: int, api: int) -> FakeCapture:
        opened_with.append((index, api))
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    webcam = Webcam(
        CameraSelector(unique_id="desk"),
        640,
        480,
        False,
        fps=24.0,
        enumerator=lambda: CAMERAS,
    )

    assert webcam.connect() == CameraActuals(2, "Desk Camera", "desk", 1280, 720, 29.97)
    expected_api = (
        cv2.CAP_V4L2
        if sys.platform not in {"darwin", "win32"}
        else (cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_MSMF)
    )
    assert opened_with == [(2, expected_api)]
    assert (cv2.CAP_PROP_FPS, 24.0) in capture.set_calls


def test_webcam_sets_fps_only_when_requested_and_zero_actual_is_none(monkeypatch) -> None:
    capture = FakeCapture(fps=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: capture)
    actuals = Webcam(0, 640, 480, False, enumerator=lambda: ()).connect()
    assert actuals.fps is None
    assert all(prop != cv2.CAP_PROP_FPS for prop, _value in capture.set_calls)


def test_webcam_invalid_dimensions_release_and_raise(monkeypatch) -> None:
    capture = FakeCapture(width=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: capture)
    with pytest.raises(CameraError, match="invalid width"):
        Webcam(0, 640, 480, False, enumerator=lambda: ()).connect()
    assert capture.release_calls == 1


def test_webcam_wraps_open_and_read_cv2_errors(monkeypatch) -> None:
    def fail_open(*_args: object) -> object:
        raise cv2.error("open failed")

    monkeypatch.setattr(cv2, "VideoCapture", fail_open)
    with pytest.raises(CameraError, match="configure"):
        Webcam(0, 640, 480, False, enumerator=lambda: ()).connect()

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: capture)
    webcam = Webcam(0, 640, 480, False, enumerator=lambda: ())
    webcam.connect()
    capture.read_error = True
    with pytest.raises(CameraError, match="read"):
        webcam.read()


def test_webcam_reconnect_follows_stable_id_across_index_reassignment(monkeypatch) -> None:
    cameras: list[CameraInfo] = [CameraInfo(1, "Desk Camera", "desk", False)]
    opened: list[int] = []

    def open_capture(index: int, _api: int) -> FakeCapture:
        opened.append(index)
        return FakeCapture()

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    webcam = Webcam(1, 640, 480, False, enumerator=lambda: tuple(cameras))
    assert webcam.identity == CameraSelector(index=1)

    assert webcam.connect().unique_id == "desk"
    assert webcam.identity == CameraSelector(unique_id="desk")
    webcam.disconnect()

    cameras[:] = [CameraInfo(1, "Other Camera", "other", False)]
    with pytest.raises(CameraError, match="camera id 'desk' matched no cameras"):
        webcam.connect()

    cameras[:] = [
        CameraInfo(1, "Other Camera", "other", False),
        CameraInfo(3, "Desk Camera", "desk", False),
    ]
    actuals = webcam.connect()
    assert (actuals.index, actuals.unique_id) == (3, "desk")
    assert webcam.device_index == 3
    assert opened == [1, 3]


def test_webcam_keeps_numeric_selection_without_a_stable_id(monkeypatch) -> None:
    opened: list[int] = []

    def open_capture(index: int, _api: int) -> FakeCapture:
        opened.append(index)
        return FakeCapture()

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    webcam = Webcam(2, 640, 480, False, enumerator=lambda: (CameraInfo(2, "Cam", None, False),))
    webcam.connect()
    webcam.disconnect()
    webcam.connect()
    assert webcam.identity == CameraSelector(index=2)
    assert opened == [2, 2]
