"""Camera enumeration, selection, and the OpenCV capture wrapper."""

from __future__ import annotations

import math
import os
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CameraError(RuntimeError):
    """Raised when a camera cannot be selected, opened, or read."""


class CameraEnumerationUnavailable(CameraError):
    """Raised when this installation cannot enumerate cameras by identity."""


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """One platform camera and its stable identity, when the OS exposes one."""

    index: int
    name: str
    unique_id: str | None
    default: bool


@dataclass(frozen=True, slots=True)
class CameraSelector:
    """Exactly one way to select a camera."""

    index: int | None = None
    name: str | None = None
    unique_id: str | None = None

    def __post_init__(self) -> None:
        if sum(value is not None for value in (self.index, self.name, self.unique_id)) != 1:
            raise ValueError("exactly one camera selector must be set")


@dataclass(frozen=True, slots=True)
class CameraActuals:
    """Identity and negotiated capture properties for an opened camera."""

    index: int
    name: str
    unique_id: str | None
    width: int
    height: int
    fps: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
            or not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
        ):
            raise CameraError("camera reported invalid capture dimensions")


@dataclass(slots=True)
class ReconnectPolicy:
    """Track sustained read failure and exponential reconnect backoff."""

    lost_after_s: float = 1.0
    initial_delay_s: float = 0.5
    factor: float = 2.0
    max_delay_s: float = 5.0
    _failed_since: float | None = field(default=None, init=False, repr=False)
    _loss_reported: bool = field(default=False, init=False, repr=False)
    _delay_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.lost_after_s, self.initial_delay_s, self.factor, self.max_delay_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reconnect policy values must be finite")
        if self.lost_after_s < 0.0:
            raise ValueError("lost_after_s must be zero or greater")
        if self.initial_delay_s < 0.0 or self.max_delay_s < 0.0:
            raise ValueError("reconnect delays must be zero or greater")
        if self.factor < 1.0:
            raise ValueError("reconnect factor must be at least one")
        self._delay_s = min(self.initial_delay_s, self.max_delay_s)

    def read_failed(self, now: float) -> bool:
        """Return true once when failures have continuously exceeded the grace period."""
        if self._failed_since is None:
            self._failed_since = now
        if self._loss_reported or now - self._failed_since < self.lost_after_s:
            return False
        self._loss_reported = True
        return True

    def frame_ok(self) -> None:
        """End a run of failed reads without changing reconnect backoff."""
        self._failed_since = None
        self._loss_reported = False

    def next_delay(self) -> float:
        """Return the next capped reconnect delay and advance the backoff."""
        delay = self._delay_s
        self._delay_s = min(self.max_delay_s, self._delay_s * self.factor)
        return delay

    def reconnected(self) -> None:
        """Reset failure tracking and backoff after a successful reopen."""
        self.frame_ok()
        self._delay_s = min(self.initial_delay_s, self.max_delay_s)


def _object_value(value: object, attribute: str) -> object:
    member = getattr(value, attribute)
    return member() if callable(member) else member


def _avfoundation_cameras(
    video_devices: Sequence[object],
    muxed_devices: Sequence[object],
) -> tuple[CameraInfo, ...]:
    """Translate AVFoundation devices in OpenCV's video-then-muxed order."""
    cameras: list[CameraInfo] = []
    for index, device in enumerate((*video_devices, *muxed_devices)):
        name = str(_object_value(device, "localizedName"))
        unique_id_value = _object_value(device, "uniqueID")
        unique_id = None if unique_id_value is None else str(unique_id_value)
        cameras.append(CameraInfo(index, name, unique_id, index == 0))
    return tuple(cameras)


def _v4l2_cameras(
    sysfs_dir: Path,
    by_id_dir: Path,
    is_capture: Callable[[Path], bool],
) -> tuple[CameraInfo, ...]:
    """Enumerate Linux V4L2 capture nodes using injectable filesystem seams."""
    nodes: list[tuple[int, Path]] = []
    try:
        entries = tuple(sysfs_dir.iterdir())
    except OSError as exc:
        raise CameraEnumerationUnavailable(f"could not enumerate Linux cameras: {exc}") from exc
    for entry in entries:
        if not entry.name.startswith("video") or not entry.name[5:].isdigit():
            continue
        nodes.append((int(entry.name[5:]), entry))
    nodes.sort(key=lambda item: item[0])

    links_by_node: dict[str, list[str]] = {}
    try:
        links = tuple(by_id_dir.iterdir())
    except FileNotFoundError:
        links = ()
    except OSError as exc:
        raise CameraEnumerationUnavailable(f"could not enumerate Linux camera ids: {exc}") from exc
    for link in links:
        try:
            target_name = link.resolve(strict=True).name
        except OSError:
            continue
        if target_name.startswith("video") and target_name[5:].isdigit():
            links_by_node.setdefault(target_name, []).append(link.name)

    cameras: list[CameraInfo] = []
    for index, node in nodes:
        device = Path("/dev") / node.name
        if not is_capture(device):
            continue
        try:
            name = (node / "name").read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CameraEnumerationUnavailable(
                f"could not read camera metadata for {device}: {exc}"
            ) from exc
        ids = sorted(links_by_node.get(node.name, ()))
        unique_id = ids[0] if ids else None
        cameras.append(CameraInfo(index, name or f"camera {index}", unique_id, index == 0))
    return tuple(cameras)


def _is_v4l2_capture(device: Path) -> bool:
    """Return whether ``device`` advertises V4L2 single-plane video capture."""
    import fcntl

    vidioc_querycap = 0x80685600
    v4l2_cap_video_capture = 0x00000001
    v4l2_cap_device_caps = 0x80000000
    capability = bytearray(104)
    try:
        descriptor = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        fcntl.ioctl(descriptor, vidioc_querycap, capability, True)
    except OSError:
        return False
    finally:
        os.close(descriptor)
    _version, capabilities, device_caps = struct.unpack_from("=III", capability, 80)
    effective = device_caps if capabilities & v4l2_cap_device_caps else capabilities
    return bool(effective & v4l2_cap_video_capture)


def enumerate_cameras() -> tuple[CameraInfo, ...]:
    """Enumerate cameras in the same order as the platform's pinned OpenCV backend."""
    if sys.platform == "darwin":
        try:
            from AVFoundation import (  # type: ignore[import-not-found]
                AVCaptureDevice,
                AVMediaTypeMuxed,
                AVMediaTypeVideo,
            )
        except ImportError as exc:
            raise CameraEnumerationUnavailable(
                "camera enumeration on macOS requires fovea-input[macos]"
            ) from exc
        return _avfoundation_cameras(
            tuple(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)),
            tuple(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed)),
        )
    if sys.platform.startswith("linux"):
        return _v4l2_cameras(
            Path("/sys/class/video4linux"),
            Path("/dev/v4l/by-id"),
            _is_v4l2_capture,
        )
    if sys.platform == "win32":
        raise CameraEnumerationUnavailable(
            "camera enumeration is not implemented on Windows yet; use --camera N"
        )
    raise CameraEnumerationUnavailable(
        f"camera enumeration is not implemented on {sys.platform}; use --camera N"
    )


def _candidate_list(cameras: Sequence[CameraInfo]) -> str:
    if not cameras:
        return "no cameras were found"
    return "; ".join(
        f"{camera.index}: {camera.name}"
        + ("" if camera.unique_id is None else f" (id: {camera.unique_id})")
        for camera in cameras
    )


def resolve_camera(
    selector: CameraSelector,
    cameras: Sequence[CameraInfo],
) -> tuple[int, CameraInfo | None]:
    """Resolve a name or stable id; numeric indices intentionally pass through."""
    if selector.index is not None:
        info = next((camera for camera in cameras if camera.index == selector.index), None)
        return selector.index, info
    if selector.name is not None:
        needle = selector.name.casefold()
        matches = [camera for camera in cameras if needle in camera.name.casefold()]
        if len(matches) == 1:
            return matches[0].index, matches[0]
        qualifier = "no" if not matches else "multiple"
        raise CameraError(
            f"camera name {selector.name!r} matched {qualifier} cameras; "
            f"candidates: {_candidate_list(cameras)}. Use --camera-id to select exactly."
        )
    assert selector.unique_id is not None
    matches = [camera for camera in cameras if camera.unique_id == selector.unique_id]
    if len(matches) == 1:
        return matches[0].index, matches[0]
    qualifier = "no" if not matches else "multiple"
    raise CameraError(
        f"camera id {selector.unique_id!r} matched {qualifier} cameras; "
        f"candidates: {_candidate_list(cameras)}"
    )


def _capture_api(cv2: Any) -> int:
    if sys.platform == "darwin":
        return int(cv2.CAP_AVFOUNDATION)
    if sys.platform == "win32":
        return int(cv2.CAP_MSMF)
    return int(cv2.CAP_V4L2)


class Webcam:
    def __init__(
        self,
        device_index: int | CameraSelector,
        width: int,
        height: int,
        mirror: bool,
        *,
        fps: float | None = None,
        enumerator: Callable[[], tuple[CameraInfo, ...]] = enumerate_cameras,
    ) -> None:
        self.selector = (
            CameraSelector(index=device_index) if isinstance(device_index, int) else device_index
        )
        self.device_index = self.selector.index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.fps = fps
        self._enumerator = enumerator
        self._capture: Any = None

    def connect(self) -> CameraActuals:
        import cv2

        cameras: tuple[CameraInfo, ...] = ()
        try:
            cameras = self._enumerator()
        except CameraEnumerationUnavailable:
            if self.selector.index is None:
                raise
        index, info = resolve_camera(self.selector, cameras)
        capture: Any = None
        try:
            capture = cv2.VideoCapture(index, _capture_api(cv2))
            if not capture.isOpened():
                msg = (
                    f"Could not open camera {index}. "
                    "Check that it is connected and permissions are granted."
                )
                raise CameraError(msg)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps is not None:
                capture.set(cv2.CAP_PROP_FPS, self.fps)
            width = self._dimension(capture.get(cv2.CAP_PROP_FRAME_WIDTH), "width")
            height = self._dimension(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), "height")
            try:
                raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
            except (TypeError, ValueError) as exc:
                raise CameraError("camera reported invalid fps") from exc
            actual_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0.0 else None
        except CameraError:
            if capture is not None:
                capture.release()
            raise
        except (cv2.error, OSError) as exc:
            if capture is not None:
                capture.release()
            raise CameraError(f"Could not configure camera {index}: {exc}") from exc
        actuals = CameraActuals(
            index=index,
            name=info.name if info is not None else f"camera {index}",
            unique_id=None if info is None else info.unique_id,
            width=width,
            height=height,
            fps=actual_fps,
        )
        self.device_index = index
        self._capture = capture
        return actuals

    @staticmethod
    def _dimension(value: Any, label: str) -> int:
        try:
            measured = float(value)
        except (TypeError, ValueError) as exc:
            raise CameraError(f"camera reported invalid {label}") from exc
        if not math.isfinite(measured) or measured <= 0.0:
            raise CameraError(f"camera reported invalid {label}")
        rounded = round(measured)
        if rounded <= 0:
            raise CameraError(f"camera reported invalid {label}")
        return rounded

    def read(self) -> NDArray[np.uint8] | None:
        import cv2

        if self._capture is None:
            raise CameraError("Webcam is not connected.")
        try:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return None
            if self.mirror:
                return np.asarray(cv2.flip(frame, 1))
            return np.asarray(frame)
        except (cv2.error, OSError) as exc:
            raise CameraError(f"Could not read from camera: {exc}") from exc

    @property
    def is_connected(self) -> bool:
        return self._capture is not None

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
