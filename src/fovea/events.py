"""Platform-neutral events emitted by the Fovea engine."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Eye(StrEnum):
    """The eye associated with an event."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class GesturePhase(StrEnum):
    """The lifecycle phase of a gesture or manipulation."""

    STARTED = "started"
    UPDATED = "updated"
    ENDED = "ended"
    CANCELLED = "cancelled"


class TrackingStatus(StrEnum):
    """The engine's current ability to produce reliable events."""

    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    LOST = "lost"


def _validate_confidence(confidence: float) -> None:
    if not 0.0 <= confidence <= 1.0:
        msg = "confidence must be between 0.0 and 1.0"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GazePoint:
    """A calibrated gaze coordinate in an adapter-defined coordinate space.

    ``pursuit`` is true while the gaze is tracking a smoothly moving target:
    sustained sub-saccadic, directionally coherent motion.
    """

    x: float
    y: float
    confidence: float
    timestamp_ns: int
    target_id: str | None = None
    snapped_x: float | None = None
    snapped_y: float | None = None
    pursuit: bool = False
    # Capture-to-ready-to-emit milliseconds; None when unknown (replay).
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class TargetEnter:
    """Gaze acquired a host-registered target."""

    id: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class TargetLeave:
    """Gaze left the hysteresis boundary of an active target."""

    id: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class DwellProgress:
    """Normalized progress toward the active target's dwell threshold."""

    id: str
    progress: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class Dwell:
    """The active target reached the configured dwell threshold."""

    id: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class Fixation:
    """A stable gaze position maintained for a measurable duration."""

    x: float
    y: float
    duration_ms: float
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class Blink:
    """A detected blink that applications may use as an explicit trigger."""

    eye: Eye
    duration_ms: float
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class Saccade:
    """A rapid gaze jump between two positions, reported when it lands.

    ``from_x``/``from_y`` are the last stable point before the jump and
    ``to_x``/``to_y`` the landing point, both in the adapter's coordinate
    space. ``amplitude`` is the Euclidean distance between them in that space,
    ``duration_ms`` spans onset to landing, and ``timestamp_ns`` is the landing
    time.
    """

    from_x: float
    from_y: float
    to_x: float
    to_y: float
    amplitude: float
    duration_ms: float
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class Wink:
    """A deliberate single-eye closure while the other eye stayed open.

    ``eye`` names the closed eye by the landmark topology label the engine was
    given (``features.left`` is ``Eye.LEFT``), not the user's anatomical side.
    """

    eye: Literal[Eye.LEFT, Eye.RIGHT]
    duration_ms: float
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        if self.eye not in (Eye.LEFT, Eye.RIGHT):
            raise ValueError("wink eye must be left or right")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class DoubleBlink:
    """Two natural blinks in quick succession; the timestamp is the second reopen."""

    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class LongBlink:
    """A deliberate blink held longer than the user's natural blink duration."""

    duration_ms: float
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class Gesture:
    """A recognized hand gesture and its lifecycle phase."""

    kind: str
    phase: GesturePhase
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class Manipulation:
    """A gesture-driven transformation applied to a gaze-selected target."""

    target_id: str
    phase: GesturePhase
    delta_x: float
    delta_y: float
    scale: float
    rotation_degrees: float
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class TrackingState:
    """A change in the engine's tracking status."""

    status: TrackingStatus
    confidence: float
    timestamp_ns: int
    detail: str = ""

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class CameraReady:
    """An opened camera and its negotiated capture properties."""

    name: str
    unique_id: str | None
    index: int
    width: int
    height: int
    fps: float | None
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CameraLost:
    """A camera outage that ends the stream or starts reconnect backoff."""

    name: str
    unique_id: str | None
    index: int
    reason: Literal["read_failed", "read_error"]
    reconnecting: bool
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CalibrationCue:
    """The calibration target the user should look at right now."""

    label: str
    x: float
    y: float
    index: int
    total: int
    samples: int
    needed: int
    instruction: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CalibrationWarning:
    """A calibration layout is usable but has weak display coverage."""

    message: str
    coverage: float
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CalibrationDone:
    """A completed calibration fit and its camera-free validation metrics."""

    n_points: int
    coverage: float
    loo_error: float
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CalibrationUpdated:
    """A successful online calibration refit and its health measurement."""

    n: int
    loo_error: float
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class GazeTestPoint:
    """One expected and predicted point from a guided gaze test."""

    expected_x: float
    expected_y: float
    predicted_x: float
    predicted_y: float
    error: float


@dataclass(frozen=True, slots=True)
class GazeTestDone:
    """A completed guided gaze test and its point-level results."""

    n_points: int
    mean_error: float
    median_error: float
    max_error: float
    points: tuple[GazeTestPoint, ...]
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Rate-limited local pipeline health measurements."""

    fps: float
    latency_ms: float
    face_width: float
    yaw_deg: float
    pitch_deg: float
    timestamp_ns: int
    # Capture-to-ready-to-emit percentiles over recent gaze points; None before the first.
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    # Admitted frames discarded because a newer frame arrived first; cumulative per session.
    dropped_frames: int = 0


type FoveaEvent = (
    GazePoint
    | TargetEnter
    | TargetLeave
    | DwellProgress
    | Dwell
    | Fixation
    | Blink
    | Saccade
    | Wink
    | DoubleBlink
    | LongBlink
    | Gesture
    | Manipulation
    | TrackingState
    | CameraReady
    | CameraLost
    | CalibrationCue
    | CalibrationWarning
    | CalibrationDone
    | CalibrationUpdated
    | GazeTestDone
    | Diagnostics
)
