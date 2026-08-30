"""Platform-neutral events emitted by the Fovea engine."""

from dataclasses import dataclass
from enum import StrEnum


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
    """A calibrated gaze coordinate in an adapter-defined coordinate space."""

    x: float
    y: float
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


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

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


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


type FoveaEvent = (
    GazePoint | Fixation | Blink | Gesture | Manipulation | TrackingState | CalibrationCue
)
