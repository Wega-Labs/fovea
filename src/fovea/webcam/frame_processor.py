"""Camera-independent translation from landmark frames to Fovea events.

``GazeFrameProcessor`` owns every per-frame detector (blink, fixation, target
tracking, saccade/pursuit, wink, and blink triggers) so that the webcam source
and fixture replay share one event-assembly path. Sources keep only what is
camera- or UI-bound: capture, display side effects, the diagnostics rate-limit
decision, and pending control events.

A processor is constructed once per ``events()`` session. The detector
instances *are* the session boundary; never reuse a processor across sessions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from fovea.events import (
    CalibrationCue,
    CalibrationDone,
    Diagnostics,
    FoveaEvent,
    GazePoint,
    GazeTestDone,
    GazeTestPoint,
    TrackingState,
    TrackingStatus,
)
from fovea.webcam.engine import GazeEngine, GazeOutput, GazeSettings
from fovea.webcam.targeting import TargetIntentEvent, TargetMatch, TargetRect, TargetTracker
from fovea.webcam.temporal import (
    BlinkDetector,
    BlinkTriggerDetector,
    FixationDetector,
    SaccadeDetector,
    VelocityUpdate,
    WinkDetector,
)

_STILL = VelocityUpdate(moving=False, saccading=False, pursuit=False)


def _tracking_status(label: str) -> TrackingStatus:
    if label == "GOOD":
        return TrackingStatus.ACTIVE
    if label in {"FAIR", "POOR"}:
        return TrackingStatus.UNCERTAIN
    return TrackingStatus.LOST


def _diagnostics_event(output: GazeOutput, timestamp_ns: int) -> Diagnostics:
    features = output.features
    return Diagnostics(
        fps=output.fps,
        latency_ms=output.latency_ms,
        face_width=0.0 if features is None else features.face_width,
        yaw_deg=0.0 if features is None else features.yaw_deg,
        pitch_deg=0.0 if features is None else features.pitch_deg,
        timestamp_ns=timestamp_ns,
    )


def _gaze_test_event(report: dict[str, object], timestamp_ns: int) -> GazeTestDone | None:
    points_raw = report.get("points")
    n_points = report.get("n")
    mean_error = report.get("mean_error")
    median_error = report.get("median_error")
    max_error = report.get("max_error")
    aggregates = (n_points, mean_error, median_error, max_error)
    if any(not isinstance(value, int | float) or isinstance(value, bool) for value in aggregates):
        return None
    if not isinstance(points_raw, list):
        return None
    points: list[GazeTestPoint] = []
    for point_raw in points_raw:
        if not isinstance(point_raw, dict):
            return None
        expected = point_raw.get("expected")
        predicted = point_raw.get("predicted")
        error = point_raw.get("error")
        if not isinstance(expected, list) or len(expected) != 2:
            return None
        if not isinstance(predicted, list) or len(predicted) != 2:
            return None
        coordinates: list[float] = []
        for value in (*expected, *predicted, error):
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            coordinates.append(float(value))
        expected_x, expected_y, predicted_x, predicted_y, point_error = coordinates
        points.append(
            GazeTestPoint(
                expected_x=expected_x,
                expected_y=expected_y,
                predicted_x=predicted_x,
                predicted_y=predicted_y,
                error=point_error,
            )
        )
    assert isinstance(n_points, int | float)
    assert isinstance(mean_error, int | float)
    assert isinstance(median_error, int | float)
    assert isinstance(max_error, int | float)
    return GazeTestDone(
        n_points=int(n_points),
        mean_error=float(mean_error),
        median_error=float(median_error),
        max_error=float(max_error),
        points=tuple(points),
        timestamp_ns=timestamp_ns,
    )


class GazeFrameProcessor:
    """Process landmark frames using the same path for webcams and replay.

    Per frame the events are ordered ``CalibrationCue``, ``TrackingState``,
    ``Diagnostics`` (when due), ``CalibrationDone``, ``GazeTestDone``,
    ``Blink``, ``LongBlink``/``DoubleBlink``, ``Wink``, ``Saccade``,
    ``GazePoint``, target events, then ``Fixation``.

    Trigger events (wink, long blink, double blink) are silent for any frame
    on which the calibration or gaze-test wizard was active at frame start,
    and stay silent afterwards until both valid eyes have been seen open on a
    later non-wizard frame, so a closure spanning the wizard boundary can never
    become a trigger.
    """

    def __init__(
        self,
        engine: GazeEngine,
        settings: GazeSettings,
        targets: Sequence[TargetRect] = (),
    ) -> None:
        self.engine = engine
        self.settings = settings
        self._blink_detector = BlinkDetector()
        self._fixation_detector = FixationDetector(
            stability_ms=settings.stability_ms,
            radius=settings.hysteresis,
        )
        self._target_tracker = TargetTracker(
            dwell_ms=settings.dwell_ms,
            hysteresis=settings.hysteresis,
            expand=settings.target_expand,
            snap_radius=settings.snap_radius,
            targets=tuple(targets),
        )
        self._saccade_detector = SaccadeDetector(
            velocity_threshold=settings.saccade_velocity,
            pursuit_velocity=settings.pursuit_velocity,
            pursuit_ms=settings.pursuit_ms,
            pursuit_coherence=settings.pursuit_coherence,
        )
        self._wink_detector = WinkDetector(
            min_ms=settings.wink_min_ms,
            max_ms=settings.wink_max_ms,
        )
        self._blink_trigger_detector = BlinkTriggerDetector(
            long_blink_ms=settings.long_blink_ms,
            long_blink_factor=settings.long_blink_factor,
            natural_blink_window=settings.natural_blink_window,
            double_blink_ms=settings.double_blink_ms,
        )
        self._trigger_quiet_until_open = False
        self._emitted_calibration_report: dict[str, object] | None = None
        self._emitted_test_report: dict[str, object] | None = None

    def replace_targets(
        self,
        targets: Sequence[TargetRect],
        timestamp_ns: int,
    ) -> tuple[TargetIntentEvent, ...]:
        """Replace host-registered targets and return any resulting leave event."""
        return self._target_tracker.replace_targets(tuple(targets), timestamp_ns)

    def events_for_lost_frame(
        self,
        fps: float,
        timestamp_ns: int,
        *,
        diagnostics_due: bool = False,
    ) -> tuple[FoveaEvent, ...]:
        """Events for a capture that produced no frame at all."""
        self._reset_detectors()
        self._target_tracker.freeze(timestamp_ns)
        events: list[FoveaEvent] = [
            TrackingState(
                status=TrackingStatus.LOST,
                confidence=0.0,
                timestamp_ns=timestamp_ns,
            )
        ]
        if diagnostics_due:
            events.append(
                Diagnostics(
                    fps=fps,
                    latency_ms=0.0,
                    face_width=0.0,
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    timestamp_ns=timestamp_ns,
                )
            )
        return tuple(events)

    def events_for_frame(
        self,
        landmarks: Sequence[Any] | None,
        image_w: float,
        image_h: float,
        dt: float,
        fps: float,
        timestamp_ns: int,
        blendshapes: Mapping[str, float] | None = None,
        *,
        diagnostics_due: bool = False,
    ) -> tuple[FoveaEvent, ...]:
        engine = self.engine
        wizard_active = engine.wizard is not None
        output = engine.process(
            landmarks,
            image_w,
            image_h,
            dt,
            fps,
            blendshapes=blendshapes,
        )
        events: list[FoveaEvent] = []
        if wizard_active:
            self._trigger_quiet_until_open = True
            self._reset_triggers()

        wizard = engine.wizard
        if wizard is not None and not wizard.done:
            events.append(
                CalibrationCue(
                    label=wizard.label,
                    x=wizard.sx,
                    y=wizard.sy,
                    index=wizard.index,
                    total=len(engine.targets),
                    samples=wizard.samples,
                    needed=wizard.needed,
                    instruction=wizard.instruction,
                    timestamp_ns=timestamp_ns,
                )
            )

        status = _tracking_status(output.tracking)
        events.append(
            TrackingState(
                status=status,
                confidence=output.confidence,
                timestamp_ns=timestamp_ns,
                detail="" if output.features is None else output.features.message,
            )
        )

        if diagnostics_due:
            events.append(_diagnostics_event(output, timestamp_ns))

        calibration_report = engine.last_calibration_report
        if calibration_report and calibration_report is not self._emitted_calibration_report:
            events.append(
                CalibrationDone(
                    n_points=cast(int, calibration_report["n_points"]),
                    coverage=cast(float, calibration_report["coverage"]),
                    loo_error=cast(float, calibration_report["loo_error"]),
                    timestamp_ns=timestamp_ns,
                )
            )
            self._emitted_calibration_report = calibration_report

        test_report = engine.last_test_report
        if test_report and test_report is not self._emitted_test_report:
            test_event = _gaze_test_event(test_report, timestamp_ns)
            if test_event is not None:
                engine.resume_after_gaze_test()
                events.append(test_event)
                self._emitted_test_report = test_report

        features = output.features
        left_closed: bool | None = None
        right_closed: bool | None = None
        if features is None or output.tracking == "LOST":
            self._reset_detectors()
        else:
            blink_ear = self.settings.blink_ear
            left_closed = features.left.ear < blink_ear if features.left.valid else None
            right_closed = features.right.ear < blink_ear if features.right.valid else None
            blink_event = self._blink_detector.update(
                features.blink,
                output.confidence,
                timestamp_ns,
            )
            quiet = self._trigger_quiet_until_open
            if blink_event is not None:
                events.append(blink_event)
                if not quiet:
                    events.extend(self._blink_trigger_detector.update(blink_event))
            if not quiet:
                wink_event = self._wink_detector.update(
                    left_closed,
                    right_closed,
                    output.confidence,
                    timestamp_ns,
                )
                if wink_event is not None:
                    events.append(wink_event)
            elif not wizard_active:
                self._reset_triggers()
                if left_closed is False and right_closed is False:
                    self._trigger_quiet_until_open = False

        target_match: TargetMatch | None = None
        target_events: tuple[TargetIntentEvent, ...] = ()
        gaze_active = status is TrackingStatus.ACTIVE and output.valid and output.screen is not None
        if gaze_active and output.screen is not None:
            target_update = self._target_tracker.update(
                output.screen.x,
                output.screen.y,
                output.confidence,
                timestamp_ns,
            )
            target_match = target_update.match
            target_events = target_update.events
        else:
            target_match = self._target_tracker.freeze(timestamp_ns)

        if output.valid and output.screen is not None:
            screen = output.screen
            eye_closed = left_closed is True or right_closed is True
            if gaze_active and not eye_closed:
                velocity = self._saccade_detector.update(screen.x, screen.y, timestamp_ns)
            else:
                self._saccade_detector.reset()
                velocity = _STILL
            if velocity.saccade is not None:
                events.append(velocity.saccade)
            events.append(
                GazePoint(
                    x=screen.x,
                    y=screen.y,
                    confidence=output.confidence,
                    timestamp_ns=timestamp_ns,
                    target_id=(None if target_match is None else target_match.target_id),
                    snapped_x=(None if target_match is None else target_match.snapped_x),
                    snapped_y=(None if target_match is None else target_match.snapped_y),
                    pursuit=velocity.pursuit,
                )
            )
            events.extend(target_events)
            if velocity.moving:
                self._fixation_detector.reset()
            fixation = self._fixation_detector.update(
                screen.x,
                screen.y,
                output.confidence,
                timestamp_ns,
            )
            if fixation is not None and not velocity.moving:
                events.append(fixation)
        else:
            self._fixation_detector.reset()
            self._saccade_detector.reset()
        return tuple(events)

    def _reset_detectors(self) -> None:
        self._blink_detector.reset()
        self._fixation_detector.reset()
        self._saccade_detector.reset()
        self._reset_triggers()

    def _reset_triggers(self) -> None:
        self._wink_detector.reset()
        self._blink_trigger_detector.reset()
