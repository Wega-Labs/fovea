# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A process CLI with NDJSON output, stdin controls, stable exit codes, and environment diagnostics.
- Cross-platform CI that opens the pinned MediaPipe graph on Ubuntu and macOS.
- Event vocabulary v2 (protocol `1.1`): `saccade`, `wink`, `double_blink`, and `long_blink`
  events, an optional `gaze_point.pursuit` flag, matching `hello` capabilities, and
  `GazeSettings` thresholds for each detector.
- Capture and latency additions (protocol `1.2`): a latest-frame hand-off, capture
  timestamps, `GazePoint.latency_ms`, `Diagnostics` latency percentiles and
  `dropped_frames`, and a `--fps` processing cap.
- Online self-calibration (protocol `1.3`): the `observe` control,
  `calibration_updated` event, schema-v4 anchor persistence, robust residual
  quarantine, guarded weighted refits, and the `--no-online` runtime switch.

### Changed

- Pinned the working MediaPipe 0.10.x line with compatible NumPy and OpenCV ranges.
- Made calibration viable at desk distance and surfaced tracking-quality detail to hosts.
- Corrected the head-pose pitch convention and uncalibrated vertical mapping.
- Webcam capture and fixture replay now share one per-frame event processor, so `fovea replay`
  emits the same fixations, measured blink durations, target events, and reports as live capture.

### Fixed

- `fixation` events now fire with real frame timestamps: the stability window kept only samples
  strictly newer than the cutoff, so its span reached `stability_ms` only when a sample landed
  exactly on it.

[Unreleased]: https://github.com/Wega-Labs/fovea/compare/v0.1.0...HEAD
