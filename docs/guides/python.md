# Python host guide

## Run without a camera

[`examples/python-minimal`](https://github.com/Wega-Labs/fovea/tree/main/examples/python-minimal) replays the
checked-in synthetic landmark fixture through the real engine, so it needs no camera, face, model download, or network:

```bash
uv run python examples/python-minimal/main.py
```

```text
gaze: x=0.500 y=0.500 confidence=1.00
tracking: active confidence=1.00
...
events: Fixation=5, GazePoint=30, TrackingState=30
```

CI runs this example on every change.

## Read the webcam

```python
from fovea import GazePoint, GazeSettings, TrackingState, WebcamEventSource

source = WebcamEventSource(GazeSettings(), max_frames=600, show_calibration=True)
try:
    for event in source.events():
        if isinstance(event, TrackingState):
            print("tracking", event.status)
        elif isinstance(event, GazePoint):
            print("gaze", event.x, event.y, event.confidence)
finally:
    source.close()
```

Download the pinned model first with `uv run python scripts/download_mediapipe_model.py`. `events()` blocks the
calling thread; `close()` is safe to call from any thread and releases the camera.

## Calibration from Python

- `force_calibrate=True` starts calibration immediately; `start_calibration(targets)` restarts it between frames.
- `show_calibration=True` opens Fovea's fullscreen target window; otherwise render the `CalibrationCue` events
  yourself.
- `set_targets(...)` registers `TargetRect`s for target-aware gaze and dwell.
- `observe(x, y)` reports a host-confirmed gaze location for online refinement.

Calibration is stored in the platform's user data directory; set `FOVEA_DATA_DIR` or
`GazeSettings.calibration_path` to change it.

All classes and settings are listed in the [Python API](../reference/python-api.md).
