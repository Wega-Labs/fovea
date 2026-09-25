# Minimal Python host

An in-process Fovea integration in one file: create an event source, iterate its typed events, and react to
`TrackingState` and `GazePoint`.

By default it replays the checked-in synthetic landmark fixture, so it runs without a camera, a face, the model
download, or network access:

```bash
uv run python examples/python-minimal/main.py
```

```text
gaze: x=0.500 y=0.500 confidence=1.00
tracking: active confidence=1.00
...
events: Fixation=5, GazePoint=30, TrackingState=30
```

To read the local camera, download the pinned model first and grant camera permission:

```bash
uv run python scripts/download_mediapipe_model.py
uv run python examples/python-minimal/main.py --webcam --max-frames 300
```

The replay path writes calibration only to a temporary directory. `tests/test_examples.py` runs the replay path in
CI.
