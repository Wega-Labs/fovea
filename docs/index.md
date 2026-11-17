# Fovea

**Your gaze is an input.** Fovea is an open-source gaze input engine that turns an ordinary camera into a stream of
typed, local events: where the user is looking, how confident the estimate is, fixations, blinks, and calibration
progress. Hosts decide what those events are allowed to control.

Fovea is early-stage. The Python engine and CLI, the typed event contract, webcam calibration, and the NDJSON process
boundary work today; desktop pointer control and gesture fusion are planned. The
[README](https://github.com/Wega-Labs/fovea#readme) tracks current status.

## Start here

| If you want to | Read |
| --- | --- |
| Understand coordinates, calibration, and confidence | [Concepts](concepts.md) |
| Embed Fovea in a Python program | [Python host guide](guides/python.md) |
| Drive Fovea from Node or Electron as a child process | [Node and Electron guide](guides/node-electron.md) |
| Implement a host in another language | [Wire protocol](protocol.md) and [Protocol messages](reference/protocol-messages.md) |
| Look up a class or setting | [Python API](reference/python-api.md) |
| Contribute, including recording fixtures | [Contributing](contributing.md) |

## Quick start

```bash
uv sync --extra dev
uv run python scripts/download_mediapipe_model.py
uv run fovea run --ndjson --calibrate
```

This opens the local webcam, shows calibration targets, and writes one JSON event per stdout line. Logs stay on stderr.
To try Fovea without a camera, run the [minimal Python example](guides/python.md#run-without-a-camera), which replays
a synthetic landmark fixture.

## Privacy defaults

Frames are processed locally and discarded after inference. Fovea performs no identity recognition, collects no
analytics by default, and requires the host to show a visible indicator while tracking is active. See
[PRIVACY.md](https://github.com/Wega-Labs/fovea/blob/main/PRIVACY.md) for the data boundary and host obligations.
