# Architecture

```text
Camera
  │ frames remain on device by default
  │
  ├──► face, eye, and iris landmarks
  │          ▼
  │    head-pose compensation + gaze estimator
  │          ▼
  │    per-user, per-screen calibration ──► gaze target
  │
  └──► hand landmarks + gesture recognizer ──► gesture state
                                                    │
                         gaze target + gesture state
                                                    ▼
Filtering, confidence, fixation, and intent engine
  │
  ├──► Fovea SDK events ──► application controls
  ├──► desktop adapter ───► pointer and accessibility APIs
  └──► local diagnostics ─► calibration and performance feedback
```

The core boundary is intentional: estimation describes what the eyes are doing, while platform and application
adapters decide what an event is allowed to control. Hand landmarks, gesture fusion, and the desktop adapter are
planned; see the [roadmap](https://github.com/Wega-Labs/fovea#roadmap).

## Two ways to host Fovea

| Boundary | Use it when | Entry point |
| --- | --- | --- |
| In-process Python | Your application is Python | `WebcamEventSource(...).events()` — see [Python host guide](guides/python.md) |
| Child process over NDJSON | Your application is in another language | `fovea run --ndjson` — see [Wire protocol](protocol.md) and [Node and Electron](guides/node-electron.md) |

Both run the same engine and emit the same events. The process boundary adds a versioned handshake, stdin controls,
and stable exit codes.

## Testing without a camera

Landmark fixtures record face landmarks, never pixels. `ReplayEventSource` feeds them through the same per-frame
processor as the webcam, so tests and examples exercise the real pipeline deterministically. See
[Contributing](contributing.md#fixtures).
