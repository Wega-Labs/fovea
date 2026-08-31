```text
 _____ _____     _______    _
|  ___/ _ \ \   / / ____|  / \
| |_ | | | \ \ / /|  _|   / _ \
|  _|| |_| |\ V / | |___ / ___ \
|_|   \___/  \_/  |_____/_/   \_\
```

# Fovea

**Your gaze is an input.**

[![CI](https://github.com/Wega-Labs/fovea/actions/workflows/ci.yml/badge.svg)](https://github.com/Wega-Labs/fovea/actions/workflows/ci.yml)

Fovea is an open-source gaze input engine that turns ordinary cameras into a new way to interact with computers. Look to point. Dwell or blink to select. Focus to highlight. Combine gaze with a deliberate gesture to drag, scroll, navigate, and control an interface without reaching for a mouse.

The first target is desktop control using a built-in webcam. The longer-term goal is a portable library that any Wega app—or any other application—can embed to add gaze as an input modality alongside touch, keyboard, pointer, gesture, and voice.

> [!NOTE]
> Fovea is early-stage. The Python engine and CLI run on macOS, Linux, and Windows with the
> pinned MediaPipe 0.10.x line. The typed event contract, webcam gaze engine, calibration, and
> NDJSON process boundary are implemented. Desktop pointer control, fixation/dwell emission,
> target awareness, and multimodal gesture fusion are still planned.

## Why Fovea

Most software assumes that intent arrives through a keyboard, mouse, or touchscreen. But people already communicate attention continuously through their eyes. Fovea turns that attention into a private, responsive, and reusable input signal using cameras people already own.

This is bigger than an eye-controlled mouse. Fovea is designed as:

- **A desktop controller** — point, click, highlight, drag, scroll, and switch focus with gaze-assisted interactions.
- **An accessibility tool** — offer a hands-free path for people who cannot comfortably use conventional input devices.
- **An input engine** — expose calibrated gaze, fixation, dwell, blink, confidence, and tracking-state events.
- **An embeddable SDK** — let apps add gaze-aware controls without implementing computer vision from scratch.
- **A multimodal building block** — use eyes to choose *what* and hands, voice, keyboard, switch, or touch to express *what to do with it*.
- **A local-first system** — process camera frames on the device by default, without requiring a cloud account.

## Interaction model

Gaze is excellent for expressing *where*, but an eye movement alone does not always mean *act*. Fovea separates targeting from confirmation so natural looking does not produce accidental clicks—the classic “Midas touch” problem.

The interaction vocabulary includes:

- **Point** — map calibrated gaze to a stabilized screen position.
- **Focus** — visually highlight the element currently under sustained gaze.
- **Select** — confirm with dwell, blink, keyboard, switch, voice, or another explicit trigger.
- **Drag** — lock an object with a deliberate trigger, move it with gaze, and release explicitly.
- **Scroll** — use configurable gaze zones or gaze plus a modifier.
- **Pause** — instantly disarm control with a hotkey, gesture, voice command, or loss of tracking.

Every action is configurable by design. No user is forced to blink, dwell, or hold their eyes in a way that causes fatigue or excludes their particular movement patterns.

## Gaze + gesture

A core Fovea interaction combines gaze and ordinary hand gestures captured by the same camera. Gaze provides fast, precise targeting; a deliberate hand gesture provides confirmation and manipulation.

For example:

1. Look at a window, object, or text region to prime it as the target.
2. Pinch your fingers to grab the visible target.
3. Move your hand to drag it while your eyes are free to look ahead.
4. Open your fingers to release it—or show an open palm to cancel.

Other possible combinations include:

- look at a window and sweep a hand to move it between displays
- look at an image and pinch or rotate to resize it
- look at a document and move a hand vertically to scroll
- look at text, pinch to begin selection, and move a hand to extend the highlight
- look at a control and use a small gesture to adjust its value
- use gaze to target while voice names the action and a gesture confirms it

The fusion layer follows an explicit state machine:

```text
idle → target primed → grabbed → manipulating → released
  └──────────────────── cancel / tracking lost ──────────┘
```

Fovea shows which target is primed before a gesture can affect it. Gestures operate only on that target, require sufficient gaze and hand confidence, and stop immediately when either signal is lost. Natural hand movement does not become a command unless multimodal control is visibly armed.

## Product surfaces

Fovea is organized as three layers built on the same engine:

1. **Fovea Controller** — a reference desktop application that can control the operating-system pointer.
2. **Fovea Engine** — the camera, landmark, calibration, gaze-estimation, filtering, and intent pipeline.
3. **Fovea SDK** — a small API and platform bindings that applications can embed directly.

On mobile, operating-system restrictions limit global cursor control. Fovea therefore begins with an SDK for gaze-aware experiences inside participating apps and adds broader control where platform accessibility APIs permit it.

## Architecture

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

The core boundary is intentional: estimation describes what the eyes are doing, while platform and application adapters decide what an event is allowed to control.

The scaffold exposes an initial platform-neutral event model:

```text
GazePoint(x, y, confidence, timestamp, target_id?, snapped_x?, snapped_y?)
TargetEnter(id) / TargetLeave(id)
DwellProgress(id, progress) / Dwell(id)
Fixation(center, duration, confidence)
Blink(eye, duration)
Gesture(kind, phase, confidence)
Manipulation(target, delta, phase)
TrackingState(active | uncertain | lost)
CalibrationCue(label, x, y, index, total, instruction)
CalibrationWarning(message, coverage) / CalibrationDone(n_points, coverage, loo_error)
GazeTestDone(n_points, median_error, points)
```

These immutable, typed events form the first public library boundary. Semantic versioning governs their evolution.

## Tech stack

Fovea uses a library-first Python stack for fast vision research and a clean path into other applications.

| Layer | Choice |
| --- | --- |
| Runtime | Python 3.12 |
| Packaging | `pyproject.toml`, `src/` layout, and `uv` or standard `venv`/`pip` |
| Camera and frames | OpenCV |
| Face, eye, and iris landmarks | MediaPipe Tasks |
| Calibration, transforms, and filtering | NumPy |
| Library boundary | Frozen dataclasses, enums, and typed protocols |
| Desktop control | Native platform adapters behind the Fovea event API |
| Tests | pytest |
| Quality | Ruff and mypy in strict mode |
| Cloud | None required |

MediaPipe provides on-device face, eye, iris, and hand landmarks. A separate calibrated estimator maps those observations to a screen position; iris landmarks alone do not reveal where someone is looking.

Landmark inference crosses the public `LandmarkBackend` protocol: backends open a
model, accept RGB frames with caller-owned timestamps, and return backend-neutral
observations. MediaPipe is the only implementation currently shipped; LiteRT and
ONNX adapters remain follow-up work rather than advertised runtime options.

The architecture keeps replaceable boundaries for:

- camera and frame capture
- face, eye, and iris landmark models
- hand landmark and gesture-recognition models
- gaze-estimation models
- calibration strategies
- temporal smoothing and fixation detection
- multimodal gaze, gesture, voice, and switch fusion
- operating-system control adapters
- language and framework bindings

Python is the research implementation and reference SDK. The stable event API keeps camera, model, and operating-system details out of consuming applications, while native or FFI-backed runtimes can be added for lower-latency desktop and mobile embedding. Core tracking requires no cloud backend.

## Development

With `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

With the standard Python toolchain:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff format --check .
ruff check .
mypy
```

Current package layout:

```text
src/fovea/
├── benchmark.py        # guided live benchmark and report math
├── cli.py              # NDJSON command-line process boundary
├── events.py           # immutable gaze, blink, gesture, and tracking events
├── interfaces.py       # event producer and consumer protocols
├── serialize.py        # stable event-to-JSON serialization
├── util.py             # shared helpers (ScreenPoint, clamp01)
├── webcam/
│   ├── backend.py      # LandmarkBackend protocol + MediaPipe implementation
│   ├── camera.py       # OpenCV webcam capture
│   ├── landmarks.py    # BGR camera compatibility session
│   ├── features.py     # iris + head-pose gaze features
│   ├── calibration.py  # 10-point ridge calibration
│   ├── calibration_view.py  # on-screen calibration targets
│   ├── model.py        # pinned FaceLandmarker URL + SHA-256
│   ├── smoothing.py    # One Euro + EMA filters
│   ├── engine.py       # gaze pipeline core
│   └── event_source.py # WebcamEventSource (EventSource implementation)
└── py.typed            # typed-package marker
scripts/
└── download_mediapipe_model.py
tests/
├── test_events.py
├── test_gaze_pipeline.py
└── test_mediapipe_model.py
```

### Webcam engine quick start

```bash
uv sync --extra dev
python scripts/download_mediapipe_model.py
uv run fovea run --ndjson --calibrate
```

The command opens the local webcam, shows the calibration targets, and writes one compact event
object per stdout line. Logs and human-readable status stay on stderr.

For an in-process Python integration, use the same engine through `WebcamEventSource`:

```python
from pathlib import Path
from fovea import GazeSettings, WebcamEventSource

source = WebcamEventSource(
    GazeSettings(),
    Path("."),
    max_frames=600,
    show_calibration=True,
)
for event in source.events():
    print(type(event).__name__, event)
```

Download the pinned MediaPipe FaceLandmarker (float16 revision 1) before running
live webcam mode. The script verifies SHA-256 and refuses a mismatched file.

Live commands accept `--backend mediapipe`; the selected backend is repeated in
the first `hello` line and by `fovea doctor --backend mediapipe`. The explicit
flag makes backend selection stable for hosts while keeping unsupported adapters
out of the CLI choices.

```bash
python scripts/download_mediapipe_model.py
```

See `models/README.md` for the pinned URL and checksum. The model is cached in the platform's
user cache directory, and calibration is saved in the platform's user data directory. Set
`FOVEA_MODEL_PATH` or `FOVEA_DATA_DIR` to override those locations.

During calibration, `WebcamEventSource` emits `CalibrationCue` events whose
`x`/`y` match the active target layout. The default is `CALIBRATION_LAYOUT`;
NDJSON hosts can supply five or more display-normalized targets with the
`calibrate` and `test` control messages. Fovea warns when their bounding box
covers less than 40% of either display axis, stores calibration targets with
the model, and emits `CalibrationDone` with coverage and leave-one-out error.
Pass `show_calibration=True` to open a fullscreen window that draws the same
targets. The window closes automatically when calibration finishes.

### CLI and host contract

`GazePoint.x` and `GazePoint.y` use display-normalized coordinates in `[0, 1]`, with `(0, 0)` at
the display's top-left and `(1, 1)` at its bottom-right. Calibration is per display; a host must
not reuse coefficients for a different display geometry.

An embedding host can use `--no-display` and render every `CalibrationCue` itself at the cue's
normalized `x`/`y`. The host remains responsible for a visible camera/tracking indicator and for
deciding whether an event may control anything.

While `fovea run` is active, send one control word per stdin line:

- `calibrate` restarts calibration;
- `test` starts the calibrated gaze test;
- `pause` suppresses stdout events without stopping local capture;
- `resume` resumes stdout events;
- `quit` closes the source and exits cleanly.

The process exits `0` on normal completion or signal-driven shutdown, `2` for usage or
configuration errors, `3` for camera errors, and `4` for model/runtime errors. Fatal errors are a
single JSON object on stdout so non-Python hosts can parse them consistently.

On macOS, camera permission is attributed to the responsible app that launches Fovea. A Python
child process spawned by a terminal or desktop host uses that host application's camera grant;
the permission prompt may therefore name the terminal or host rather than `fovea`.

Hosts can also enable target-aware intent by replacing the current UI rectangles:

```json
{"cmd":"targets","items":[{"id":"save","x":0.72,"y":0.82,"w":0.2,"h":0.1}],"space":"display_normalized"}
```

Fovea keeps raw `GazePoint.x`/`y`, adds the selected `target_id` and snapped
target-center coordinates, and emits `TargetEnter`, `TargetLeave`,
`DwellProgress`, and one `Dwell` per continuous hold. Selection and dwell timing
freeze while tracking is uncertain or lost.

Live performance is measured with the guided `fovea bench` command. It records
accuracy at 50/60/75 cm, two-second fixation jitter, yaw robustness, ten-minute
drift, and inference latency in a JSON report. Use the comparable-run procedure
in [bench/PROTOCOL.md](bench/PROTOCOL.md); verified results belong in
[BENCHMARKS.md](BENCHMARKS.md).

### Privacy-safe landmark fixtures

Record normalized face landmarks and blendshape scores without retaining camera pixels:

```bash
uv run fovea record --landmarks recording.jsonl --seconds 10
```

Replay the same landmark frames through the webcam engine's shared processing path without a
camera or model:

```bash
uv run fovea replay tests/fixtures/synthetic/frontal_30f.jsonl --ndjson
```

The fixture format and consent requirements are documented in
[`tests/fixtures/README.md`](tests/fixtures/README.md). Fovea never writes image pixels in record
mode.

## Accuracy and reliability

Fovea measures more than whether the pointer moves. Evaluation covers:

- calibration time and repeatability
- angular and on-screen point error
- end-to-end latency
- jitter while fixating
- drift over time
- head movement and posture changes
- glasses, contact lenses, partial occlusion, and lighting
- different cameras, screen sizes, and multi-monitor layouts
- false activations and recovery after tracking loss

The UI communicates uncertainty. If confidence falls below a safe threshold, Fovea freezes or disarms actions instead of guessing.

## Privacy and safety

An eye tracker observes a face continuously and can reveal sensitive behavioral signals. Fovea adopts strict defaults:

- process video locally and discard frames immediately after inference
- never perform identity recognition
- never record or stream camera input without explicit, visible consent
- provide a persistent indicator whenever tracking is active
- make raw-frame diagnostics opt-in, time-limited, and easy to delete
- expose a universal pause and emergency-disable control
- require explicit confirmation before destructive or security-sensitive actions
- collect no analytics by default

Applications embedding Fovea receive the minimum event data they need—not unrestricted camera
access.

See [PRIVACY.md](PRIVACY.md) for the data boundary, threat model, diagnostic
retention policy, and host obligations. Security reports use the private channel
in [SECURITY.md](SECURITY.md).

## Accessibility principles

Fovea expands access without claiming to be medical-grade or universally usable without evidence. Eye movement, fatigue, vision, motor control, and camera positioning vary widely. Development involves people with relevant access needs early, supports alternative confirmation methods, and always preserves another way to pause or exit.

## Roadmap

- [x] Define the platform-neutral Fovea event API
- [x] Validate real-time eye and iris landmarks from a standard webcam (initial engine)
- [x] Build guided multi-point calibration (initial 10-point wizard)
- [x] Map gaze to a stabilized screen position with head-pose compensation
- [x] Define repeatable accuracy, latency, jitter, and failure benchmarks
- [ ] Prototype dwell, blink, and modifier-based selection
- [ ] Validate real-time hand landmarks alongside eye tracking
- [ ] Prototype gaze-to-target plus pinch-to-drag interaction
- [ ] Define the multimodal gaze-and-gesture state machine
- [ ] Add a permissioned desktop pointer adapter
- [ ] Package the engine as an embeddable library
- [ ] Explore mobile SDK and accessibility integrations
- [ ] Test with diverse users, cameras, lighting, eyewear, and displays

## References and prior art

The following projects are useful references for experiments and architecture:

- [pallab2o/Eye-controlled-mouse](https://github.com/pallab2o/Eye-controlled-mouse) — a compact OpenCV demonstration using a built-in or phone IP camera.
- [Ileriayo/computer_pointer_controller](https://github.com/Ileriayo/computer_pointer_controller) — a staged OpenVINO pipeline covering face detection, head pose, eye landmarks, gaze estimation, and pointer control.
- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) — Apache-2.0, cross-platform on-device vision building blocks and face/iris landmarks.
- [antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking) — a Python eye-tracking library with a simple integration surface.
- [brownhci/WebGazer](https://github.com/brownhci/WebGazer) — browser-based webcam gaze estimation and calibration research.

The first webcam gaze engine adapts work from Wega Labs' internal Silent Input
research prototype and is released here under the Apache License 2.0.

GitHub does not currently detect a license for the two repositories suggested as starters, so Fovea uses them as conceptual references only unless their authors clarify reuse terms. Before importing any code, model, dataset, or asset, contributors must verify its license compatibility, preserve required attribution, and document its origin. Contributors send generally useful fixes back upstream when appropriate.

## Contributing

Fovea is in the research and design stage. Early contributions are especially useful around:

- gaze-estimation and calibration research
- camera and landmark backends
- signal filtering, fixation, dwell, and blink detection
- hand tracking, gesture recognition, and multimodal fusion
- desktop accessibility and pointer APIs
- portable SDK and FFI design
- privacy-preserving evaluation datasets
- accessible interaction design and user testing

Open an issue with a focused proposal before starting a large implementation.
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, required gates, camera-free fixtures, and pull
request expectations. Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
