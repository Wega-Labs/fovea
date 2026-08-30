```text
 _____ _____     _______    _
|  ___/ _ \ \   / / ____|  / \
| |_ | | | \ \ / /|  _|   / _ \
|  _|| |_| |\ V / | |___ / ___ \
|_|   \___/  \_/  |_____/_/   \_\
```

# Fovea

**Your gaze is an input.**

Fovea is a planned open-source gaze input engine that turns ordinary cameras into a new way to interact with computers. Look to point. Dwell or blink to select. Focus to highlight. Combine gaze with a deliberate gesture to drag, scroll, navigate, and control an interface without reaching for a mouse.

The first target is desktop control using a built-in webcam. The longer-term goal is a portable library that any Wega app—or any other application—can embed to add gaze as an input modality alongside touch, keyboard, pointer, gesture, and voice.

> [!NOTE]
> Fovea is currently a product vision and research plan. No tracking or control code has been implemented, and no code from the referenced projects has been copied into this repository.

## Why Fovea

Most software assumes that intent arrives through a keyboard, mouse, or touchscreen. But people already communicate attention continuously through their eyes. Fovea aims to turn that attention into a private, responsive, and reusable input signal using cameras people already own.

This is bigger than an eye-controlled mouse. Fovea should become:

- **A desktop controller** — point, click, highlight, drag, scroll, and switch focus with gaze-assisted interactions.
- **An accessibility tool** — offer a hands-free path for people who cannot comfortably use conventional input devices.
- **An input engine** — expose calibrated gaze, fixation, dwell, blink, confidence, and tracking-state events.
- **An embeddable SDK** — let apps add gaze-aware controls without implementing computer vision from scratch.
- **A multimodal building block** — use eyes to choose *what* and hands, voice, keyboard, switch, or touch to express *what to do with it*.
- **A local-first system** — process camera frames on the device by default, without requiring a cloud account.

## Interaction model

Gaze is excellent for expressing *where*, but an eye movement alone should not always mean *act*. Fovea should separate targeting from confirmation so natural looking does not produce accidental clicks—the classic “Midas touch” problem.

The initial interaction vocabulary should include:

- **Point** — map calibrated gaze to a stabilized screen position.
- **Focus** — visually highlight the element currently under sustained gaze.
- **Select** — confirm with dwell, blink, keyboard, switch, voice, or another explicit trigger.
- **Drag** — lock an object with a deliberate trigger, move it with gaze, and release explicitly.
- **Scroll** — use configurable gaze zones or gaze plus a modifier.
- **Pause** — instantly disarm control with a hotkey, gesture, voice command, or loss of tracking.

Every action should be configurable. No user should be forced to blink, dwell, or hold their eyes in a way that causes fatigue or excludes their particular movement patterns.

## Gaze + gesture

The most powerful Fovea interaction may be the combination of gaze and ordinary hand gestures captured by the same camera. Gaze provides fast, precise targeting; a deliberate hand gesture provides confirmation and manipulation.

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

The fusion layer should behave as an explicit state machine:

```text
idle → target primed → grabbed → manipulating → released
  └──────────────────── cancel / tracking lost ──────────┘
```

Fovea must show which target is primed before a gesture can affect it. Gestures should operate only on that target, require sufficient gaze and hand confidence, and stop immediately when either signal is lost. Natural hand movement must not become a command unless multimodal control is visibly armed.

## Product surfaces

Fovea should eventually ship as three layers built on the same engine:

1. **Fovea Controller** — a reference desktop application that can control the operating-system pointer.
2. **Fovea Engine** — the camera, landmark, calibration, gaze-estimation, filtering, and intent pipeline.
3. **Fovea SDK** — a small API and platform bindings that applications can embed directly.

On mobile, operating-system restrictions may limit global cursor control. The first mobile integration may therefore be an SDK for gaze-aware experiences inside participating apps, with broader control explored where platform accessibility APIs permit it.

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

The core boundary is intentional: estimation should describe what the eyes are doing, while platform and application adapters decide what an event is allowed to control.

An eventual SDK should expose concepts such as:

```text
GazePoint(x, y, confidence, timestamp)
Fixation(center, duration, confidence)
Blink(eye, duration)
Gesture(kind, phase, confidence)
Manipulation(target, delta, phase)
TrackingState(active | uncertain | lost)
```

These names and shapes are illustrative, not a committed API.

## Technical direction

The first research spike will likely use Python, OpenCV, and an on-device landmark backend such as MediaPipe because they make camera and calibration experiments fast. MediaPipe can provide face, eye, and iris landmarks; a separate calibrated estimator is still required to infer where on a screen someone is looking.

The implementation should preserve replaceable boundaries for:

- camera and frame capture
- face, eye, and iris landmark models
- hand landmark and gesture-recognition models
- gaze-estimation models
- calibration strategies
- temporal smoothing and fixation detection
- multimodal gaze, gesture, voice, and switch fusion
- operating-system control adapters
- language and framework bindings

A successful prototype should later be reduced to a portable core with a stable event API. Python can remain the research and reference layer while native or FFI-backed runtimes are evaluated for low-latency desktop and mobile embedding. No cloud backend should be required for core tracking.

## Accuracy and reliability

Fovea must measure more than whether the pointer moves. Evaluation should cover:

- calibration time and repeatability
- angular and on-screen point error
- end-to-end latency
- jitter while fixating
- drift over time
- head movement and posture changes
- glasses, contact lenses, partial occlusion, and lighting
- different cameras, screen sizes, and multi-monitor layouts
- false activations and recovery after tracking loss

The UI must communicate uncertainty. If confidence falls below a safe threshold, Fovea should freeze or disarm actions instead of guessing.

## Privacy and safety

An eye tracker observes a face continuously and can reveal sensitive behavioral signals. Fovea should adopt strict defaults:

- process video locally and discard frames immediately after inference
- never perform identity recognition
- never record or stream camera input without explicit, visible consent
- provide a persistent indicator whenever tracking is active
- make raw-frame diagnostics opt-in, time-limited, and easy to delete
- expose a universal pause and emergency-disable control
- require explicit confirmation before destructive or security-sensitive actions
- collect no analytics by default

Applications embedding Fovea should receive the minimum event data they need—not unrestricted camera access.

## Accessibility principles

Fovea is intended to expand access, but it should not claim to be medical-grade or universally usable without evidence. Eye movement, fatigue, vision, motor control, and camera positioning vary widely. Development should involve people with relevant access needs early, support alternative confirmation methods, and always preserve another way to pause or exit.

## Roadmap

- [ ] Validate real-time eye and iris landmarks from a standard webcam
- [ ] Define repeatable accuracy, latency, jitter, and failure benchmarks
- [ ] Build guided multi-point calibration
- [ ] Map gaze to a stabilized screen position with head-pose compensation
- [ ] Prototype dwell, blink, and modifier-based selection
- [ ] Validate real-time hand landmarks alongside eye tracking
- [ ] Prototype gaze-to-target plus pinch-to-drag interaction
- [ ] Define the multimodal gaze-and-gesture state machine
- [ ] Add a permissioned desktop pointer adapter
- [ ] Define the platform-neutral Fovea event API
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

GitHub does not currently detect a license for the two repositories suggested as starters, so Fovea will use them as conceptual references only unless their authors clarify reuse terms. Before importing any code, model, dataset, or asset, contributors must verify its license compatibility, preserve required attribution, and document its origin. Generally useful fixes should be contributed back upstream when appropriate.

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

## License

Licensed under the [Apache License 2.0](LICENSE).
