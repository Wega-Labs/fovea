# Plan 001 — Webcam engine hardening and an embeddable process boundary

**Status:** DRAFT (2026-08-31). Baseline `main` at `75a37be` (PR #1, "webcam gaze engine ported
from Silent Input"). Every `file:line` was verified against that commit; re-verify before editing.

**Why this plan exists.** PR #1 turned Fovea from an event schema into a real engine. Running it
live on Apple Silicon (macOS 26.2, built-in camera, 640×480) showed that the engine is fast and
the pipeline is sound, but three small defects stop anyone from actually using it, and the library
has no process boundary a host application can consume. This plan fixes those first, then closes
the gaps between what the README promises and what the code emits. The first consumer is
October's canvas (`october-desktop/plan/39-gaze-input-fovea.md`), which needs **F1–F4**.

---

## 0. Findings (measured 2026-08-31)

| # | Finding | Evidence |
|---|---|---|
| 1 | **Pinned `mediapipe>=1.0.1,<2` aborts at graph open on macOS arm64.** SIGABRT in `TensorsToDetectionsCalculator::Open` → `-[DrishtiMetalHelper initWithCalculatorContext:]` → `graph_service.h:139 Check failed: service_ Service is unavailable`. Reproduces with `delegate=CPU` and with synthetic frames (no camera). | `pyproject.toml:31`; `landmarks.py:69-78` builds the options |
| 2 | **`mediapipe==0.10.21` works.** Model load ≈ 1.0 s, camera open ≈ 2.1 s, 29.9 fps (camera-bound), landmarker 7.7 ms p50 / 8.2 ms p95, `GazeEngine.process` 0.4 ms, face on 117/120 frames, `eyeLook*` blendshapes present, uncalibrated points at ~30 Hz. 0.10.21 requires `numpy<2` and pulls `opencv-contrib-python` 4.x — so the numpy/opencv pins must move with it. | scratch venv, `pip show mediapipe` |
| 3 | **Calibration cannot complete at a normal sitting distance.** Measured face width 0.173 of frame vs `min_face_width = 0.18` → `"POOR: Face too far from camera"` on 57/57 face frames; `PointCollector.add` rejects POOR → zero samples ever accepted. | `engine.py:36`, `features.py:273-274`, `sampler.py:21-23` |
| 4 | **Head-pose pitch ≈ 162° while frontal** (median over 57 frames; yaw −18°, plausible). Consistent with the y-up `_FACE_3D` model (`features.py:28-38`) being solved against y-down image coordinates. Consequences: a ±180° wrap sits inside the normal head-pose range (discontinuity in a regression feature), and `uncalibrated_map` subtracts `0.10·pitch/30 ≈ 0.54` from y (`calibration.py:168`) — the first live points were pinned at `y = 0.0`. | `features.py:164-201`, `calibration.py:164-169` |
| 5 | **The test suite cannot see 1–4.** All 25 tests pass in 1.5 s because every test replaces `Webcam` and `FaceLandmarkEstimator` with fakes; CI is `ubuntu-latest` only. The real graph is never opened anywhere in CI. | `tests/test_gaze_pipeline.py:236-265`, `tests/test_event_source_close.py:11-47`, `.github/workflows/ci.yml:10` |
| 6 | **Contract gaps.** `Fixation` is exported (`__init__.py:7`) but never emitted; `dwell_ms`, `stability_ms`, `hysteresis` (`engine.py:29-31`) are unused; `Blink.duration_ms` is always `0.0` (`event_source.py:137`); landmark timestamps are synthetic `+= 33` (`landmarks.py:97`). | grep |
| 7 | **Paths assume a source checkout.** `DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models"` (`model.py:22`) resolves under `site-packages/../..` when installed as a wheel; calibration lives at `project_root / "data/gaze_calibration.json"` (`engine.py:41, 78-80`). | code |
| 8 | **No process boundary.** `WebcamEventSource.events()` is a blocking in-process iterator; no console script (`pyproject.toml` has no `[project.scripts]`). Any non-Python host (Electron, Swift, a browser) has nothing to spawn. | code |
| 9 | **Calibration is per display but the model does not know which display.** `CalibrationModel` stores coefficients, counts, quality, timestamp — not display size/id, camera index, or frame size (`calibration.py:53-61`). A multi-monitor user silently gets a wrong mapping. | code |

---

## 1. Fixes

Ordered by "unblocks the most". Each has an acceptance test; none changes the public event
contract except by **adding** optional fields or new event types (semver-minor).

### F1 — Dependency pins that work on macOS (blocking)

- `pyproject.toml:31-33` → `mediapipe>=0.10.21,<0.11`, `numpy>=1.26,<3`,
  `opencv-contrib-python>=4.10,<6`. Regenerate `uv.lock`.
- Keep the 1.0 crash tracked as **F10**; do not carry a `<2` range that no macOS user can install.
- **Acceptance:** new `tests/test_graph_open.py` constructs the real `FaceLandmarkEstimator` and
  runs one synthetic 640×480 frame; `pytest.skip` when the model file is absent, but CI (F6)
  always has it. This is the test that would have caught finding 1.

### F2 — Tracking gates that pass at a desk (blocking)

- `engine.py:36` `min_face_width`: default **0.12**, and scale the check by frame aspect so
  1280×720 and 640×480 behave alike (face width as a fraction of the *shorter* frame side, or
  express the gate in landmark-space pixels).
- `sampler.py:18-23`: reject only `LOST` and blink frames; count `POOR` samples but weight them
  0.5 in `quality_label`, or accept them once the collector has ≥ 6 `GOOD/FAIR` rows. The user
  must be able to finish the wizard; quality is reported, not enforced by starvation.
- Surface the reason: add optional `detail: str = ""` to `TrackingState` (additive) carrying
  `features.message` so hosts can say "move closer" instead of guessing.
- **Acceptance:** `test_gaze_pipeline.py` case with synthetic faces at width 0.15 completes all 10
  points; a case at 0.08 still reports `POOR` with `detail`.

### F3 — Head-pose convention (blocking)

- `features.py:164-201`: make a frontal face yield `pitch ≈ 0`. Either negate the y (and z) rows of
  `_FACE_3D` so the model is y-down like the image, or flip the image y before `solvePnP`; then
  wrap all three angles to `(-180, 180]` and re-check signs (nod down → positive pitch, turn right
  → positive yaw, defined once in the docstring).
- Re-derive the constants in `uncalibrated_map` (`calibration.py:164-169`) after the fix; they were
  tuned against the wrong sign.
- **Acceptance:** unit test projects `_FACE_3D` with the same camera matrix used in
  `estimate_head_pose` at 0°, ±15° pitch, ±20° yaw and asserts recovered angles within 2° with the
  documented signs; `uncalibrated_map` of the frontal projection ≈ `(0.5, 0.5)` ± 0.05.

### F4 — `fovea` CLI with NDJSON output (blocking for hosts)

- New `src/fovea/cli.py`; `[project.scripts] fovea = "fovea.cli:main"` and `python -m fovea`.
- `fovea run --ndjson [--camera N] [--width W --height H] [--calibrate] [--no-display]
  [--calibration-path P] [--model P] [--max-frames N]`: one JSON object per line on stdout,
  `{"type": "gaze_point", "x": …, "y": …, "confidence": …, "timestamp_ns": …}` — `type` is the
  snake_case class name; fields via `dataclasses.asdict`; `StrEnum` → string; flush per line;
  stderr for logs only.
- stdin control lines: `calibrate\n` (restart wizard), `test\n`, `pause\n`, `resume\n`, `quit\n`.
  `SIGTERM`/`SIGINT` → `close()` and exit 0. Non-zero exit with a one-line JSON `{"type":
  "error", "message": …}` for `CameraError` / `MediaPipeUnavailableError` / checksum errors.
- `fovea calibrate` and `fovea test` as thin aliases; `fovea doctor` prints versions, model path +
  checksum status, camera count, and (macOS) whether the process is authorized for video.
- Design: `main(argv, source_factory=None)` so tests inject a fake `EventSource`.
- **Acceptance:** `tests/test_cli.py` — serialization round-trip for every event type; a fake
  source produces the expected NDJSON lines; `quit` on stdin ends the loop; invalid `--camera`
  exits 2 with a JSON error line.

### F5 — Paths that work when installed

- Add `platformdirs`; model → `user_cache_dir("fovea")/models/face_landmarker.task`, calibration →
  `user_data_dir("fovea")/calibration/<key>.json`. Env overrides `FOVEA_MODEL_PATH`,
  `FOVEA_DATA_DIR`. `download_face_landmarker()` keeps its pinned URL + SHA-256 (`model.py:14-20`).
- `WebcamEventSource.project_root` (`event_source.py:39`) becomes optional and deprecated;
  `GazeSettings.calibration_path` (`engine.py:41`) accepts absolute paths only or `None`.
- **Acceptance:** tests run with `tmp_path` via the env overrides; no test writes into the repo.

### F6 — CI that opens the real graph on macOS

- `.github/workflows/ci.yml:10`: matrix `[ubuntu-latest, macos-latest]` (arm64 runners); cache the
  model by SHA-256; run the full suite including `test_graph_open.py` on both.
- **Acceptance:** green on both; a deliberate re-pin to `mediapipe>=1.0.1` must fail the macOS job.

### F7 — Emit what the contract promises

- `Fixation`: I-DT (dispersion-threshold) detector over the smoothed point using `stability_ms`
  as the window and `hysteresis` as the radius; emit at ≤ 10 Hz while held with growing
  `duration_ms`; end on dispersion break, blink, or tracking loss. `dwell_ms` becomes the
  documented threshold hosts may compare against (or emit a `Dwell` event — decide once).
- `Blink.duration_ms`: measure from consecutive blink frames; emit on the closing edge with the
  real duration (`event_source.py:134-141`).
- **Acceptance:** table-driven tests on synthetic point streams (hold / saccade / jitter / blink).

### F8 — Real, monotonic timestamps

- `landmarks.py:97`: replace `+= 33` with monotonic milliseconds, guarded `max(prev + 1, now)`
  (MediaPipe VIDEO mode requires strictly increasing timestamps).
- **Acceptance:** test with irregular `dt` asserts monotonic timestamps and no MediaPipe error.

### F9 — Calibration identity

- Store `display: {id?, width, height}`, `camera_index`, `frame: {w, h}` in the model JSON; bump
  `CALIBRATION_VERSION` 2 → 3 (`calibration.py:15`); `from_dict` accepts v2 as valid-but-unlabeled;
  `load_model(path, expect=…)` returns `None` on mismatch so hosts recalibrate instead of guessing.
- **Acceptance:** roundtrip + mismatch tests.

### F10 — MediaPipe 1.0 on macOS (timeboxed ½ day)

- Minimal repro is the synthetic-frame script from §0 finding 1. File upstream with the stack
  trace, versions, and the fact that `Delegate.CPU` does not avoid the Metal helper. Try: creating
  the graph on the main thread only, `running_mode=IMAGE`, and an explicit GPU-service opt-out if
  the 1.0 Python API exposes one. Outcome is documentation, not a blocker.

### F11 — Diagnostics (additive)

- New `Diagnostics(fps, latency_ms, face_width, yaw_deg, pitch_deg, timestamp_ns)` event at ≤ 2 Hz
  so hosts can show health without re-deriving it; off by default, `--diagnostics` on the CLI.

### F12 — Docs

- README quick start uses `fovea run --ndjson`; document the coordinate space (display-normalized,
  origin top-left, calibration is per display), the host contract (host may render
  `CalibrationCue` targets itself with `--no-display`), stdin control lines, exit codes, and the
  macOS note that camera permission is attributed to the **responsible app** of the spawning
  process (a child Python inherits the host app's grant).
- Move the roadmap ticks honestly: "dwell/blink selection" is unchecked until F7 ships.

---

## 2. Order and effort

| Step | Items | Effort |
|---|---|---|
| 1 | F1 + F6 (pins + real-graph CI) | ½ day |
| 2 | F2 + F3 (gates + pitch) | ½ day |
| 3 | F4 + F5 (CLI + paths) | 1 day |
| 4 | F7 + F8 + F9 (Fixation/Blink, timestamps, calibration identity) | 1 day |
| 5 | F10 (timebox) + F11 + F12 | ½ day |

**≈ 3–4 days.** Steps 1–3 unblock October (plan 39); steps 4–5 make the README true.

## 3. How to re-measure

```bash
uv sync --extra dev && python scripts/download_mediapipe_model.py
uv run fovea doctor
uv run fovea run --ndjson --max-frames 300 --diagnostics | tee /tmp/fovea.ndjson
# expect: model load ~1 s, ~30 fps, landmark p50 < 10 ms, tracking mostly "active",
#         and a frontal face reporting |pitch| < 10° in Diagnostics.
uv run fovea test            # after calibrating: mean / median / max normalized error
```

Benchmarks to record per machine (README "Accuracy and reliability"): calibration time, test
error at ~50 / 60 / 75 cm, fps, p50/p95 latency, jitter while fixating (std of the smoothed point
over 2 s), and recovery time after covering the camera.

## 4. Non-goals

Gesture fusion, the OS pointer adapter, and mobile bindings are untouched by this plan.
