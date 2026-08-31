# Executor runbook 002 — Make Fovea contributor-ready, then infra-grade

**Audience:** an autonomous executor agent with shell access, a clone of `Wega-Labs/fovea`, and
`gh` authenticated with write access. Work top to bottom. Every step names its files, its tests,
its acceptance gate, and the GitHub issue it closes.

**Status:** DRAFT handoff (2026-08-31). **Baseline:** `main` at `75a37be`. All `file:line`
references were verified against that commit; re-verify before editing (line numbers move).

**Outcome you are building toward:**
1. **Works on clone** on macOS Apple Silicon, Linux, and Windows — the quick start in the README
   actually runs.
2. **Contributor-ready** — a stranger can clone, run tests without a camera, make a verifiable
   change, and open a PR that CI can judge.
3. **Infra-grade foundations** — a versioned wire protocol, target-aware interaction semantics,
   calibration identity, privacy artifacts, benchmarks, and the backend seam that keeps the
   project alive when dependencies break or platforms ship their own gaze APIs.
4. **Launched** — `v0.2.0` tagged with a changelog, starter issues labelled, README truthful.

Companion documents: `docs/plans/001-webcam-engine-hardening.md` (the findings and the F1–F12
fixes) · GitHub tracking issues **#6** (Plan 001) and **#7** (Roadmap 002) · issues **#2–#5**
(blockers) and **#8–#33** (roadmap items). Appendix C maps every step to its issue.

---

## 0. Read first

### 0.1 What Fovea is

An open-source gaze input engine: webcam → MediaPipe FaceLandmarker (478 landmarks incl. iris,
plus `eyeLook*` blendshapes) → a 10-feature vector (`src/fovea/webcam/features.py:81-96`) →
10-point ridge calibration (`calibration.py:36-47, :133-161`) → One Euro + EMA smoothing →
typed, immutable events (`src/fovea/events.py`) yielded by `WebcamEventSource.events()`
(`src/fovea/webcam/event_source.py`). ~1,700 lines, Python 3.12, `src/` layout, ruff + mypy
strict, pytest. README carries the product thesis (gaze = *where*; a separate deliberate trigger =
*act*; "Midas touch" is the enemy).

Repo map you will touch:

```
pyproject.toml                  deps (:30-33), mypy (:46-53), ruff (:55-67); no [project.scripts] yet
.github/workflows/ci.yml        ubuntu-only (:10); uv sync --frozen → lock must match pins
scripts/download_mediapipe_model.py
src/fovea/events.py             the public event contract (frozen dataclasses)
src/fovea/interfaces.py         EventSource / EventSink protocols
src/fovea/webcam/{camera,landmarks,features,calibration,calibration_view,sampler,smoothing,engine,event_source,model}.py
tests/                          25 tests; every one fakes Webcam + FaceLandmarkEstimator
models/                         gitignored *.task (pinned download); README.md documents the pin
docs/plans/001-*.md             findings + fixes (this runbook's parent)
```

### 0.2 Measured facts (2026-08-31, Apple Silicon, macOS 26.2, built-in camera 640×480)

| Fact | Number / evidence |
|---|---|
| Pinned `mediapipe>=1.0.1,<2` on macOS arm64 | **SIGABRT** at graph open: `TensorsToDetectionsCalculator::Open` → `DrishtiMetalHelper` "Service is unavailable". Reproduces with `Delegate.CPU` and with synthetic frames (no camera). Not a Python exception — cannot be caught. |
| `mediapipe==0.10.21` (pulls `opencv-contrib-python 4.11`, requires `numpy<2` → `1.26.4`) | Works: model load ≈ 1.0 s · camera open ≈ 2.1 s · **29.9 fps** (camera-bound) · landmarker **7.7 ms p50 / 8.2 ms p95** · `GazeEngine.process` 0.4 ms · face on 117/120 frames · blendshapes present · uncalibrated points at ~30 Hz, confidence ≈ 0.5 |
| Calibration at ~60 cm | Impossible: median `face_width` **0.173** < gate `min_face_width` **0.18** (`engine.py:36`) → `POOR: "Face too far from camera"` on 57/57 face frames (`features.py:273-274`) → `sampler.py:21-23` rejects every sample |
| Head pose while frontal | `pitch_deg` median **162.6°** (yaw −18.2°, plausible). Consistent with the y-up `_FACE_3D` model (`features.py:28-38`) solved against y-down image coords. `uncalibrated_map` subtracts `0.10·pitch/30 ≈ 0.54` from y (`calibration.py:168`) → first live points pinned at `y = 0.0` |
| Test suite | 25 pass in 1.5 s; none opens the real graph or a camera (`tests/test_gaze_pipeline.py:236-265`, `tests/test_event_source_close.py:11-47`); CI ubuntu-only (`ci.yml:10`) |
| Dependency footprint (0.10.21 + opencv + numpy + transitive jax/jaxlib/matplotlib/sounddevice) | **755 MB** site-packages |
| Camera permission on macOS | Attributed to the **responsible app** of the spawning process (a child Python inherited the host terminal app's grant). Standalone binaries need their own `NSCameraUsageDescription`. |

### 0.3 Standing decisions (do not re-litigate)

1. **Python stays** the research engine and reference implementation. No rewrite in Go/Rust/etc.
   in this runbook. Native/compiled engines come later, *behind the same protocol*.
2. **The wire protocol is the product boundary.** Hosts consume NDJSON from a spawned process;
   the engine language is invisible to them. Everything in Phase C serves that boundary.
3. **The public event contract only grows.** Add optional fields or new event types (semver-minor).
   Never rename or remove a field without a protocol major bump — and there is no major bump in
   this runbook.
4. **Privacy is a feature, not a doc.** No frames persist; no network calls except the pinned
   model download; no analytics; hosts get events, never pixels. CI enforces the network rule (C6).
5. **Accuracy strategy:** webcam gaze is a node-/target-level signal (~100–200 px after
   calibration). Value comes from interaction semantics (targets, dwell, fusion) and honest
   confidence — not from pretending to be a mouse.
6. **Tests must not need a camera or a face.** Landmark fixtures (B1) are the mechanism.

### 0.4 Working rules

- **One PR per issue**, branch `feat/<issue>-<slug>` or `fix/<issue>-<slug>`, body ends with
  `Closes #N`. Squash-merge is fine. Never force-push `main`. Never rewrite history that CI has
  seen.
- **Gate before every commit:** `ruff format --check .` · `ruff check .` · `mypy` · `pytest`.
  CI runs exactly these (`.github/workflows/ci.yml`). A red gate is a blocker, not a note.
- **Keep `uv.lock` in sync** with `pyproject.toml` (CI uses `uv sync --frozen`; a stale lock fails
  the install step, not the tests — read the log).
- **Never commit:** `models/*.task`, `data/`, `.venv/`, any fixture containing image pixels, any
  file with a real person's name or face, API keys, `.cursor/`. `.gitignore` already covers most;
  check `git status` before every commit.
- **Never add:** telemetry, network calls, a second language runtime, a new copyleft dependency,
  or a change to `LICENSE`.
- **README must stay truthful.** If you un-tick or tick a roadmap box, the code must match on the
  same PR.
- **Do not `@mention` people** in issues or PRs. Do not close issues you did not resolve.
- **Report as you go** (§5): a comment when you start an issue, a comment with the PR link when
  you finish, and tick the checkbox in #6 / #7.
- If a step is blocked (needs an owner decision, a secret, a platform you cannot test), do every
  other step, then write the blocker in the final report. Do not silently narrow scope.

### 0.5 Environment setup and the smoke check

```bash
cd fovea
# uv preferred; fall back to venv if uv is not on PATH
uv --version || python3.12 -m pip install --user uv
uv sync --extra dev            # after A1 this resolves; before A1 it installs the crashing pins
uv run python scripts/download_mediapipe_model.py   # pinned float16 rev 1, SHA-256 verified
uv run pytest                  # 25 passed (fakes) — proves nothing about the real graph yet
```

Smoke check for the real graph (no camera; this is the test that fails before A1 and passes
after):

```bash
uv run python - <<'PY'
import numpy as np, sys; sys.path.insert(0, "src")
from fovea.webcam.landmarks import FaceLandmarkEstimator
est = FaceLandmarkEstimator()
est.process(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
print("graph opened OK")
PY
```

Live check (optional; needs a camera and the terminal app's Camera permission): Appendix A.
If your environment has no camera, skip every "Live:" line in this runbook — nothing else depends
on them.

---

## 1. Phase A — Works on clone (blockers #2 #3 #4 #5 + CI). ≈ 2 days

Order matters: A1 first (nothing runs on macOS without it), A2 second (so it cannot regress).

### A1 — Dependency pins that work on macOS — closes **#2**

**Edit** `pyproject.toml:30-33`:

```toml
dependencies = [
  "mediapipe>=0.10.21,<0.11",
  "numpy>=1.26,<3",
  "opencv-contrib-python>=4.10,<6",
]
```

`mediapipe 0.10.21` requires `numpy<2` and depends on `opencv-contrib-python` 4.x; the three pins
move together. Then `uv lock` and commit the regenerated `uv.lock`.

**Add** `tests/test_graph_open.py`:

```python
"""Open the real MediaPipe graph on a synthetic frame. This is the test that catches #2."""

import numpy as np
import pytest

from fovea.webcam.model import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.is_file(), reason="run scripts/download_mediapipe_model.py"
)


def test_real_graph_opens_on_synthetic_frame() -> None:
    from fovea.webcam.landmarks import FaceLandmarkEstimator

    est = FaceLandmarkEstimator()
    try:
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        assert est.process(frame) is None  # noise has no face; the point is: no abort
    finally:
        est.close()
```

**Acceptance:** `uv run pytest tests/test_graph_open.py` passes on macOS arm64 (it aborted before).
`mypy`, `ruff` clean. **Live:** Appendix A prints ~30 fps and `landmark_ms_p50 < 10`.

**PR title:** `fix: pin mediapipe 0.10.x so the graph opens on macOS; add real-graph smoke test`.

### A2 — CI on macOS with the real model — closes Plan 001 **F6** (checkbox in #6)

**Edit** `.github/workflows/ci.yml`:

```yaml
jobs:
  check:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]   # macos-latest = arm64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { python-version: "3.12" }
      - name: Install dependencies
        run: uv sync --frozen --extra dev
      - name: Cache FaceLandmarker model
        uses: actions/cache@v4
        with:
          path: models/face_landmarker.task
          key: face-landmarker-64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
      - name: Download model (verified by SHA-256)
        run: uv run python scripts/download_mediapipe_model.py
      - run: uv run ruff format --check .
      - run: uv run ruff check .
      - run: uv run mypy
      - run: uv run pytest
```

**Acceptance:** both jobs green on the A1 branch. **Proof of the guard:** on a scratch branch,
revert the pin to `mediapipe>=1.0.1,<2`, push, watch the macOS job fail in `test_graph_open`,
delete the branch. Mention the run URL in the PR.

**PR title:** `ci: run the suite on macOS and open the real graph`.

### A3 — Tracking gates that pass at a desk — closes **#3**

**Edit** `src/fovea/webcam/engine.py:36`: `min_face_width: float = 0.12`.

**Edit** `src/fovea/webcam/features.py:273-274` — compute the gate against the *shorter* frame
side so 640×480 and 1280×720 agree: pass `image_w, image_h` into the comparison (they are already
parameters of `extract_features`) and compare `width * image_w / min(image_w, image_h)`… keep it
simple and documented; the intent is "a face ~12 % of the short side is fine".

**Edit** `src/fovea/webcam/sampler.py:18-23`: reject only `LOST` and blink. `POOR` rows are
accepted with weight 0.5 in `quality_label` (add a `weighted_count` property) **or** accepted
once ≥ 6 `GOOD/FAIR` rows exist — pick one, document it in the docstring. The user must be able to
finish the wizard.

**Add** optional `detail: str = ""` to `TrackingState` (`src/fovea/events.py:108-117`) — additive.
Populate it from `features.message` in `event_source.py:127-132`.

**Tests** (`tests/test_gaze_pipeline.py`): a synthetic face at width 0.15 completes all 10 points;
a face at width 0.08 reports `POOR` with a non-empty `detail`. Update `tests/test_events.py` for
the new field default.

**Acceptance:** tests green; **Live:** calibration completes at ~60 cm without leaning in.

**PR title:** `fix: let calibration complete at a normal sitting distance; surface tracking detail`.

### A4 — Head-pose sign convention — closes **#4**

**Edit** `src/fovea/webcam/features.py:164-201`:
- Make the 3D model y-down to match image coordinates: negate the y **and** z rows of `_FACE_3D`
  (or flip image y before `solvePnP` — choose one, comment why).
- After `cv2.Rodrigues`, wrap `yaw/pitch/roll` to `(-180, 180]`.
- Define the convention in the docstring: **nod down → positive pitch; turn to the camera's right
  → positive yaw; tilt clockwise (as seen by the camera) → positive roll.** Note that `mirror=True`
  in `Webcam` flips the image horizontally and therefore the yaw sign — state which frame the
  convention applies to (post-mirror, i.e. what the engine sees).
- Re-derive the constants in `uncalibrated_map` (`calibration.py:164-169`) after the fix: with a
  frontal face the map must return ≈ `(0.5, 0.5)`.

**Add** `tests/test_head_pose.py`: build a synthetic landmark list (length ≥ 292; only indices
`1, 152, 263, 33, 291, 61` matter) by rotating `_FACE_3D` with a known `(pitch, yaw, roll)`,
projecting with the same camera matrix `estimate_head_pose` builds (`fx = fy = w`, `cx = w/2`,
`cy = h/2`), normalizing to `[0,1]`. Cases: `(0,0,0)`, `(±15,0,0)`, `(0,±20,0)`, `(0,0,±10)`.
Assert recovered angles within 2° with the documented signs. Assert
`uncalibrated_map(frontal) ≈ (0.5, 0.5) ± 0.05` (blendshapes zero, iris centered).

**Acceptance:** tests green; **Live:** Appendix A prints `|pitch| < 10` for a frontal face.

**PR title:** `fix: head-pose pitch convention (frontal ≈ 0°); re-derive uncalibrated map`.

### A5 — `fovea` CLI with NDJSON output — closes **#5**

**Add** `src/fovea/serialize.py`: `to_json(event) -> str` — `{"type": <snake_case class name>,
**dataclasses.asdict(event)}` with `StrEnum` → `str`; `event_type_name(cls)`.

**Add** `src/fovea/cli.py` and `src/fovea/__main__.py`; **edit** `pyproject.toml`:
`[project.scripts] fovea = "fovea.cli:main"`.

Spec (from #5): `fovea run --ndjson [--camera N] [--width W --height H] [--calibrate]
[--no-display] [--calibration-path P] [--model P] [--max-frames N]`; one JSON object per line on
stdout, flushed per line; logs to stderr only; stdin control lines `calibrate`, `test`, `pause`,
`resume`, `quit`; `SIGTERM`/`SIGINT` → `close()` and exit 0; fatal errors → one
`{"type":"error","message":…}` line and exit **2** (usage/config), **3** (camera), **4** (model).
`fovea calibrate` / `fovea test` aliases. `fovea doctor` prints versions, model path + checksum
status, camera count, and on macOS the AVFoundation authorization status (use `objc`-free
approach: shell out to a tiny Swift snippet is *not* acceptable — read
`AVCaptureDevice.authorizationStatus` only if `pyobjc-framework-AVFoundation` is an optional extra;
otherwise print "unknown (install fovea-input[macos])").

Design: `main(argv, source_factory=None)`; stdin reader on a daemon thread posting to a
`queue.Queue`; the main loop drains control messages between frames.

**Tests** `tests/test_cli.py`: serialization round-trip for every event type in
`fovea.events`; a fake `EventSource` produces the expected NDJSON lines; `quit` on stdin ends the
loop; invalid `--camera` exits 2 with a JSON error line; `--max-frames 3` yields exactly the
fake's first three frames' events.

**Acceptance:** tests green; `uv run fovea run --ndjson --max-frames 30 | head` streams events
(Live). README quick start switched to the CLI (B4).

**PR title:** `feat: fovea CLI with NDJSON output, stdin control, and doctor`.

---

## 2. Phase B — Contributor-ready. ≈ 2–3 days

### B1 — Landmark fixtures + `fovea record` / `fovea replay` — closes **#17**

Fixture format (`*.jsonl`, one frame per line, **no pixels**):

```json
{"ts_ns": 1788170380123456789, "w": 640, "h": 480,
 "landmarks": [[x, y, z], …478 entries…],
 "blendshapes": {"eyelookinleft": 0.12, …},
 "transform": null}
```

- `fovea record --landmarks out.jsonl [--seconds N]` (Live only) writes it.
- `fovea replay in.jsonl --ndjson` runs `GazeEngine` over it exactly as `WebcamEventSource`
  would (factor the frame loop out of `event_source.py:77-151` so both paths share it).
- **Synthetic generator** `tests/synth.py`: projects `_FACE_3D` + a parametric iris/eyelid model
  to produce landmark lists for given `(pitch, yaw, roll, gaze_dx, gaze_dy, blink)`. Use it to
  write exact-answer fixtures; check them in under `tests/fixtures/synthetic/`.
- `tests/fixtures/README.md`: consent rules (contributor-recorded fixtures only of the
  contributor, no minors, no third parties), no pixels ever, ≤ 200 KB per file.
- Convert `test_gaze_pipeline.py` cases that currently hand-build landmarks to use the generator;
  add replay-based regression tests for features, head pose (A4), calibration fit, smoothing.

**Acceptance:** `uv run fovea replay tests/fixtures/synthetic/frontal_30f.jsonl --ndjson` emits
`tracking_state` + `gaze_point` lines deterministically (byte-identical across two runs and two
machines); CI runs the replay tests.

### B2 — Small correctness items that must precede a release — closes Plan 001 **F5 F8 F11** (checkboxes in #6; open one issue each first, see B3)

- **F5 paths:** add `platformdirs`; `DEFAULT_MODEL_PATH` → `user_cache_dir("fovea")/models/
  face_landmarker.task` (`model.py:22`), calibration → `user_data_dir("fovea")/calibration/
  default.json` (`engine.py:41, 78-80`); env overrides `FOVEA_MODEL_PATH`, `FOVEA_DATA_DIR`;
  `WebcamEventSource.project_root` optional + `DeprecationWarning`. `scripts/download_mediapipe_
  model.py` targets the new default. Tests use `tmp_path` via the env vars; **no test writes into
  the repo**.
- **F8 timestamps:** `landmarks.py:97` → monotonic milliseconds, `max(prev + 1, now_ms)`.
- **F11 diagnostics:** additive `Diagnostics(fps, latency_ms, face_width, yaw_deg, pitch_deg,
  timestamp_ns)` event at ≤ 2 Hz behind `--diagnostics`.

### B3 — Issue hygiene and starter labels

- Open standalone issues for **F5, F7, F8, F9, F10, F11, F12** (today they are only checkboxes in
  #6), each body = the paragraph from `docs/plans/001-webcam-engine-hardening.md` §1, each
  starting with `Part of #6`. Edit #6's task list to link them.
- Label `good first issue`: the F8 issue, the F12 issue, **#31** (TOML config half only — note it
  in a comment), **#33**. Label `help wanted`: **#12**, **#13**, **#23**, **#24**, **#26**, **#30**.
- Add a pinned issue "Start here" (or use #7) that links CONTRIBUTING and the starter labels.

### B4 — Community files (Appendix B has the skeletons)

- `CONTRIBUTING.md` — setup, the four gates, fixture workflow (B1), "how to add an event" (additive
  rule), PR expectations, review SLA (state a real one, e.g. "first response within 5 working
  days"), how to run `bench` (C8) when it exists.
- `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1 verbatim; contact address is an **owner
  decision** (put a placeholder, flag in the report).
- `SECURITY.md` — private reporting via GitHub Security Advisories; what counts (camera/data
  handling, model download integrity, RCE via control messages); response target.
- `.github/ISSUE_TEMPLATE/bug_report.yml` (asks for `fovea doctor` output), `feature_request.yml`,
  `config.yml` (blank issues off, links to #7 and CONTRIBUTING).
- `.github/PULL_REQUEST_TEMPLATE.md` — issue link, gates checklist, "event contract: additive?",
  "fixtures added/updated?".
- `.github/CODEOWNERS` — `* @harshsaver` as placeholder; **owner confirms** the maintainer list.
- `CHANGELOG.md` — keep-a-changelog; `## [Unreleased]` and `## [0.2.0]` (B7).

### B5 — README truthfulness pass

- Quick start uses `fovea run --ndjson`; the Python snippet stays as the library example.
- Status note: "early-stage; runs on macOS/Linux/Windows with `mediapipe 0.10.x`".
- Roadmap: tick only what shipped (after B2/C3: "dwell, blink" stays unticked until F7 lands).
- Add: CI badge, "Contributing" section (links), "Privacy" pointer to `PRIVACY.md` (C6).
- Document the coordinate space (display-normalized `[0,1]`, origin top-left, per-display
  calibration) and the macOS camera-attribution note.

### B6 — Commit the plans

`git add docs/plans/001-webcam-engine-hardening.md docs/plans/002-executor-runbook.md` in the
first PR you open (the runbook is the audit trail for everything after).

### B7 — Release `v0.2.0`

- `pyproject.toml` `version = "0.2.0"`, `src/fovea/__init__.py` `__version__`.
- `CHANGELOG.md` `[0.2.0]`: pins, CLI, fixtures/replay, gates, head pose, paths, diagnostics.
- Tag `v0.2.0`; GitHub Release with the changelog section.
- Add `.github/workflows/release.yml` that builds sdist + wheel on tag and publishes via PyPI
  **trusted publishing** (`pypa/gh-action-pypi-publish`, `id-token: write`). Creating the PyPI
  project and the trusted-publisher entry is an **owner action** — prepare the workflow, document
  the two clicks in the report, do not block on it.

**Phase B acceptance:** a fresh clone on macOS → `uv sync --extra dev && uv run python
scripts/download_mediapipe_model.py && uv run pytest` is green with **zero** camera access; a
contributor can change `features.py`, run the replay tests, and see them fail/pass; CONTRIBUTING,
CoC, SECURITY, templates, CODEOWNERS exist; ≥ 6 issues carry starter labels; `v0.2.0` exists.

---

## 3. Phase C — Infra-grade foundations. ≈ 2–3 weeks; each item is its own PR

Order is by dependency; C1 and B1 unblock almost everything.

### C1 — Versioned wire protocol — closes **#8**

- `hello` as the first NDJSON line: `{"type":"hello","protocol":"1.0","fovea":"<version>",
  "backend":"mediapipe","coordinate_space":"display_normalized","indicator_required":true,
  "capabilities":["calibration_cue","diagnostics"]}` — capabilities grow as C3–C5 land.
- `fovea schema` prints JSON Schema (draft 2020-12) for every event **and** control message,
  generated from the dataclasses + a `Command` union you introduce in `src/fovea/protocol.py`.
  Commit `schema/fovea-protocol-v1.json`; CI step regenerates and `git diff --exit-code`.
- JSON control messages `{"cmd":"calibrate"}` alongside the bare words from A5.
- `docs/protocol.md`: compatibility rules (additive = minor; clients ignore unknown fields).
- Test: every event round-trips through the schema; a client stub that ignores unknown fields
  still parses a stream with a synthetic extra field.

### C2 — Calibration identity — closes Plan 001 **F9**

`CalibrationModel` gains `display {id?, width, height}`, `camera_index`, `frame {w, h}`, and
`targets` (C4); `CALIBRATION_VERSION` 2 → 3 (`calibration.py:15`); `from_dict` accepts v2 as
valid-but-unlabeled; `load_model(path, expect=…)` returns `None` on mismatch and the CLI emits
`{"type":"warning","code":"calibration_mismatch"}`. Roundtrip + mismatch tests.

### C3 — Emit what the contract promises — closes Plan 001 **F7**

I-DT `Fixation` detector over the smoothed point (`stability_ms` window, `hysteresis` radius),
emitted at ≤ 10 Hz with growing `duration_ms`; real `Blink.duration_ms` from consecutive blink
frames (`event_source.py:134-141`). Table-driven tests on synthetic streams (hold / saccade /
jitter / blink). Tick the README roadmap box in the same PR.

### C4 — Windowed calibration with host-provided targets — closes **#10**

`{"cmd":"calibrate","targets":[{"label","x","y"},…]}`; ≥ 5 targets; coverage check (bounding box
≥ 40 % of the display per axis → else `warning`); targets stored with the model (C2);
`calibration_done {n_points, coverage, loo_error}`; `{"cmd":"test","targets":[…]}`. Tests: custom
5-target set completes and persists; 3 targets → `error`; low coverage → `warning`.

### C5 — Target-aware mode — closes **#11**

`{"cmd":"targets","items":[{"id","x","y","w","h"}],"space":"display_normalized"}` (replace-all,
≤ 10 Hz). Point-in-rect with confidence-scaled `expand`; nearest-target magnetism within
`snap_radius`; hysteresis; events `target_enter`, `target_leave`, `dwell_progress` (≤ 15 Hz),
`dwell` at `dwell_ms`; `gaze_point` gains optional `target_id`, `snapped_x`, `snapped_y`; frozen
while `uncertain/lost`. Table-driven tests around two adjacent rects; a jittery stream inside one
rect yields exactly one `dwell`. This is the item that turns noisy points into usable intent —
give it the most care.

### C6 — Privacy & safety artifacts — closes **#27**

`PRIVACY.md` (threat model: nothing leaves the process; only calibration coefficients persist;
diagnostics opt-in, `--diagnostics-retention 24h`, auto-delete; no identity recognition; no
analytics), `SECURITY.md` (if not done in B4), `hello.indicator_required: true` documented, and a
CI lint: `grep -rn "urllib\|requests\|http" src/` must match only `model.py`. Test for retention
deletion.

### C7 — Benchmark suite — closes **#18**

`fovea bench` (Live): guided calibrate → test at ~50/60/75 cm, 2 s fixation jitter, 10-minute
drift re-test, yaw ±20° robustness, latency; JSON report; `BENCHMARKS.md` + `bench/PROTOCOL.md`.
You may not be able to run it (no camera) — build it, unit-test the report math on fixtures, and
flag the live run as an owner/maintainer action.

### C8 — Backend seam — closes **#15** (first half)

Introduce `LandmarkBackend` protocol (`open(model)`, `process(frame_rgb, ts)`, `close()`), move
today's MediaPipe code behind it, add `--backend` and report it in `hello`/`doctor`. The `litert`
and `onnx` backends are follow-ups; do **not** start them in this runbook. Parity test = replay
fixtures through the backend interface.

### C9 — `@wega-labs/fovea-client` — closes **#9** (can run in parallel with C4–C8)

`packages/fovea-client` (TypeScript, zero deps): spawn, `hello` check, NDJSON parse with a buffer
cap, exit-code mapping, optional reconnect, coalescing, `toClientPoint(event, displayBounds,
contentBounds)`, `runCalibration({targets?, onCue})`; types generated from `schema/*.json`;
`examples/electron-host` (~150 lines). Tests against a fake child process. Publish is an owner
action (npm org).

### C10 — Not in this runbook (open a design comment on the issue instead)

Standalone binaries (#14), `litert`/`onnx` backends (#15 second half), CoreML (#16), gesture
fusion (#25), pointer adapter (#26), `fovea-web` (#29), iOS (#30), self-calibration (#21),
pursuit calibration (#22), robustness (#23), event vocabulary v2 (#24), API ergonomics (#31),
docs site (#33). Each needs a short design comment before code; leave them for the next runbook.

**Phase C acceptance:** `fovea run --ndjson` starts with `hello`; `schema/` is generated and
CI-checked; a host can send `targets` and receive `dwell`; calibration knows its display;
`Fixation` events exist; `PRIVACY.md` is enforced by CI; `bench` exists; the MediaPipe code sits
behind `LandmarkBackend`; the TS client passes its tests.

---

## 4. Phase D — Launch checklist (owner + executor)

- [ ] Fresh-clone test on macOS **and** Linux by someone who is not you (owner or a second
      machine): quick start works without touching a camera.
- [ ] README claims match code; roadmap ticks honest.
- [ ] `v0.2.0` (or `v0.3.0` if Phase C shipped) tagged, changelog, release notes.
- [ ] Starter issues labelled; #7 pinned; CONTRIBUTING linked from README.
- [ ] `PRIVACY.md` and `SECURITY.md` present; CoC contact filled in (owner).
- [ ] PyPI trusted publisher configured (owner) and the release workflow has run once.
- [ ] A 60-second demo (screen recording of `fovea run --ndjson` + the Electron example) linked
      from the README — the owner records it; the executor prepares the example.
- [ ] Announcement draft (owner): the thesis (gaze = *where*, trigger = *act*), what works
      today (numbers from §0.2 / `BENCHMARKS.md`), what is wanted (starter labels).

---

## 5. Reporting

- **Per issue:** comment "Starting — branch `<name>`" when you begin; comment the PR link when
  the PR is up; tick the checkbox in #6 or #7 when merged.
- **Per phase:** a comment on #7 summarizing what merged, what is blocked, and the numbers
  (test count, CI matrix status, live measurements if any).
- **Final report:** append `## Execution report <date>` to this file with: PRs merged (links),
  issues closed, items blocked (with the exact owner action needed — PyPI project, npm org, CoC
  contact, CODEOWNERS, live bench run, macOS Gatekeeper test), measurements after A1–A4 (Appendix
  A output), and anything you learned that contradicts §0.2 (update §0.2 rather than leaving it
  wrong).

Never mark a step done that you did not verify with its acceptance command.

---

## Appendix A — Live smoke script

Requires a camera and Camera permission for the terminal/host app. Save as
`scripts/live_check.py` (commit it; it is the manual counterpart of `test_graph_open.py`).

```python
"""Live smoke: camera → landmarks → engine for N frames; prints fps, latency, tracking, pose."""

import json, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, "src")
from fovea.webcam.camera import Webcam
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.features import extract_features
from fovea.webcam.landmarks import FaceLandmarkEstimator

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
t = time.perf_counter()
est = FaceLandmarkEstimator()
load_ms = (time.perf_counter() - t) * 1e3
cam = Webcam(0, 640, 480, True)
t = time.perf_counter()
cam.connect()
open_ms = (time.perf_counter() - t) * 1e3
eng = GazeEngine(GazeSettings(calibration_path=str(Path("/tmp/fovea_live_cal.json"))), Path("/tmp"))
lm, track, pitch, fw, pts = [], {}, [], [], 0
t0 = last = time.perf_counter()
n = 0
try:
    for _ in range(N):
        f = cam.read()
        if f is None:
            continue
        n += 1
        now = time.perf_counter()
        dt = now - last
        last = now
        a = time.perf_counter()
        obs = est.process(f)
        lm.append((time.perf_counter() - a) * 1e3)
        h, w = f.shape[:2]
        out = eng.process(
            None if obs is None else obs.landmarks,
            float(w),
            float(h),
            dt,
            0.0,
            blendshapes=None if obs is None else obs.blendshapes,
        )
        track[out.tracking] = track.get(out.tracking, 0) + 1
        if obs is not None:
            ft = extract_features(
                obs.landmarks, float(w), float(h), 0.16, 0.12, 35.0, obs.blendshapes
            )
            pitch.append(ft.pitch_deg)
            fw.append(ft.face_width)
        if out.valid:
            pts += 1
finally:
    wall = time.perf_counter() - t0
    cam.disconnect()
    est.close()
lm.sort()
print(
    json.dumps(
        {
            "model_load_ms": round(load_ms),
            "camera_open_ms": round(open_ms),
            "frames": n,
            "fps": round(n / wall, 1),
            "landmark_ms_p50": round(lm[len(lm) // 2], 1) if lm else None,
            "landmark_ms_p95": round(lm[int(0.95 * (len(lm) - 1))], 1) if lm else None,
            "tracking": track,
            "valid_points": pts,
            "pitch_median_deg": round(st.median(pitch), 1) if pitch else None,
            "face_width_median": round(st.median(fw), 3) if fw else None,
        },
        indent=1,
    )
)
```

Targets after A1–A4 on Apple Silicon: `fps ≈ 30`, `landmark_ms_p50 < 10`, `tracking` mostly
`GOOD`, `|pitch_median_deg| < 10`.

## Appendix B — File skeletons

**`CONTRIBUTING.md`** (sections): Welcome + thesis in two lines · Setup (uv and venv paths; model
download) · The four gates · Tests without a camera (fixtures, `synth.py`, `fovea replay`) · Adding
an event (additive rule, schema regen) · Recording a fixture (consent, no pixels, size) · PR
checklist · Review expectations · Where to ask.

**`.github/ISSUE_TEMPLATE/bug_report.yml`** fields: what happened · expected · `fovea doctor`
output (textarea, required) · OS / camera / lighting / glasses · steps · fixture attached? (checkbox).

**`.github/ISSUE_TEMPLATE/feature_request.yml`** fields: problem · proposal · which layer (engine /
protocol / CLI / client / docs) · additive to the event contract? (dropdown).

**`.github/PULL_REQUEST_TEMPLATE.md`**: `Closes #` · what changed · gates run (4 checkboxes) ·
event contract additive? · fixtures added/updated? · README/CHANGELOG updated?

**`SECURITY.md`**: report privately via GitHub Security Advisories; in scope: camera/data handling,
model download integrity, control-message parsing, anything that could exfiltrate frames; target
first response 5 working days.

## Appendix C — Step → issue map

| Step | Issue(s) | Closes |
|---|---|---|
| A1 | #2 | #2 |
| A2 | #6 (F6) | checkbox |
| A3 | #3 | #3 |
| A4 | #4 | #4 |
| A5 | #5 | #5 |
| B1 | #17 | #17 |
| B2 | #6 (F5 F8 F11 → new issues in B3) | 3 issues |
| B3 | #6 #7 labels | — |
| B4 | #7 (community files) | part of #27 / #33 |
| B5 | README | — |
| B7 | #32 (partial: changelog, tag, publish workflow) | comment on #32 |
| C1 | #8 | #8 |
| C2 | #6 (F9) | new issue |
| C3 | #6 (F7) | new issue |
| C4 | #10 | #10 |
| C5 | #11 | #11 |
| C6 | #27 | #27 |
| C7 | #18 | #18 (code); live run = owner |
| C8 | #15 | comment (first half) |
| C9 | #9 | #9 (publish = owner) |

## Execution report 2026-08-31

### Merged pull requests and closed issues

- [#34](https://github.com/Wega-Labs/fovea/pull/34) pinned the compatible MediaPipe
  line and added real-graph coverage; issue #2 is closed.
- [#35](https://github.com/Wega-Labs/fovea/pull/35) added the macOS CI leg and model
  cache. The deliberately failing guard proof was reviewed separately in closed,
  unmerged PR [#36](https://github.com/Wega-Labs/fovea/pull/36).
- [#37](https://github.com/Wega-Labs/fovea/pull/37) fixed desk-distance calibration
  gates and exposed tracking detail; issue #3 is closed.
- [#38](https://github.com/Wega-Labs/fovea/pull/38) fixed the head-pose convention;
  issue #4 is closed.
- [#39](https://github.com/Wega-Labs/fovea/pull/39) added the NDJSON CLI, controls,
  doctor, and stable exit codes; issue #5 is closed.

No later issue is reported closed. At the owner's direction, the remaining work
was first committed and pushed without local acceptance commands. Draft PRs were
then opened as an explicitly dependency-ordered review stack; none is represented
as verified, merge-ready, merged, or closed:

| Area | Draft PR | Branch | Tip when prepared |
|---|---|---|---|
| Landmark record/replay | #41 | `feat/17-landmark-record-replay` | `640672e` |
| Platform-native paths | #42 | `fix/platform-paths` | `e0e2ded` |
| Landmark timestamps | #43 | `fix/monotonic-landmark-timestamps` | `04c26f7` |
| Community health files | #44 | `docs/community-health-files` | `4e150fc` |
| README truthfulness | #45 | `docs/readme-truthfulness` | `95bbc06` |
| Diagnostics events | #46 | `feat/diagnostics-events` | `faa6866` |
| Protocol and schema | #47 | `feat/8-versioned-wire-protocol` | `2fcb49c` |
| Calibration identity | #48 | `feat/calibration-identity` | `c82326d` |
| Fixation and blink | #49 | `feat/fixation-blink-events` | `f02fc91` |
| Windowed calibration | #50 | `feat/10-windowed-calibration` | `c557477` |
| Target-aware mode | #51 | `feat/11-target-aware-mode` | `e96cd39` |
| Privacy enforcement | #52 | `feat/27-privacy-safety-artifacts` | `6ca58a4` |
| Benchmark suite | #53 | `feat/18-benchmark-suite` | `f8c735f` |
| Backend seam (first half) | #54 | `feat/15-landmark-backend-seam` | `dc11dc3` |
| TypeScript client | #55 | `feat/9-typescript-client` | `36af048` |

The C1–C9 branches are dependency-stacked. Review and merge them in their
dependency order; the B-series branches are independent where their histories
show that they branch directly from `main`.

Duplicate CLI PR #40 was closed as superseded by merged PR #39. `main` branch
protection now requires the current Ubuntu and macOS CI contexts, one approving
review, dismissal of stale reviews, and resolved conversations. It applies to
administrators, rejects force-pushes and deletion, and restricts pushes to the
`harshsaver` account.

### Verification and measurements

The executor did not run the deferred branch test suites, type checkers, package
builds, generated-file commands, live benchmark, or Appendix A camera smoke test
after the owner's instruction to leave execution for later. Consequently there
is no new test count, live FPS/latency/pitch measurement, three-machine benchmark,
or Electron end-to-end result to report. The branches contain camera-free tests
and CI definitions, but their acceptance boxes must remain unticked until those
commands actually pass.

No new evidence was collected that contradicts the measured facts in §0.2. This
is an absence of measurement, not a revalidation of those numbers.

### Remaining owner and release actions

- Run the per-branch acceptance commands and merge the dependency stack; close
  issues only through the corresponding verified PRs.
- Create/configure the PyPI project and trusted publisher, and create the npm
  organization/access needed for `@wega-labs/fovea-client`.
- Supply the Code of Conduct contact address and confirm the intended CODEOWNERS.
- Run and publish the live benchmark on three real machine/camera combinations.
- Have a second person perform fresh-clone macOS and Linux checks, including the
  macOS camera permission and Gatekeeper path.
- Record the 60-second engine plus Electron demo.
- Only after those checks, update the changelog and create the appropriate
  release tag; no tag or package publication was performed in this execution.
