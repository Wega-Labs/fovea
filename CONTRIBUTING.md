# Contributing to Fovea

Welcome. Fovea turns local camera observations into a stable, platform-neutral gaze event
stream. The project treats privacy, accessibility, and an additive protocol boundary as product
requirements, not follow-up work.

## Before you start

For a focused bug fix, open or claim an issue and describe the behavior you intend to change.
For a new event, backend, dependency, or architecture change, discuss the design in an issue
before writing code. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) and
[Security Policy](SECURITY.md).

## Development setup

Fovea requires Python 3.12. With `uv`:

```bash
uv sync --extra dev
uv run python scripts/download_mediapipe_model.py
```

With the standard Python toolchain:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_mediapipe_model.py
```

The model download is pinned and SHA-256 verified. Do not commit model files, virtual
environments, calibration data, camera output, or local tool caches.

## Required gates

Run all four gates before each commit and again before opening a pull request:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

If you use the standard toolchain, invoke `ruff`, `mypy`, and `pytest` directly. Keep `uv.lock`
in sync whenever `pyproject.toml` changes. A PR should not weaken or skip a gate to pass CI.

## Tests without a camera

The default test suite must never need camera permission, a face, or a network connection.
Use the synthetic generator in `tests/synth.py` for exact pose, gaze, and blink inputs. Checked-in
landmark JSONL under `tests/fixtures/synthetic/` can be replayed through the production engine:

```bash
uv run fovea replay tests/fixtures/synthetic/frontal_30f.jsonl --ndjson
uv run pytest tests/test_replay.py
```

The real MediaPipe graph test uses a synthetic pixel array and the verified local model; it does
not open a camera.

## Recording a fixture

Record landmarks only with explicit, informed consent:

```bash
uv run fovea record --landmarks recording.jsonl --seconds 10
```

Only record yourself. Never submit a fixture of a minor or third party. A fixture must contain no
image pixels, video, names, or identifying metadata and must be at most 200 KB. Review the JSONL
before committing it and follow `tests/fixtures/README.md`.

## Adding or changing an event

The event contract only grows within a major protocol version. Add a new event or an optional
field; do not rename, remove, reorder semantically, or reinterpret an existing field. Update:

1. the frozen dataclass and `FoveaEvent` union in `src/fovea/events.py`;
2. public exports in `src/fovea/__init__.py`;
3. serialization and schema coverage for every event type;
4. the committed protocol schema once schema generation is available;
5. the README and changelog when host-visible behavior changes.

Regenerate committed schemas with the repository's schema command and confirm the diff contains
only the intended additive changes.

## Benchmarks

The guided live benchmark is still being built. Once `fovea bench` lands, run it only when a PR
changes live tracking, calibration, smoothing, or latency behavior, and attach its JSON report.
The benchmark requires a consenting maintainer with a camera; ordinary tests remain camera-free.

## Pull requests

Keep one issue per PR, use `Closes #N`, and preserve unrelated work in the checkout. A complete PR:

- explains the user-visible result and relevant tradeoffs;
- includes regression tests or fixtures where applicable;
- keeps the event contract additive;
- updates documentation and `CHANGELOG.md` for visible changes;
- passes all four gates on every supported CI platform;
- contains no telemetry, unapproved network calls, frames, credentials, or generated local data.

Maintainers aim to provide a first response within five working days. Complex research or
security-sensitive changes may take longer; the maintainer should post an update rather than
leave the contributor guessing.

## Where to ask

Use the issue connected to your work for implementation questions. Use the repository's
"Start here" tracking issue for contributor guidance. Report vulnerabilities privately as
described in [SECURITY.md](SECURITY.md), never in a public issue.
