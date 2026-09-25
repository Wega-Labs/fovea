# Contributing

The full guide is [CONTRIBUTING.md](https://github.com/Wega-Labs/fovea/blob/main/CONTRIBUTING.md). This page covers
the parts that are specific to documentation and fixtures.

## Required gates

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

## Fixtures

The default test suite never needs a camera, a face, or network access. Use the synthetic generator in `tests/synth.py`
for exact pose, gaze, and blink inputs, or replay checked-in landmark JSONL through the production engine:

```bash
uv run fovea replay tests/fixtures/synthetic/frontal_30f.jsonl --ndjson
```

Record a fixture only of yourself and with informed consent:

```bash
uv run fovea record --landmarks recording.jsonl --seconds 10
```

A fixture contains landmarks only: no pixels, video, names, or identifying metadata, and at most 200 KB.

## Documentation

The site is built with MkDocs Material from `docs/`:

```bash
uv sync --extra docs
uv run mkdocs serve
```

`docs/reference/protocol-messages.md` is generated from the JSON Schema. After changing an event or control message,
regenerate both files:

```bash
uv run python scripts/generate_protocol_schema.py
uv run python scripts/generate_protocol_reference.py
```

CI fails if either generated file is out of date, and builds the site with `mkdocs build --strict`.
