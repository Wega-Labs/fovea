# Fovea wire protocol

Fovea writes UTF-8 NDJSON to stdout. A successful stream starts with one `hello`
object and then emits one event per line. Hosts send control messages as one JSON
object per line on stdin. Logs belong on stderr.

Protocol `1.1` uses `display_normalized` coordinates: `(0, 0)` is the display's
top-left and `(1, 1)` is its bottom-right. Hosts must display a visible camera-use
indicator whenever capture is active, as declared by `indicator_required` in the
handshake.

The `hello.backend` field reports the landmark implementation selected with the
CLI `--backend` option. Hosts may record it for diagnostics and benchmark
comparability, but must not infer additional event fields from its value. The
currently shipped value is `mediapipe`.

## Compatibility

The checked-in schema is `schema/fovea-protocol-v1.json`. Additive event types,
fields, commands, and capability strings are backward-compatible protocol-minor
changes. Consumers must ignore unknown object fields and event types. Removing a
field, changing its meaning or type, or changing the coordinate space requires a
new protocol major version.

The `online_calibration` handshake capability means the protocol understands the
`observe` control and `calibration_updated` event. Capabilities describe protocol
support, not runtime state: launching Fovea with `--no-online` still advertises the
capability but makes observations no-ops.

Nullable `GazePoint.target_id`, `snapped_x`, and `snapped_y` preserve the raw gaze
coordinate while exposing target-aware intent. `GazeTestDone` carries expected and
predicted points so benchmark hosts do not need access to engine internals. Event
timestamps are nanoseconds; ordering follows the NDJSON stream.

## Controls

The legacy bare controls `calibrate`, `test`, `pause`, `resume`, and `quit` remain
supported. Their JSON forms use a `cmd` field. Calibration and testing can receive
five or more custom targets:

```json
{
  "cmd": "calibrate",
  "targets": [
    {"label": "top-left", "x": 0.1, "y": 0.1},
    {"label": "top-right", "x": 0.9, "y": 0.1},
    {"label": "center", "x": 0.5, "y": 0.5},
    {"label": "bottom-left", "x": 0.1, "y": 0.9},
    {"label": "bottom-right", "x": 0.9, "y": 0.9}
  ]
}
```

Coordinates must be finite and within `[0, 1]`.

Hosts may confirm a click, tap, or dwell-confirmed location to improve an existing
v4 calibration:

```json
{"cmd":"observe","x":0.42,"y":0.68,"weight":0.8,"timestamp_ns":1720000000000000000}
```

`x` and `y` are finite `display_normalized` coordinates. `weight` is optional,
defaults to `1.0`, and must be within `(0, 1]`. `timestamp_ns` is optional; when
present, Fovea associates the observation with the nearest eligible gaze feature
within 200 ms. Without it, Fovea uses the most recent eligible feature within the
same 200 ms frame-time window. Stale, blinking, lost-tracking, calibration-wizard,
disabled, and pre-v4 observations are ignored.

After a quarantined drift group passes the transaction guards and is installed,
Fovea emits one event per successful refit. `n` is the cumulative count of trusted
online observations and `loo_error` is a health measurement, not an acceptance gate:

```json
{"type":"calibration_updated","n":5,"loo_error":0.061,"timestamp_ns":1720000000100000000}
```

Use `--no-online` on capture commands to disable association, refitting, and online
state changes for that run.

Target-aware mode uses replace-all registration. Send an empty `items` array to
disable it, and resend the full set after a layout change (at no more than 10 Hz):

```json
{"cmd":"targets","items":[{"id":"save","x":0.72,"y":0.82,"w":0.2,"h":0.1}],"space":"display_normalized"}
```

Rectangle `x`/`y` identify the top-left corner; `w`/`h` must be positive and the
rectangle must fit within normalized display space. IDs must be non-empty and
unique within a replacement set.

## Schema workflow

`fovea schema` prints the schema derived from the event and command dataclasses.
Contributors changing the wire contract must update the committed schema in the
same change. CI checks that the generated and committed forms are byte-identical.
