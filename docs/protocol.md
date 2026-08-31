# Fovea wire protocol

Fovea writes UTF-8 NDJSON to stdout. A successful stream starts with one `hello`
object and then emits one event per line. Hosts send control messages as one JSON
object per line on stdin. Logs belong on stderr.

Protocol `1.0` uses `display_normalized` coordinates: `(0, 0)` is the display's
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
