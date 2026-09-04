# Fovea wire protocol

Fovea writes UTF-8 NDJSON to stdout. A successful stream starts with one `hello`
object and then emits one event per line. Hosts send control messages as one JSON
object per line on stdin. Logs belong on stderr.

The current protocol version is `1.3`. The `1.x` line uses `display_normalized`
coordinates: `(0, 0)` is the display's top-left and `(1, 1)` is its bottom-right.
Hosts must display a visible camera-use indicator whenever capture is active, as
declared by `indicator_required` in the handshake.

The `hello.backend` field reports the landmark implementation selected with the
CLI `--backend` option. Hosts may record it for diagnostics and benchmark
comparability, but must not infer additional event fields from its value. The
currently shipped value is `mediapipe`.

## Compatibility

The checked-in schema is `schema/fovea-protocol-v1.json`; the file name denotes
the protocol major. Additive event types, fields, commands, and capability strings
are backward-compatible protocol-minor changes. Consumers must ignore unknown
object fields and event types. Removing a field, changing its meaning or type, or
changing the coordinate space requires a new protocol major version.

Hosts written for protocol `1.0` remain compatible with `1.3`: every `1.0` field
keeps its meaning and type, the relative order of existing events within a frame
is unchanged, and new commands, event types, and optional fields are ignored by
contract. Clients should compare only the major component of `hello.protocol`.

The `online_calibration` handshake capability means the protocol understands the
`observe` control and `calibration_updated` event. Capabilities describe protocol
support, not runtime state: launching Fovea with `--no-online` still advertises the
capability but makes observations no-ops.

Nullable `GazePoint.target_id`, `snapped_x`, and `snapped_y` preserve the raw gaze
coordinate while exposing target-aware intent. `GazeTestDone` carries expected and
predicted points so benchmark hosts do not need access to engine internals. Event
timestamps are nanoseconds; ordering follows the NDJSON stream.

## Version history

- `1.3` — the `observe` control, `calibration_updated` event, and
  `online_calibration` capability for bounded online self-calibration.
- `1.2` — optional capture-to-emit latency on `gaze_point`, rolling latency
  percentiles and dropped-frame counts on `diagnostics`, and latest-frame capture.
- `1.1` — event vocabulary v2: the `saccade`, `wink`, `double_blink`, and
  `long_blink` event types; the optional boolean `gaze_point.pursuit` member; and
  the `saccade`, `pursuit`, `wink`, `double_blink`, and `long_blink` capability
  strings in `hello.capabilities`.
- `1.0` — the initial handshake, controls, JSON Schema, and the `gaze_point`,
  target, `fixation`, `blink`, tracking, calibration, gaze-test, and
  `diagnostics` events.

## Event vocabulary v2

All v2 detectors run on the same smoothed, display-normalized gaze stream that
produces `gaze_point`, so they behave identically for live capture and fixture
replay. Their thresholds live in `GazeSettings` (`saccade_velocity`,
`pursuit_velocity`, `pursuit_ms`, `pursuit_coherence`, `wink_min_ms`,
`wink_max_ms`, `double_blink_ms`, `long_blink_ms`, `long_blink_factor`, and
`natural_blink_window`); the defaults are heuristics validated on synthetic
streams, not guarantees.

Within one frame, events are ordered `calibration_updated`, `calibration_cue`,
`tracking_state`, `diagnostics` (when due), `calibration_done`, `gaze_test_done`,
`blink`, `long_blink` or `double_blink`, `wink`, `saccade`, `gaze_point`, target
events, then `fixation`. `calibration_updated` is present only when a successful
online refit has been queued since the preceding frame.

### `saccade`

A rapid gaze jump, reported once it lands. `from_x`/`from_y` are the last
sub-threshold point before the jump and `to_x`/`to_y` the landing point.
`amplitude` is the Euclidean distance between them in `display_normalized` units,
`duration_ms` spans onset to landing, and `timestamp_ns` is the landing time, so
the event follows the last in-flight `gaze_point` and precedes the landing
sample's `gaze_point`. Saccades are emitted only while tracking is `active`, so
they carry no `confidence`. A saccade is detected by velocity (I-VT) on the
smoothed gaze, so a jump that is heavily smoothed may land before the gaze has
fully settled; target transitions that occur during flight are reported as they
happen and are not retracted.

### `gaze_point.pursuit`

`true` while the gaze is tracking a smoothly moving target: the velocity has
stayed between `pursuit_velocity` and `saccade_velocity` for at least
`pursuit_ms` and the motion over that window is directionally coherent. Pursuit
never coincides with `fixation`, which is reset whenever the gaze is moving.
Target tracking continues during pursuit because the target may itself be moving.
Streams from engines older than `1.1` omit the member; treat a missing value as
`false`.

### `wink`

A deliberate single-eye closure while the other eye stayed open for between
`wink_min_ms` and `wink_max_ms`. `eye` is `"left"` or `"right"`, `duration_ms` is
the closure length, and `confidence` is the highest tracking confidence seen over
the closure and the reopening frame. A closure is rejected when the other eye also
closes (a natural blink starting asymmetrically), when either eye's landmarks
become invalid, or when it outlasts `wink_max_ms` (an occlusion).

`eye` is the landmark **topology label**, not the user's anatomical side: the
engine reports the eye that MediaPipe labels "left" (mesh indices 263/362/473)
as `"left"`. With the default mirrored capture (`mirror=True`) MediaPipe labels
by appearance, so the topology-left eye is likely the user's anatomical right.
This has not yet been verified against the real backend; hosts that need the
anatomical side should let the user pick which label triggers an action rather
than assuming one. If a swap proves necessary it will be applied where `mirror`
is known, not in the frame processor, and announced as a protocol change.

### `double_blink`

Two natural blinks in quick succession: the second blink closed within
`double_blink_ms` of the first reopening. `timestamp_ns` is the **second blink's
reopen time**. Three natural blinks in a row yield one `double_blink` with the
third left pending; four yield two. A `long_blink` clears any pending blink, so a
natural blink followed by a long blink and another natural blink never pairs.

### `long_blink`

A deliberate blink held at least as long as the current long-blink threshold;
`duration_ms` is the measured closure. The threshold starts at `long_blink_ms`
(600 ms by default, above ordinary spontaneous blinks) and, once at least five
natural blinks have been observed in the session, rises to
`long_blink_factor × median` of the most recent `natural_blink_window` natural
blink durations. Adaptation can only raise the threshold, long blinks are not
added to the learned distribution, and the distribution lives only for the
current stream.

### Trigger silence around the wizard

`wink`, `double_blink`, and `long_blink` are never emitted for a frame on which
the calibration or gaze-test wizard was active at frame start, and stay silent
after the wizard until both eyes have been seen open and valid on a later frame.
`blink` events are unaffected, so a blink that starts inside the wizard and
reopens afterwards is still reported as a `blink` but cannot become a trigger.

### Latency and capture

`gaze_point.latency_ms` is the number of milliseconds from the capture clock (a
monotonic clock sampled when the camera read returned) to the moment the frame's
complete event sequence is ready to be handed to the host. It includes the capture
hand-off, landmark inference, engine processing, and event assembly; it excludes
consumer pauses, NDJSON serialization, and pipe writes. It is `null` when unknown,
for example during replay. `diagnostics.latency_ms` keeps its inference-only meaning.

`diagnostics.latency_p50_ms` and `diagnostics.latency_p95_ms` are interpolated
percentiles of `gaze_point.latency_ms` over the most recent 90 emitted gaze points,
and `null` before the first one. `diagnostics.dropped_frames` is the cumulative number
of admitted camera frames discarded because a newer frame arrived before the previous
one was processed; it resets only when the source restarts, so hosts should diff it.

Fovea never queues more than one captured frame: a slow consumer always receives the
newest frame. The CLI `--fps N` option caps the processing rate by skipping camera
frames before inference; skipped frames are not counted as dropped. `timestamp_ns`
is unchanged: it remains the wall-clock time at capture.

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
Contributors changing the wire contract must update the committed schema and the
generated TypeScript types (`npm run generate --prefix packages/fovea-client`) in
the same change. CI checks that the generated and committed forms are
byte-identical.
