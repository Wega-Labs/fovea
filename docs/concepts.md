# Concepts

## Coordinate space

Protocol `1.x` reports gaze in **display-normalized** coordinates: `x` and `y` are in `[0, 1]`, with `(0, 0)` at the
display's top-left and `(1, 1)` at its bottom-right. Hosts convert to their own pixels or window coordinates; the
TypeScript client includes helpers for that.

Targets that hosts register with the `targets` control use the same space (`"space": "display_normalized"`), as do
calibration targets and `observe` points.

## Calibration

A webcam gaze estimate needs a per-user, per-display calibration:

- **Per display.** A calibration is bound to the display, camera, and capture geometry it was made with. Fovea does not
  reuse it for a different display geometry; the host must not either.
- **Layout.** The default layout is `CALIBRATION_LAYOUT`. Hosts can supply five or more display-normalized targets with
  the `calibrate` and `test` controls. Fovea warns (`calibration_warning`) when the targets' bounding box covers less
  than 40% of either axis.
- **Cues.** During calibration Fovea emits `calibration_cue` events with the target's `x`/`y`. A host can render them
  itself (for example with `--no-display`) or let Fovea open its own fullscreen window.
- **Result.** `calibration_done` reports the number of points, coverage, and leave-one-out error.
- **Online refinement.** Hosts that know where the user was looking (for example, at a clicked button) can send
  `observe` to refine the model without re-running the wizard; accepted refits are reported as
  `calibration_updated`.

## Confidence and tracking state

Every `gaze_point` carries a `confidence`. When tracking degrades, Fovea emits `tracking_state` changes (`active`,
`uncertain`, `lost`) with an optional human-readable `detail`. The design rule is that Fovea freezes or disarms rather
than guessing: hosts should stop acting on gaze when tracking is not `active`.

## Targeting versus confirmation

Gaze expresses *where*, not *act*. Fovea separates the two so that natural looking does not trigger actions (the
"Midas touch" problem):

- **Point and focus.** `gaze_point`, `target_enter` / `target_leave`, and `fixation` describe attention.
- **Confirm.** `dwell_progress` / `dwell`, blinks, winks, or a host-side keyboard, switch, or voice trigger express
  intent.
- **Pause.** Hosts must always keep a way to disarm control.

Which events may control anything is always the host's decision.

## Events and versioning

Events are immutable, typed dataclasses in Python and one JSON object per line on the wire. Within protocol major `1`,
only additive changes are made: new events, optional fields, commands, and capabilities. Consumers must ignore unknown
fields and event types. See [Wire protocol](protocol.md) for the handshake and compatibility rules and
[Protocol messages](reference/protocol-messages.md) for every field.
