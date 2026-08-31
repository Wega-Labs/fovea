# Privacy

Fovea is local-first gaze infrastructure. Its default camera and gaze pipeline does not send
frames, landmarks, gaze events, calibration data, or diagnostics to Wega Labs or any third party.
Fovea includes no analytics or telemetry client.

## Data flow

Live camera pixels are held in process memory long enough to run local landmark inference and
are then discarded. Applications consume typed gaze and tracking events, not unrestricted camera
frames. Fovea does not perform identity or emotion recognition.

The engine may persist calibration coefficients, sample counts, quality labels, and a creation
timestamp at the configured calibration path. These values describe a gaze mapping; image pixels
and landmark frames are not part of the calibration file. Delete that file to remove the saved
calibration.

The optional landmark recorder requires an explicit command and writes normalized landmarks and
blendshape scores only. It never writes camera pixels. Landmark recordings can still describe
behavior and must be treated as sensitive data; consent and fixture rules live in
`tests/fixtures/README.md`.

## Network behavior

The runtime engine and CLI do not require a network connection. The model download script makes
an explicit request to the pinned MediaPipe asset URL and verifies the file with a committed
SHA-256 digest before use. No camera or calibration data is included in that request.

## Diagnostics and hosts

Diagnostics are opt-in event data and are not persisted by Fovea. Embedding applications decide
whether to retain events and must disclose any behavior that differs from Fovea's defaults.
Hosts must show a persistent indicator while tracking is active, provide an immediate pause/stop
control, and avoid using gaze as sole confirmation for destructive or security-sensitive actions.

## Reports

Report a suspected privacy or security vulnerability privately according to
[SECURITY.md](SECURITY.md). Do not attach faces, frames, or identifying landmark recordings to a
public issue.
