# Privacy and Safety

Fovea is a local input engine, not an analytics service. Its default runtime has
no telemetry, account, cloud inference, advertising, or network client. Camera
frames are processed in memory and discarded after landmark inference.

## Data boundary

During an ordinary run, Fovea reads frames from the selected local camera and
emits typed events to the local host process over stdout. It does not transmit
frames, landmarks, calibration data, gaze coordinates, target IDs, diagnostics,
or behavioral events over a network.

The only network-capable runtime source is the explicit pinned-model downloader
in `src/fovea/webcam/model.py`. It downloads one declared MediaPipe model and
verifies its SHA-256 before installation. Fovea never downloads a model as a
side effect of starting camera capture.

## Persistence

An ordinary run persists only the calibration record selected by the user or
host. That JSON file contains regression coefficients plus the calibration's
feature names, sample counts and quality labels, display/camera/frame identity,
and target layout. Schema-v4 records also contain the bounded feature vectors and
normalized coordinates from the calibration wizard, plus up to 200 trusted feature
vectors paired with host-confirmed click, tap, or dwell coordinates and weights.
Starting a new calibration clears the prior online observations before the new
wizard anchors are written. These records contain no image pixels and are not an
identity template, but they can still reveal device geometry and behavioral
characteristics; protect them as personal application data.

Target registration, unconfirmed gaze events, and health diagnostics are not
persisted by the core engine. Online observations are stored only after the host
explicitly confirms a coordinate through the `observe` control. The explicit
landmark-recording workflow is an exception: it
writes the contributor-selected destination and never includes pixels. Recorded
landmarks are biometric-adjacent data; contributors must record only themselves,
provide informed consent, and never include minors or third parties.

The guided benchmark is another explicit-output workflow. Its JSON report stores
derived errors, point coordinates, device metadata, and timing measurements but
no frames or landmarks. Review machine labels and output paths before publishing.

## Diagnostics and retention

Diagnostics are off by default. `--diagnostics` adds rate-limited health events
to the local NDJSON stream; it does not enable network access or raw-frame
capture. If a host retains those events in Fovea's diagnostics directory, the
CLI removes expired `.json`, `.jsonl`, and `.ndjson` artifacts before starting a
diagnostic session. The default is `--diagnostics-retention 24h`; hosts may set
an app-specific directory with `--diagnostics-dir`.

Managed filenames begin with `fovea-diagnostics-`. Cleanup is deliberately
non-recursive and ignores other names, unknown file types, and directories. A
host that copies diagnostics elsewhere owns the retention and deletion policy
for that copy.

## Prohibited uses and absent features

Fovea does not perform face recognition, identity matching, emotion inference,
demographic classification, covert recording, or analytics. Applications must
not repurpose its camera or gaze data for surveillance or identity decisions.

Fovea is not a medical device. Gaze estimates can be inaccurate, and target or
dwell events must not be the sole confirmation for destructive, financial,
security-sensitive, or safety-critical actions.

## Visible capture and user control

The protocol handshake declares `indicator_required: true`. Every host must show
a persistent, unambiguous indicator for the full time camera capture is active.
Hosts must also provide a non-gaze way to pause and stop capture, preserve an
alternative input path, and avoid designs that force prolonged fixation or
blinking.

## Threat model

Fovea's controls reduce these risks:

- Accidental network exfiltration: CI rejects networking imports outside the
  pinned model downloader.
- Supply-chain model replacement: the downloader uses a versioned URL and a
  hard-coded SHA-256; mismatches fail closed.
- Silent long-term retention: runtime data is ephemeral by default, calibration
  persistence is explicit, and opt-in diagnostics have bounded cleanup.
- Stale or cross-device calibration: calibration records include display,
  camera, and frame identity and are rejected on mismatch.
- Hidden capture: the wire contract requires a host-visible indicator and the
  process supports pause, quit, and clean camera release.

These controls cannot protect data after a host reads stdout, a local process
with permission to inspect Fovea's memory or files, a compromised Python/native
dependency, or an operating system with a compromised camera stack. Embedders
must apply platform permissions, least-privilege file access, dependency review,
and their own user-facing privacy notice.

Security and privacy vulnerabilities should be reported through the private
channel described in `SECURITY.md`. Do not attach faces, frames, or identifying
landmark recordings to a public issue.
