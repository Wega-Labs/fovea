# Fovea benchmark protocol

This procedure standardizes live reports across contributors. It requires a
camera and a person; CI covers report math only.

## Setup

1. Measure the visible display width and height in centimeters.
2. Record the machine model, exact camera, capture resolution, lighting, and
   whether glasses or contact lenses are worn.
3. Place distance markers at approximately 50, 60, and 75 cm from the display.
4. Use the same camera position, display scaling, room lighting, and chair for
   the complete run. Close applications that may contend for the camera.
5. Ensure the visible capture indicator is present and preserve a keyboard or
   switch method to stop the run.

## Command

```bash
fovea bench \
  --screen-width-cm 30.9 \
  --screen-height-cm 17.4 \
  --camera-name "Integrated 1080p camera" \
  --lighting "office indirect daylight" \
  --glasses "none" \
  --width 640 --height 480 \
  --output bench/results/<machine>-<date>.json
```

Create the output directory before running. Do not use `--yes` for a published
human run; each prompt is a checkpoint for posture and distance.

## Guided phases

The command performs these phases in order:

1. Calibration at about 60 cm.
2. Guided target tests at about 50, 60, and 75 cm.
3. A two-second neutral-head center fixation for jitter.
4. Center fixation with head yaw near -20° and +20° for tracking robustness.
5. Ten minutes of ordinary neutral viewing, followed by a 60 cm drift re-test.

The report includes point-level normalized and physical angular errors, jitter
around the median point, active-tracking rate during yaw, drift in median error,
and inference latency. Angular error uses the measured display dimensions and
the distance assigned to each phase.

## Publication checks

- Confirm all three distance phases contain the full target count.
- Confirm each yaw phase reached its requested angle; otherwise mark the run
  incomplete rather than treating it as robust.
- Check that the ten-minute drift duration was not shortened.
- Remove usernames and unintended machine identifiers from metadata or paths.
- Never attach images, video, or raw landmark recordings to a benchmark report.
- Add every valid run to `BENCHMARKS.md`, including results above the target.
