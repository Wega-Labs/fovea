"""Live smoke: camera to landmarks to engine; print performance and pose metrics.

Pass ``--mirror`` / ``--no-mirror`` and wink one eye during the run to check the
``Wink.eye`` topology-label convention: ``left_only_closed`` counts frames where
only the eye MediaPipe labels "left" (mesh 263/362/473) read as closed.
"""

import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from fovea.webcam.camera import Webcam
from fovea.webcam.engine import GazeEngine, GazeSettings
from fovea.webcam.features import extract_features
from fovea.webcam.landmarks import FaceLandmarkEstimator

ARGS = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
N = int(ARGS[0]) if ARGS else 120
MIRROR = "--no-mirror" not in sys.argv
t = time.perf_counter()
estimator = FaceLandmarkEstimator()
load_ms = (time.perf_counter() - t) * 1e3
camera = Webcam(0, 640, 480, MIRROR)
t = time.perf_counter()
camera.connect()
open_ms = (time.perf_counter() - t) * 1e3
engine = GazeEngine(
    GazeSettings(calibration_path="/tmp/fovea_live_cal.json"),
    Path("/tmp"),
)
landmark_ms: list[float] = []
tracking: dict[str, int] = {}
pitches: list[float] = []
face_widths: list[float] = []
valid_points = 0
left_only_closed = 0
right_only_closed = 0
t0 = last = time.perf_counter()
frames = 0
try:
    for _ in range(N):
        frame = camera.read()
        if frame is None:
            continue
        frames += 1
        now = time.perf_counter()
        dt = now - last
        last = now
        started = time.perf_counter()
        observation = estimator.process(frame)
        landmark_ms.append((time.perf_counter() - started) * 1e3)
        height, width = frame.shape[:2]
        output = engine.process(
            None if observation is None else observation.landmarks,
            float(width),
            float(height),
            dt,
            0.0,
            blendshapes=None if observation is None else observation.blendshapes,
        )
        tracking[output.tracking] = tracking.get(output.tracking, 0) + 1
        if observation is not None:
            features = extract_features(
                observation.landmarks,
                float(width),
                float(height),
                0.16,
                0.12,
                35.0,
                observation.blendshapes,
            )
            pitches.append(features.pitch_deg)
            face_widths.append(features.face_width)
            left_closed = features.left.valid and features.left.ear < 0.16
            right_closed = features.right.valid and features.right.ear < 0.16
            left_only_closed += left_closed and features.right.valid and not right_closed
            right_only_closed += right_closed and features.left.valid and not left_closed
        if output.valid:
            valid_points += 1
finally:
    wall = time.perf_counter() - t0
    camera.disconnect()
    estimator.close()

landmark_ms.sort()
print(
    json.dumps(
        {
            "model_load_ms": round(load_ms),
            "camera_open_ms": round(open_ms),
            "frames": frames,
            "fps": round(frames / wall, 1),
            "landmark_ms_p50": (
                round(landmark_ms[len(landmark_ms) // 2], 1) if landmark_ms else None
            ),
            "landmark_ms_p95": (
                round(landmark_ms[int(0.95 * (len(landmark_ms) - 1))], 1) if landmark_ms else None
            ),
            "tracking": tracking,
            "valid_points": valid_points,
            "mirror": MIRROR,
            "left_only_closed": left_only_closed,
            "right_only_closed": right_only_closed,
            "pitch_median_deg": round(st.median(pitches), 1) if pitches else None,
            "face_width_median": (round(st.median(face_widths), 3) if face_widths else None),
        },
        indent=1,
    )
)
