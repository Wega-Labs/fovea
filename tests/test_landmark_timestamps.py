from collections.abc import Iterator
from itertools import pairwise

from fovea.webcam.landmarks import next_video_timestamp_ms


def test_video_timestamps_stay_strictly_monotonic(monkeypatch) -> None:
    readings: Iterator[int] = iter(
        (
            1_000_000_000,
            1_000_000_000,
            900_000_000,
            1_004_000_000,
        )
    )
    monkeypatch.setattr("fovea.webcam.landmarks.time.monotonic_ns", lambda: next(readings))

    timestamps: list[int] = []
    previous = -1
    for _ in range(4):
        previous = next_video_timestamp_ms(previous)
        timestamps.append(previous)

    assert timestamps == [1000, 1001, 1002, 1004]
    assert all(later > earlier for earlier, later in pairwise(timestamps))
