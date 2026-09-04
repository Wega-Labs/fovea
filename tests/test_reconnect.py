from __future__ import annotations

import pytest

from fovea.webcam.camera import ReconnectPolicy


def test_read_failure_is_reported_once_after_continuous_grace() -> None:
    policy = ReconnectPolicy(lost_after_s=1.0)
    assert not policy.read_failed(10.0)
    assert not policy.read_failed(10.999)
    assert policy.read_failed(11.0)
    assert not policy.read_failed(12.0)
    policy.frame_ok()
    assert not policy.read_failed(20.0)
    assert policy.read_failed(21.0)


def test_zero_grace_reports_first_failed_read() -> None:
    policy = ReconnectPolicy(lost_after_s=0.0)
    assert policy.read_failed(0.0)
    assert not policy.read_failed(0.0)


def test_reconnect_backoff_is_capped_and_resets() -> None:
    policy = ReconnectPolicy(initial_delay_s=0.5, factor=2.0, max_delay_s=5.0)
    assert [policy.next_delay() for _ in range(7)] == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0]
    policy.reconnected()
    assert policy.next_delay() == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lost_after_s": -1.0},
        {"initial_delay_s": -1.0},
        {"factor": 0.5},
        {"max_delay_s": float("inf")},
    ],
)
def test_reconnect_policy_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ReconnectPolicy(**kwargs)
