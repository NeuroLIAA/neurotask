import math

import pytest

from neurotask.tmt.metrics.sampling_period import (
    SamplingPeriodCalculator,
    compute_sampling_period_metrics,
)
from neurotask.tmt.model.tmt_model import TMTSubject, TMTTrial
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject, prefix: str = None) -> dict:
    """
    Compute sampling period metrics using SamplingPeriodCalculator.
    """
    calculator = SamplingPeriodCalculator(prefix=prefix)
    return calculator.add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=[],
        calculate_crosses=False,
        speed_threshold=None,
        consecutive_points=None,
    )


# =============================================================================
# Tests for normal sampling periods
# =============================================================================

def test_uniform_sampling_returns_constant_period():
    """
    Uniform sampling (constant time intervals) should have:
    - mean_sampling_period == all individual periods
    - std_sampling_period == 0
    """
    # Samples at t=0, 10, 20, 30 ms -> intervals of 10 ms
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 10.0),
        (2.0, 0.0, 20.0),
        (3.0, 0.0, 30.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_sampling_period"] == pytest.approx(10.0)
    assert metrics["median_sampling_period"] == pytest.approx(10.0)
    assert metrics["std_sampling_period"] == pytest.approx(0.0)
    assert metrics["min_sampling_period"] == pytest.approx(10.0)
    assert metrics["max_sampling_period"] == pytest.approx(10.0)


def test_variable_sampling_computes_statistics_correctly():
    """
    Variable sampling periods should compute correct statistics.
    """
    # Intervals: 10, 20, 30 ms
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 10.0),   # interval = 10
        (2.0, 0.0, 30.0),   # interval = 20
        (3.0, 0.0, 60.0),   # interval = 30
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_sampling_period"] == pytest.approx(20.0)
    assert metrics["median_sampling_period"] == pytest.approx(20.0)
    assert metrics["min_sampling_period"] == pytest.approx(10.0)
    assert metrics["max_sampling_period"] == pytest.approx(30.0)
    assert metrics["sample_count"] == 4
    assert metrics["valid_interval_count"] == 3


def test_returns_all_seven_metrics():
    """
    SamplingPeriodCalculator should return all 7 expected metrics.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 10.0),
        (2.0, 0.0, 20.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    expected_keys = [
        "mean_sampling_period",
        "median_sampling_period",
        "std_sampling_period",
        "min_sampling_period",
        "max_sampling_period",
        "sample_count",
        "valid_interval_count",
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_with_prefix_adds_prefix_to_all_keys():
    """
    When initialized with a prefix, all metric keys should have that prefix.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 10.0),
        (2.0, 0.0, 20.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject, prefix="non_cut_")

    # Check that prefixed keys exist
    assert "non_cut_mean_sampling_period" in metrics
    assert "non_cut_median_sampling_period" in metrics
    assert "non_cut_sample_count" in metrics

    # Check that non-prefixed keys do not exist
    assert "mean_sampling_period" not in metrics
    assert "sample_count" not in metrics


# =============================================================================
# Tests for edge cases
# =============================================================================

def test_empty_cursor_trail_returns_nan():
    """
    Empty cursor trail should return NaN for all statistical metrics.
    """
    cursor_trail = []
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert math.isnan(metrics["mean_sampling_period"])
    assert math.isnan(metrics["median_sampling_period"])
    assert math.isnan(metrics["std_sampling_period"])
    assert math.isnan(metrics["min_sampling_period"])
    assert math.isnan(metrics["max_sampling_period"])
    assert metrics["sample_count"] == 0
    assert metrics["valid_interval_count"] == 0


def test_single_point_returns_nan():
    """
    Single point cursor trail (no intervals) should return NaN for statistics.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert math.isnan(metrics["mean_sampling_period"])
    assert math.isnan(metrics["median_sampling_period"])
    assert metrics["sample_count"] == 1
    assert metrics["valid_interval_count"] == 0


def test_two_points_returns_single_interval():
    """
    Two points should return a single interval as all statistics.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 16.0),  # 16 ms interval (typical ~60 Hz)
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_sampling_period"] == pytest.approx(16.0)
    assert metrics["median_sampling_period"] == pytest.approx(16.0)
    assert metrics["std_sampling_period"] == pytest.approx(0.0)
    assert metrics["min_sampling_period"] == pytest.approx(16.0)
    assert metrics["max_sampling_period"] == pytest.approx(16.0)
    assert metrics["sample_count"] == 2
    assert metrics["valid_interval_count"] == 1


def test_duplicate_timestamps_are_filtered():
    """
    Duplicate timestamps (dt=0) should be filtered out.
    """
    # Two points at same time, then valid interval
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),   # duplicate timestamp (dt=0, filtered)
        (2.0, 0.0, 10.0),  # valid interval from t=0 to t=10
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Only one valid interval (10 ms)
    assert metrics["mean_sampling_period"] == pytest.approx(10.0)
    assert metrics["sample_count"] == 3
    assert metrics["valid_interval_count"] == 1


def test_all_duplicate_timestamps_returns_nan():
    """
    When all timestamps are the same (all intervals = 0), should return NaN.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 5.0),
        (1.0, 0.0, 5.0),
        (2.0, 0.0, 5.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert math.isnan(metrics["mean_sampling_period"])
    assert metrics["sample_count"] == 3
    assert metrics["valid_interval_count"] == 0


def test_negative_time_intervals_are_filtered():
    """
    Negative time intervals (time going backwards) should be filtered out.
    """
    # Time goes backwards in the middle
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 10.0),  # valid interval = 10
        (2.0, 0.0, 5.0),   # negative interval (filtered)
        (3.0, 0.0, 20.0),  # valid interval = 15 (from t=5 to t=20)
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Two valid intervals: 10 and 15 ms
    assert metrics["mean_sampling_period"] == pytest.approx(12.5)
    assert metrics["sample_count"] == 4
    assert metrics["valid_interval_count"] == 2


# =============================================================================
# Tests for typical refresh rates
# =============================================================================

def test_typical_60hz_sampling():
    """
    Typical 60 Hz sampling (~16.67 ms period) should be computed correctly.
    """
    # Simulate 60 Hz sampling
    interval_ms = 1000.0 / 60.0  # ~16.67 ms
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, interval_ms),
        (2.0, 0.0, interval_ms * 2),
        (3.0, 0.0, interval_ms * 3),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_sampling_period"] == pytest.approx(interval_ms)
    assert metrics["std_sampling_period"] == pytest.approx(0.0)


def test_typical_120hz_sampling():
    """
    Typical 120 Hz sampling (~8.33 ms period) should be computed correctly.
    """
    # Simulate 120 Hz sampling
    interval_ms = 1000.0 / 120.0  # ~8.33 ms
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, interval_ms),
        (2.0, 0.0, interval_ms * 2),
        (3.0, 0.0, interval_ms * 3),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_sampling_period"] == pytest.approx(interval_ms)


# =============================================================================
# Tests for custom start
# =============================================================================

def test_with_custom_start_uses_trail_from_start():
    """
    When trial has custom start, only cursor trail after start should be used.
    """
    from neurotask.tmt.model.tmt_model import CursorInfo, Coordinate

    # Full trail: 5 points
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 100.0),  # 100 ms interval (before start)
        (2.0, 0.0, 110.0),  # start point
        (3.0, 0.0, 120.0),  # 10 ms interval (after start)
        (4.0, 0.0, 130.0),  # 10 ms interval (after start)
    ])

    start = CursorInfo(Coordinate(2.0, 0.0), 110.0)
    trial, subject = build_trial_and_subject(
        cursor_trail,
        with_custom_start=True,
        start=start
    )

    metrics = _compute_metrics(trial, subject)

    # Should only compute from start point onwards (intervals: 10, 10)
    assert metrics["mean_sampling_period"] == pytest.approx(10.0)
    assert metrics["sample_count"] == 3  # 3 points after start
    assert metrics["valid_interval_count"] == 2
