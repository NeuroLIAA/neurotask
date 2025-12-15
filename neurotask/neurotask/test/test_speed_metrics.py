import pytest

from neurotask.tmt.config import INVALID_SPEED_THRESHOLD
from neurotask.tmt.metrics.speed_metrics import (
    SpeedMetricsCalculator,
    InvalidSpeedError,
    NonMonotonicTimeError,
)
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTarget,
    TMTTrial,
    TrialType,
)


# =============================================================================
# Helper functions
# =============================================================================

def _build_cursor_trail(positions_and_times: list[tuple[float, float, float]]) -> list[CursorInfo]:
    """
    Build a cursor trail from a list of (x, y, time) tuples.
    """
    return [
        CursorInfo(Coordinate(x, y), t)
        for x, y, t in positions_and_times
    ]


def _build_trial_and_subject(
        cursor_trail: list[CursorInfo],
) -> tuple[TMTTrial, TMTSubject]:
    """
    Build a minimal trial and subject with the given cursor trail.
    """
    targets = [TMTTarget("0", Coordinate(0.0, 0.0))]

    trial = TMTTrial(
        stimuli=targets,
        cursor_trail=cursor_trail,
        trial_type=TrialType.PART_A,
        id="test_trial",
        order_of_appearance=1,
        rt=cursor_trail[-1].time if cursor_trail else 0.0,
    )

    subject = TMTSubject(
        training_trials=[],
        testing_trials=[trial],
        target_radius=1.0,
        canvas_size=None,
    )

    return trial, subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject, prefix: str = None) -> dict:
    """
    Compute speed metrics using SpeedMetricsCalculator.
    """
    calculator = SpeedMetricsCalculator(prefix=prefix)
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
# Tests
# =============================================================================

def test_uniform_motion_returns_constant_speed():
    """
    Uniform motion (constant speed) should have:
    - mean_speed == peak_speed
    - std_speed == 0
    """
    # Move 2 units every 1 ms -> speed = 2 px/ms
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 1.0),
        (4.0, 0.0, 2.0),
        (6.0, 0.0, 3.0),
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_speed"] == pytest.approx(2.0)
    assert metrics["std_speed"] == pytest.approx(0.0)
    assert metrics["peak_speed"] == pytest.approx(2.0)


def test_accelerating_motion_returns_positive_acceleration():
    """
    Accelerating motion (increasing speed) should have:
    - mean_acceleration > 0
    - peak_acceleration > 0
    """
    # Speeds: 1, 2, 3 px/ms -> accelerations: 1, 1 px/ms²
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),   # speed = 1
        (3.0, 0.0, 2.0),   # speed = 2
        (6.0, 0.0, 3.0),   # speed = 3
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_acceleration"] == pytest.approx(1.0)
    assert metrics["peak_acceleration"] == pytest.approx(1.0)
    assert metrics["std_acceleration"] == pytest.approx(0.0)


def test_decelerating_motion_returns_negative_acceleration():
    """
    Decelerating motion (decreasing speed) should have:
    - mean_acceleration < 0
    - mean_negative_acceleration < 0
    - peak_negative_acceleration is the most negative value
    """
    # Speeds: 3, 2, 1 px/ms -> accelerations: -1, -1 px/ms²
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),   # speed = 3
        (5.0, 0.0, 2.0),   # speed = 2
        (6.0, 0.0, 3.0),   # speed = 1
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_acceleration"] == pytest.approx(-1.0)
    assert metrics["mean_negative_acceleration"] == pytest.approx(-1.0)
    assert metrics["peak_negative_acceleration"] == pytest.approx(-1.0)


def test_returns_all_twelve_metrics():
    """
    SpeedMetricsCalculator should return all 12 expected metrics.
    """
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (3.0, 0.0, 2.0),
        (6.0, 0.0, 3.0),
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    expected_keys = [
        "mean_speed", "std_speed", "peak_speed",
        "mean_acceleration", "std_acceleration", "peak_acceleration",
        "mean_abs_acceleration", "std_abs_acceleration", "peak_abs_acceleration",
        "mean_negative_acceleration", "std_negative_acceleration", "peak_negative_acceleration",
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_with_prefix_adds_prefix_to_all_keys():
    """
    When initialized with a prefix, all metric keys should have that prefix.
    """
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 1.0),
        (4.0, 0.0, 2.0),
        (6.0, 0.0, 3.0),
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject, prefix="non_cut_")

    # Check that prefixed keys exist
    assert "non_cut_mean_speed" in metrics
    assert "non_cut_peak_speed" in metrics
    assert "non_cut_mean_acceleration" in metrics

    # Check that non-prefixed keys do not exist
    assert "mean_speed" not in metrics
    assert "peak_speed" not in metrics


def test_invalid_speed_raises_error():
    """
    When speed exceeds INVALID_SPEED_THRESHOLD, InvalidSpeedError should be raised.
    """
    # Create movement that exceeds threshold (8 px/ms)
    # Moving 100 pixels in 1 ms = 100 px/ms > 8 px/ms
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 1.0),
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    with pytest.raises(InvalidSpeedError, match="exceeds INVALID_SPEED_THRESHOLD"):
        _compute_metrics(trial, subject)


def test_mixed_acceleration_computes_abs_correctly():
    """
    Mixed acceleration (both positive and negative) should compute
    abs_acceleration correctly.
    """
    # Speeds: 1, 3, 1 px/ms -> accelerations: +2, -2 px/ms²
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),   # speed = 1
        (4.0, 0.0, 2.0),   # speed = 3
        (5.0, 0.0, 3.0),   # speed = 1
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Mean acceleration should be 0 (symmetric)
    assert metrics["mean_acceleration"] == pytest.approx(0.0)
    
    # Mean abs acceleration should be 2
    assert metrics["mean_abs_acceleration"] == pytest.approx(2.0)
    
    # Peak abs acceleration should be 2
    assert metrics["peak_abs_acceleration"] == pytest.approx(2.0)
    
    # Negative acceleration stats
    assert metrics["mean_negative_acceleration"] == pytest.approx(-2.0)
    assert metrics["peak_negative_acceleration"] == pytest.approx(-2.0)


def test_speed_at_threshold_is_valid():
    """
    Speed exactly at INVALID_SPEED_THRESHOLD should not raise an error.
    """
    # Create movement at exactly the threshold (need 3 points for acceleration calculation)
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (INVALID_SPEED_THRESHOLD, 0.0, 1.0),      # speed = threshold
        (INVALID_SPEED_THRESHOLD * 2, 0.0, 2.0),  # speed = threshold
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["mean_speed"] == pytest.approx(INVALID_SPEED_THRESHOLD)


def test_non_monotonic_time_raises_error():
    """
    When timestamps go backwards, NonMonotonicTimeError should be raised.
    """
    # Time goes from 0 -> 2 -> 1 (backwards)
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0),
        (2.0, 0.0, 1.0),  # time goes backwards
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    with pytest.raises(NonMonotonicTimeError, match="current_cursor.time must be greater than previous_cursor.time"):
        _compute_metrics(trial, subject)


def test_equal_time_raises_non_monotonic_error():
    """
    When two consecutive timestamps are equal, NonMonotonicTimeError should be raised.
    """
    # Time stays at 1.0 for two consecutive points
    cursor_trail = _build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),  # same time as previous
    ])
    trial, subject = _build_trial_and_subject(cursor_trail)

    with pytest.raises(NonMonotonicTimeError, match="current_cursor.time must be greater than previous_cursor.time"):
        _compute_metrics(trial, subject)
