import pytest

from neurotask.tmt.segmentation.segmentation_metric import SegmentationMetricCalculator
from neurotask.tmt.model.tmt_model import (
    TMTSubject,
    TMTTrial,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject, speed_threshold: float, 
                     consecutive_points: int, prefix: str = None) -> dict:
    """
    Compute segmentation metrics using SegmentationMetricCalculator.
    
    :param trial: TMTTrial object
    :param subject: TMTSubject object
    :param speed_threshold: Speed threshold for segmentation (must be positive)
    :param consecutive_points: Number of consecutive points for state transitions (must be positive)
    :param prefix: Optional prefix for metric names
    :return: Dictionary of segmentation metrics
    """
    calculator = SegmentationMetricCalculator(prefix=prefix)
    return calculator.add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=[],
        calculate_crosses=False,
        speed_threshold=speed_threshold,
        consecutive_points=consecutive_points,
    )


# =============================================================================
# Tests
# =============================================================================

def test_returns_all_expected_metrics():
    """
    Verify that SegmentationMetricCalculator returns all 15 expected metrics.
    
    Expected metrics:
    - hesitation_time, travel_time, search_time
    - hesitation_distance, travel_distance, search_distance
    - hesitation_avg_speed, travel_avg_speed, search_avg_speed
    - state_transitions
    - hesitation_ratio
    - total_hesitations, average_duration, max_duration, hesitation_periods
    """
    # Create a cursor trail that will generate different states
    # Start at target, move away (Search -> Travel), slow down (Travel -> Hesitation)
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),      # Start at target (Search)
        (5.0, 0.0, 1.0),      # Move away, speed = 5
        (15.0, 0.0, 2.0),     # Continue moving, speed = 10 (Travel)
        (25.0, 0.0, 3.0),     # Continue moving, speed = 10 (Travel)
        (30.0, 0.0, 4.0),     # Slow down, speed = 5 (Hesitation)
        (32.0, 0.0, 5.0),     # Very slow, speed = 2 (Hesitation)
    ])
    
    # Create targets - first target at (0, 0) with radius 2.0
    trial, subject = build_trial_and_subject(
        cursor_trail=cursor_trail,
        targets=None,  # Will create default target at (0, 0)
        target_radius=2.0,
    )
    
    # Use reasonable parameters for segmentation
    speed_threshold = 3.0  # Threshold to distinguish Travel from Hesitation
    consecutive_points = 2  # Number of consecutive points needed for state change
    
    metrics = _compute_metrics(trial, subject, speed_threshold, consecutive_points)
    
    # Verify all expected metrics are present
    expected_keys = [
        "hesitation_time",
        "travel_time",
        "search_time",
        "hesitation_distance",
        "travel_distance",
        "search_distance",
        "hesitation_avg_speed",
        "travel_avg_speed",
        "search_avg_speed",
        "state_transitions",
        "hesitation_ratio",
        "total_hesitations",
        "average_duration",
        "max_duration",
        "hesitation_periods",
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"

