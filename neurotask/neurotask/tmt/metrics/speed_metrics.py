import logging
from typing import Dict, Any, Tuple, List, NamedTuple

import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.distance_calculation import calculate_distance
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, TMTSubject, TMTTarget
from neurotask.tmt.config import INVALID_SPEED_THRESHOLD


class SpeedResult(NamedTuple):
    """Result of a speed calculation with validity flag."""
    is_valid: bool
    value: float


class InvalidSpeedError(Exception):
    """Exception raised when speed exceeds INVALID_SPEED_THRESHOLD."""
    pass


class NonMonotonicTimeError(Exception):
    """Exception raised when cursor timestamps are not strictly increasing."""
    pass


class SpeedMetricsCalculator(BaseMetricCalculator):

    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points) -> dict:
        speed_metrics = compute_speed_and_acceleration_metrics(trial)

        # Apply get_metric_name method to each key in the dictionary
        for key, value in speed_metrics.items():
            metrics[self.get_metric_name(key)] = value

        return metrics


def compute_speed_and_acceleration_metrics(trial: TMTTrial) -> Dict[str, Any]:
    """
    Compute and return speed and acceleration statistics from the trial's cursor movements.

    The metrics include:
      - Mean, standard deviation, and peak speed.
      - Mean, standard deviation, and peak acceleration.
      - Mean, standard deviation, and peak of the absolute acceleration.
      - Mean, standard deviation, and peak (minimum) negative acceleration.

    Args:
        trial (TMTTrial): The trial containing cursor movement data.

    Returns:
        Dict[str, Any]: A dictionary with computed speed and acceleration metrics.
    """
    speeds = calculate_speeds_between_cursor_positions(trial)
    accelerations = calculate_accelerations_between_cursor_positions(trial)
    abs_accelerations = np.abs(accelerations)
    negative_accelerations = [acc for acc in accelerations if acc < 0]

    mean_speed, std_speed, peak_speed = safe_stats(speeds)
    mean_acc, std_acc, peak_acc = safe_stats(accelerations)
    mean_abs_acc, std_abs_acc, peak_abs_acc = safe_stats(abs_accelerations)
    # For negative accelerations, use np.min to capture the most negative value.
    mean_neg_acc, std_neg_acc, peak_neg_acc = safe_stats(negative_accelerations, peak_func=np.min)

    return {
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "peak_speed": peak_speed,
        "mean_acceleration": mean_acc,
        "std_acceleration": std_acc,
        "peak_acceleration": peak_acc,
        "mean_abs_acceleration": mean_abs_acc,
        "std_abs_acceleration": std_abs_acc,
        "peak_abs_acceleration": peak_abs_acc,
        "mean_negative_acceleration": mean_neg_acc,
        "std_negative_acceleration": std_neg_acc,
        "peak_negative_acceleration": peak_neg_acc
    }


def calculate_speed(current_cursor: CursorInfo, previous_cursor: CursorInfo) -> float:
    if current_cursor.time <= previous_cursor.time:
        raise NonMonotonicTimeError("current_cursor.time must be greater than previous_cursor.time")
    distance = calculate_distance(current_cursor.position, previous_cursor.position)
    time = current_cursor.time - previous_cursor.time
    speed = distance / time
    if speed > INVALID_SPEED_THRESHOLD:
        logging.warning(f"Speed value of {speed} detected. This may be an error.")
        raise InvalidSpeedError(f"Speed value of {speed} exceeds INVALID_SPEED_THRESHOLD ({INVALID_SPEED_THRESHOLD}).")
    return speed


def calculate_acceleration(current_speed: float, previous_speed: float, current_time: float,
                           previous_time: float) -> float:
    if current_time <= previous_time:
        raise NonMonotonicTimeError("current_time must be greater than previous_time")

    time_diff = current_time - previous_time

    speed_diff = current_speed - previous_speed
    acceleration = speed_diff / time_diff
    return acceleration


def calculate_speeds_between_cursor_positions(trial: TMTTrial, raise_on_error: bool = False) -> List[float]:
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return calculate_speeds(cursor_trail_from_first_click, raise_on_error)


def calculate_speeds_with_validity(cursor_trail: List[CursorInfo]) -> List[SpeedResult]:
    """
    Calculate speeds between consecutive cursor positions with validity flags.

    Args:
        cursor_trail: List of cursor positions.

    Returns:
        List of SpeedResult. Always len(result) == len(cursor_trail) - 1.
        If is_valid=False, value=0.0.
    """
    if len(cursor_trail) < 2:
        raise ValueError("At least two points are required to calculate velocity")

    results = []
    for i in range(1, len(cursor_trail)):
        try:
            speed = calculate_speed(cursor_trail[i], cursor_trail[i - 1])
            results.append(SpeedResult(is_valid=True, value=speed))
        except (InvalidSpeedError, NonMonotonicTimeError):
            results.append(SpeedResult(is_valid=False, value=0.0))

    return results


def calculate_speeds_between_cursor_positions_with_validity(trial: TMTTrial) -> List[SpeedResult]:
    """
    Calculate speeds between cursor positions with validity flags for a trial.

    Args:
        trial: The TMT trial.

    Returns:
        List of SpeedResult. Always len(result) == len(cursor_trail) - 1.
    """
    cursor_trail = trial.get_cursor_trail_from_start()
    return calculate_speeds_with_validity(cursor_trail)


def calculate_speeds(cursor_trail: List[CursorInfo], raise_on_error: bool = False) -> List[float]:
    speed_results = calculate_speeds_with_validity(cursor_trail)

    # Assert to validate alignment
    assert len(speed_results) == len(cursor_trail) - 1, \
        f"Speed results length ({len(speed_results)}) must equal cursor_trail length - 1 ({len(cursor_trail) - 1})"

    if raise_on_error:
        # If raise_on_error=True and there's any invalid result, recalculate to raise the exception
        for i, result in enumerate(speed_results):
            if not result.is_valid:
                # Recalculate to trigger the original exception
                calculate_speed(cursor_trail[i + 1], cursor_trail[i])

    # Return only valid speeds
    return [result.value for result in speed_results if result.is_valid]


def calculate_accelerations_between_cursor_positions(trial: TMTTrial) -> List[float]:
    """
    Calculate acceleration between each cursor position.

    Returns:
        List of accelerations between consecutive points.
    """
    cursor_trail = trial.get_cursor_trail_from_start()

    if len(cursor_trail) < 3:
        raise ValueError("At least three points are required to calculate acceleration")

    speed_results = calculate_speeds_between_cursor_positions_with_validity(trial)

    # Assert to validate index alignment
    assert len(speed_results) == len(cursor_trail) - 1, \
        f"Speed results length ({len(speed_results)}) must equal cursor_trail length - 1 ({len(cursor_trail) - 1})"

    accelerations = []
    for i in range(1, len(speed_results)):
        # Only calculate acceleration if both speeds are valid
        if not speed_results[i].is_valid or not speed_results[i - 1].is_valid:
            continue  # Skip this acceleration

        current_cursor = cursor_trail[i + 1]
        previous_cursor = cursor_trail[i]

        # Verify monotonic time before calculating
        if current_cursor.time <= previous_cursor.time:
            continue  # Skip non-monotonic time

        acceleration = calculate_acceleration(
            speed_results[i].value, speed_results[i - 1].value,
            current_cursor.time, previous_cursor.time
        )
        accelerations.append(acceleration)

    return accelerations


def safe_stats(data, peak_func=np.max) -> Tuple[float, float, float]:
    """
    Compute mean, standard deviation, and a peak value (using the provided peak function)
    for a list of numbers. If the list is empty, returns (np.nan, np.nan, np.nan).

    Args:
        data (Iterable[float]): The data from which to compute statistics.
        peak_func (Callable): Function to compute the peak value (default: np.max).

    Returns:
        Tuple[float, float, float]: (mean, std, peak_value)
    """
    if len(data) > 0:
        return np.mean(data), np.std(data), peak_func(data)
    return np.nan, np.nan, np.nan
