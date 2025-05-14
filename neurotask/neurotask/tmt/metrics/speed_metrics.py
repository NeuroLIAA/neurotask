import logging
from typing import Dict, Any, Tuple, List

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.distance_calculation import calculate_distance
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, TMTSubject, TMTTarget


class SpeedMetricsCalculator(BaseMetricCalculator):

    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_targets_touches, wrong_targets_touches) -> dict:
        metrics.update(compute_speed_and_acceleration_metrics(trial))
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
        raise ValueError("current_cursor.time must be greater than previous_cursor.time")
    distance = calculate_distance(current_cursor.position, previous_cursor.position)
    time = current_cursor.time - previous_cursor.time
    return distance / time


def calculate_acceleration(current_speed: float, previous_speed: float, current_time: float,
                           previous_time: float) -> float:
    if current_time <= previous_time:
        raise ValueError("current_time must be greater than previous_time")
        # TODO GIAN VER en old tmt

    time_diff = current_time - previous_time

    speed_diff = current_speed - previous_speed
    acceleration = speed_diff / time_diff
    return acceleration


def calculate_speeds_between_cursor_positions(trial: TMTTrial) -> List[float]:
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return calculate_speeds(cursor_trail_from_first_click)


def calculate_speeds(cursor_trail: List[CursorInfo]) -> List[float]:
    if len(cursor_trail) < 2:
        raise ValueError("At least two points are required to calculate velocity")

    speeds = []

    for i in range(1, len(cursor_trail)):
        current_cursor = cursor_trail[i]
        previous_cursor = cursor_trail[i - 1]
        speed = calculate_speed(current_cursor, previous_cursor)
        if speed > 50:  # TODO GIAN PROBAR, ver porque pasa esto seguro es el sampling rate
            logging.warning(f"Speed value of {speed} detected. This may be an error.")
            raise ValueError(f"Speed value of {speed} detected. This may be an error.")
        speeds.append(speed)

    return speeds


def calculate_accelerations_between_cursor_positions(trial: TMTTrial) -> List[float]:
    """
    Calcula la aceleración entre cada posición del cursor.

    Returns:
    - Lista de aceleraciones entre puntos consecutivos.
    """
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()

    if len(cursor_trail_from_first_click) < 3:
        print(cursor_trail_from_first_click)
        raise ValueError("At least three points are required to calculate acceleration")

    accelerations = []

    # Calculamos las velocidades primero
    speeds = calculate_speeds_between_cursor_positions(trial)

    # Ahora calculamos la aceleración entre las velocidades
    for i in range(1, len(speeds)):
        current_cursor = cursor_trail_from_first_click[i + 1]  # i+1 porque estamos viendo del tercer punto en adelante
        previous_cursor = cursor_trail_from_first_click[i]
        current_speed = speeds[i]
        previous_speed = speeds[i - 1]

        acceleration = calculate_acceleration(
            current_speed, previous_speed, current_cursor.time, previous_cursor.time
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
