import math
from typing import List

from .base_metric import BaseMetricCalculator
from ..model.tmt_model import Coordinate, TMTTrial, CursorInfo, TMTSubject, TMTTarget


class TotalDistanceCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, target_radius_multiplier: float,
                    crosses_time_threshold: float) -> dict:
        metrics[self.get_metric_name('total_distance')] = calculate_total_distance(trial)
        return metrics


def calculate_distance(pos1: Coordinate, pos2: Coordinate) -> float:
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    return math.hypot(dx, dy)


def is_inside_target(
        cursor_pos: Coordinate,
        target: TMTTarget,
        target_radius: float,
        multiplier: float
) -> bool:
    """
    Determine if the cursor is inside the target's effective radius.

    :param cursor_pos: The cursor position.
    :param target: The target to check.
    :param target_radius: The base target radius.
    :param multiplier: The radius multiplier.
    :return: True if cursor is inside the target's effective radius.
    """
    effective_radius = target_radius * multiplier
    distance = calculate_distance(cursor_pos, target.position)
    return distance < effective_radius


def calculate_total_distance(trial: TMTTrial):
    """
    Calculate the total distance of the cursor trail in a TMT trial.

    :param trial: TMTTrial object containing the cursor trail.
    :return: Total distance traveled by the cursor.
    """
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return sum(
        calculate_distance(cursor_trail_from_first_click[i].position, cursor_trail_from_first_click[i + 1].position)
        for i in range(len(cursor_trail_from_first_click) - 1)
    )


def calculate_total_distance_from_segment(segment_trial: List[CursorInfo]) -> float:
    return sum(
        calculate_distance(segment_trial[i].position, segment_trial[i + 1].position)
        for i in range(len(segment_trial) - 1)
    )
