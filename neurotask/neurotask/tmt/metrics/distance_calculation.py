import math
from typing import List

from .base_metric import BaseMetricCalculator
from ..model.tmt_model import Coordinate, TMTTrial, CursorInfo, TMTSubject, TMTTarget


class TotalDistanceCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals, wrong_intervals) -> dict:
        metrics['total_distance'] = calculate_total_distance(trial)
        return metrics


def calculate_distance(pos1: Coordinate, pos2: Coordinate) -> float:
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    return math.hypot(dx, dy)


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
