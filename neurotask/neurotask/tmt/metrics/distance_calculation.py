import math

from .base_metric import BaseMetricCalculator
from ..model.tmt_model import Coordinate, TMTTrial

class TotalDistanceCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial: TMTTrial, **params):
        metrics['total_distance'] = calculate_total_distance(trial)
        return metrics


def calculate_distance(pos1: Coordinate, pos2: Coordinate) -> float:
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    return math.hypot(dx, dy)


def calculate_total_distance(trial):
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return sum(
        calculate_distance(cursor_trail_from_first_click[i].position, cursor_trail_from_first_click[i + 1].position)
        for i in range(len(cursor_trail_from_first_click) - 1)
    )
