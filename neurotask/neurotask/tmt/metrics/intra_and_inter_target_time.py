from typing import List

import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.targets_touched import correct_touched_targets_for_every_cursor_point
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class TargetTime(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals) -> dict:

        if subject is None:
            raise ValueError("Subject must be provided")


def calculate_intra_target_time(
        trial: TMTTrial,
        subject: TMTSubject
) -> float:
    """
    Calculate the intra-target time for a given trial and subject.
    The intra-target time is defined as the sum of the times spent within each target area.
    """

    pass