import numpy as np
from neurotask.tmt.crosses.crosses import calculate_crosses_for_trial
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class CrossesMetricCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals) -> dict:
        metrics[self.get_metric_name("number_of_crosses")] = calculate_crosses_for_trial(trial) if calculate_crosses else np.nan
        return metrics
