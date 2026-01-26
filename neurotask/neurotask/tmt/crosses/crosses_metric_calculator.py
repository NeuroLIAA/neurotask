import numpy as np
from neurotask.tmt.crosses.crosses import calculate_crosses_for_trial
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class CrossesMetricCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, target_radius_multiplier: float,
                    crosses_time_threshold: float) -> dict:
        metric_key = self.get_metric_name("number_of_crosses")

        if calculate_crosses:
            num_crosses, _ = calculate_crosses_for_trial(trial, crosses_time_threshold)
            metrics[metric_key] = num_crosses
        else:
            metrics[metric_key] = np.nan

        return metrics
