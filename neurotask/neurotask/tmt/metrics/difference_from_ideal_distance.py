import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.distance_calculation import calculate_distance, calculate_total_distance_from_segment
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, TMTTarget, TMTSubject


class DifferenceFromIdealDistance(BaseMetricCalculator):

    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, target_radius_multiplier: float) -> dict:

        differences = []
        for trail in trails_between_targets:
            target, cursor_trail = trail
            if target is None:
                continue
            difference = self.calculate_distance_difference_from_ideal(cursor_trail)
            differences.append(difference)

        if differences:
            metrics[self.get_metric_name('distance_difference_from_ideal')] = float(np.mean(differences))
        else:
            raise ValueError(
                f"Cannot calculate distance_difference_from_ideal: No valid target segments found. "
                f"Trial {trial.id} has {len(trails_between_targets)} trails_between_targets, "
                f"but none had valid targets (all targets were None or segments were invalid)."
            )

        return metrics

    def calculate_distance_difference_from_ideal(self, segment: list[CursorInfo]) -> float:

        segment_distance = calculate_total_distance_from_segment(segment)
        ideal_distance = calculate_distance(segment[0].position, segment[-1].position)

        return abs(segment_distance - ideal_distance)
