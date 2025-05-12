from typing import Dict, Any

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.distance_calculation import calculate_distance, calculate_total_distance_from_segment
from neurotask.tmt.metrics.targets_touch_calculator import get_all_trails_between_targets
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, TMTTarget


class DifferenceFromIdealDistance(BaseMetricCalculator):

    def add_metrics(
            self,
            metrics: Dict[str, Any],
            trial: TMTTrial,
            **params
    ) -> Dict[str, Any]:
        subject = params.get('subject')

        trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]] = (
            get_all_trails_between_targets(trial, subject.target_radius))

        ideal_distances = []
        for trail in trails_between_targets:
            target, cursor_trail = trail
            if target is None:
                continue

            # Calculate the ideal distance
            ideal_distance = self.calculate_distance_difference_from_ideal(cursor_trail)
            ideal_distances.append(ideal_distance)

        metrics['distance_difference_from_ideal'] = np.mean(ideal_distances)

        return metrics

    def calculate_distance_difference_from_ideal(self, segment: list[CursorInfo]) -> float:

        segment_distance = calculate_total_distance_from_segment(segment)
        ideal_distance = calculate_distance(segment[0].position, segment[-1].position)

        return abs(segment_distance - ideal_distance)
