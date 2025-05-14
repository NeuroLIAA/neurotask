from typing import Dict, Any, List

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.targets_touch_calculator import get_correct_and_incorrect_target_touch_intervals
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class TargetTime(BaseMetricCalculator):
    def add_metrics(
            self,
            metrics: dict,
            trial: TMTTrial,
            subject: TMTSubject,
            trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
            calculate_crosses: bool,
            speed_threshold,
            consecutive_points,
            correct_targets_touches,
            wrong_targets_touches
    ) -> dict:

        if subject is None:
            raise ValueError("Subject must be provided")
        target_radius = subject.target_radius

        # Segmentos correctos en orden (números y letras alternados)
        correct_segments, _ = get_correct_and_incorrect_target_touch_intervals(trial, target_radius)

        # Calculate time inside the targets
        # 3. Para cada segmento, calcular el tiempo dentro del target
        intra_times = []
        for target, start_ci, end_ci in correct_segments:
            # end_ci.time es el instante en que deja de tocar
            # start_ci.time es el instante en que comienza a tocar
            dwell_time = end_ci.time - start_ci.time
            intra_times.append(dwell_time)

        # 4. Media de los tiempos, o NaN si no hay segmentos
        metrics['intra_target_time'] = float(np.mean(intra_times))

        total_dwell = float(np.sum(intra_times))

        inter_time = self.calculate_inter_time(correct_segments, metrics, total_dwell, trial)
        metrics['inter_target_time'] = inter_time

        return metrics

    def calculate_inter_time(self, correct_segments, metrics, total_dwell, trial):

        total_time = trial.get_cursor_trail_from_start()[-1].time - trial.start.time

        inter_time = total_time - total_dwell

        inter_time = float(inter_time)

        self.validate_inter_time(correct_segments, inter_time)

        return inter_time

    def validate_inter_time(self, correct_segments, inter_time):

        gaps: List[float] = []
        for prev_seg, next_seg in zip(correct_segments[:-1], correct_segments[1:]):
            _, _, prev_end_ci = prev_seg
            _, next_start_ci, _ = next_seg
            gap = next_start_ci.time - prev_end_ci.time
            if gap > 0:
                gaps.append(gap)
        alt_inter_time = float(np.sum(gaps))

        assert np.isclose(inter_time, alt_inter_time, atol=1e-6), (
            f"inter_time ({inter_time}) != alt_inter_time ({alt_inter_time})"
        )
