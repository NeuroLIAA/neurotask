from typing import List

import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class TargetTime(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals) -> dict:

        if subject is None:
            raise ValueError("Subject must be provided")

        # Calculate time inside the targets
        # 3. Para cada segmento, calcular el tiempo dentro del target
        intra_times = []
        for target, start_ci, end_ci in correct_intervals:
            # end_ci.time es el instante en que deja de tocar
            # start_ci.time es el instante en que comienza a tocar
            dwell_time = end_ci.time - start_ci.time
            intra_times.append(dwell_time)

        # 4. Media de los tiempos, o NaN si no hay segmentos
        metrics[self.get_metric_name('intra_target_time')] = float(np.mean(intra_times))

        total_dwell = float(np.sum(intra_times))

        inter_time = self.calculate_inter_time(correct_intervals, total_dwell, trial)
        metrics[self.get_metric_name('inter_target_time')] = inter_time

        return metrics

    def calculate_inter_time(self, correct_segments, total_dwell, trial):

        finish_time = trial.get_cursor_trail_from_start()[-1].time
        total_time = finish_time - trial.start.time

        inter_time = total_time - total_dwell

        inter_time = float(inter_time)

        if len(correct_segments) > 0:
            self.validate_inter_time(correct_segments, inter_time, finish_time)

        return inter_time

    # Esta validacion solo esta por si acaso
    # Nunca deberia fallar, ambas metodologias deberian dar el mismo resultado
    def validate_inter_time(self, correct_segments, inter_time, finish_time):


        gaps: List[float] = []
        for prev_seg, next_seg in zip(correct_segments[:-1], correct_segments[1:]):
            _, _, prev_end_ci = prev_seg
            _, next_start_ci, _ = next_seg
            gap = next_start_ci.time - prev_end_ci.time
            if gap > 0:
                gaps.append(gap)

        #add last gap until finish time

        last_gap = finish_time - correct_segments[-1][2].time
        if last_gap > 0:
            gaps.append(last_gap)
        # Calculate the alternative inter_time

        alt_inter_time = float(np.sum(gaps))

        # assert np.isclose(inter_time, alt_inter_time, atol=1e-6), (
        #     f"inter_time ({inter_time}) != alt_inter_time ({alt_inter_time})"
        # )
