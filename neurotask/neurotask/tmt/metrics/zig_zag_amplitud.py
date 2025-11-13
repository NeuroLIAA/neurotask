import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.intra_and_inter_target_time import get_intra_target_intervals
from neurotask.tmt.model.tmt_model import TMTTrial, TrialType, TMTSubject, TMTTarget, CursorInfo


class ZigZagAmplitude(BaseMetricCalculator):

    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals) -> dict:

        intra_target_intervals = get_intra_target_intervals(trial, subject)

        # Solo aplicable a Parte B
        if trial.trial_type != TrialType.PART_B:
            metrics[self.get_metric_name('zigzag_amplitude')] = np.nan
            return metrics

        # Remove first correct interval if it corresponds to the initial target
        if intra_target_intervals and intra_target_intervals[0][0].content == '1':
            intra_target_intervals = intra_target_intervals[1:]

        time_differences = []
        for i in range(0, len(intra_target_intervals) - 1, 2):
            letter_target, letter_start_cursor_info, _ = intra_target_intervals[i]
            number_target, number_start_cursor_info, _ = intra_target_intervals[i + 1]

            assert number_target.content.isdigit(), f"Expected number, got {number_target.content}"
            assert letter_target.content.isalpha(), f"Expected letter, got {letter_target.content}"

            # time_difference = tiempo de llegada a letra - tiempo de llegada a número
            time_difference = number_start_cursor_info.time - letter_start_cursor_info.time
            time_differences.append(time_difference)

        # Media de las diferencias, o NaN si no hay pares completos
        if time_differences:
            metrics[self.get_metric_name('zigzag_amplitude')] = float(np.mean(time_differences))
        else:
            metrics[self.get_metric_name('zigzag_amplitude')] = np.nan

        return metrics
