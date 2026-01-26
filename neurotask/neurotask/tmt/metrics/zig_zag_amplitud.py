import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.intra_and_inter_target_time import get_intra_target_intervals
from neurotask.tmt.model.tmt_model import TMTTrial, TrialType, TMTSubject, TMTTarget, CursorInfo


class ZigZagAmplitude(BaseMetricCalculator):

    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, target_radius_multiplier: float,
                    crosses_time_threshold: float) -> dict:

        intra_target_intervals = get_intra_target_intervals(trial, subject.target_radius, target_radius_multiplier)

        letter_to_number_key = self.get_metric_name('letter_to_number_latency')
        number_to_letter_key = self.get_metric_name('number_to_letter_latency')

        # Inicializar métricas en NaN
        metrics[letter_to_number_key] = np.nan
        metrics[number_to_letter_key] = np.nan

        # Solo aplicable a Parte B
        if trial.trial_type != TrialType.PART_B:
            return metrics

        letter_to_number_differences = []
        number_to_letter_differences = []

        for i in range(len(intra_target_intervals) - 1):
            current_target, current_start_cursor_info, _ = intra_target_intervals[i]
            next_target, next_start_cursor_info, _ = intra_target_intervals[i + 1]

            if current_target.content.isalpha():
                assert next_target.content.isdigit(), f"Expected number after letter, got {next_target.content}"
                letter_to_number_time = next_start_cursor_info.time - current_start_cursor_info.time
                letter_to_number_differences.append(letter_to_number_time)
            else:
                assert current_target.content.isdigit(), f"Expected number or letter, got {current_target.content}"
                assert next_target.content.isalpha(), f"Expected letter after number, got {next_target.content}"
                number_to_letter_time = next_start_cursor_info.time - current_start_cursor_info.time
                number_to_letter_differences.append(number_to_letter_time)

        # Media de las diferencias, o NaN si no hay pares completos
        if letter_to_number_differences:
            metrics[letter_to_number_key] = float(np.mean(letter_to_number_differences))

        if number_to_letter_differences:
            metrics[number_to_letter_key] = float(np.mean(number_to_letter_differences))

        return metrics
