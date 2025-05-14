import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TrialType, TMTSubject, TMTTarget, CursorInfo


class ZigZagAmplitude(BaseMetricCalculator):

    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_targets_touches, wrong_targets_touches,
                    correct_intervals, wrong_intervals) -> dict:

        # Solo aplicable a Parte B
        if trial.trial_type != TrialType.PART_B:
            metrics['zigzag_amplitude'] = np.nan
            return metrics

        target_radius = subject.target_radius

        time_differences = []
        # Recorremos pares [número, letra]
        for i in range(0, len(correct_intervals) - 1, 2):
            number_target, number_start_cursor_info, _ = correct_intervals[i]
            letter_target, letter_start_cursor_info, _ = correct_intervals[i + 1]

            assert number_target.content.isdigit(), f"Expected number, got {number_target.content}"
            assert letter_target.content.isalpha(), f"Expected letter, got {letter_target.content}"

            # time_difference = tiempo de llegada a letra - tiempo de llegada a número
            time_difference = letter_start_cursor_info.time - number_start_cursor_info.time
            time_differences.append(time_difference)

        # Media de las diferencias, o NaN si no hay pares completos
        if time_differences:
            metrics['zigzag_amplitude'] = float(np.mean(time_differences))
        else:
            metrics['zigzag_amplitude'] = np.nan

        return metrics
