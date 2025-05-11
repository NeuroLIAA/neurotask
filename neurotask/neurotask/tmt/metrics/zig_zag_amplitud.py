import logging
from typing import Dict, Any

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.targets_touch_calculator import get_correct_and_incorrect_target_touch_intervals
from neurotask.tmt.model.tmt_model import TMTTrial, TrialType


class ZigZagAmplitude(BaseMetricCalculator):
    def add_metrics(
            self,
            metrics: Dict[str, Any],
            trial: TMTTrial,
            **params
    ) -> Dict[str, Any]:

        # Solo aplicable a Parte B
        if trial.trial_type != TrialType.PART_B:
            metrics['zigzag_amplitude'] = np.nan
            return metrics

        subject = params.get('subject')
        if subject is None:
            raise ValueError("Subject must be provided")
        target_radius = subject.target_radius

        # Segmentos correctos en orden (números y letras alternados)
        correct_segments, _ = get_correct_and_incorrect_target_touch_intervals(trial, target_radius)

        time_differences = []
        # Recorremos pares [número, letra]
        for i in range(0, len(correct_segments) - 1, 2):
            number_target, number_start_cursor_info, _ = correct_segments[i]
            letter_target, letter_start_cursor_info, _ = correct_segments[i + 1]

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
