from typing import Dict, Any

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.metrics import get_correct_and_incorrect_segments
# from neurotask.tmt.metrics.metrics import calculate_distance
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
            metrics['zigzag_time_diff'] = np.nan
            return metrics

        subject = params.get('subject')
        if subject is None:
            raise ValueError("Subject must be provided")
        target_radius = subject.target_radius

        # Segmentos correctos en orden (números y letras alternados)
        correct_segments, _ = get_correct_and_incorrect_segments(trial, target_radius)

        time_diffs = []
        # Recorremos pares [número, letra]
        for i in range(0, len(correct_segments) - 1, 2):
            num_target, num_start_ci, _ = correct_segments[i]
            let_target, let_start_ci, _ = correct_segments[i + 1]

            assert num_target.content.isdigit(), f"Expected number, got {num_target.content}"
            assert let_target.content.isalpha(), f"Expected letter, got {let_target.content}"

            # dt = tiempo de llegada a letra - tiempo de llegada a número
            dt = let_start_ci.time - num_start_ci.time
            time_diffs.append(dt)

        # Media de las diferencias, o NaN si no hay pares completos
        if time_diffs:
            metrics['zigzag_amplitude'] = float(np.mean(time_diffs))
        else:
            metrics['zigzag_amplitude'] = np.nan

        return metrics

# --- Función para calcular amplitud del serrucho ---
# class ZigZagAmplitude(BaseMetricCalculator): #(items, times):
#     def add_metrics(self, metrics: dict, trial: TMTTrial, **params) -> dict:
#         """Calcula la amplitud promedio del 'serrucho' en ensayos tipo B"""
#         time_deltas = np.diff([0] + list(trial.rt))
#
#         num_times = []
#         let_times = []
#
#         for item, dt in zip(trial.stimuli, time_deltas):
#             if item.isnumeric():
#                 num_times.append(dt)
#             elif item.isalpha():
#                 let_times.append(dt)
#
#         min_len = min(len(num_times), len(let_times))
#         num_times = num_times[:min_len]
#         let_times = let_times[:min_len]
#
#         zigzag_amplitudes = np.abs(np.array(num_times) - np.array(let_times))
#         metrics['zigzag_amplitude'] = np.mean(zigzag_amplitudes)
#         return metrics
