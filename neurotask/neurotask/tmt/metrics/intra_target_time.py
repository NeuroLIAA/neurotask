from typing import Dict, Any

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.targets_touch_calculator import get_correct_and_incorrect_target_touch_intervals
from neurotask.tmt.model.tmt_model import TMTTrial


class TargetTime(BaseMetricCalculator):
    def add_metrics(
            self,
            metrics: Dict[str, Any],
            trial: TMTTrial,
            **params
    ) -> Dict[str, Any]:

        subject = params.get('subject')
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

        return metrics
