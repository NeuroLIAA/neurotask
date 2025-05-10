from typing import Dict, Any
from typing import List, Tuple

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial


def get_ideal_target_segments(
        trial: TMTTrial,
        num_interpolation_points: int = 20
) -> List[List[Tuple[float, float]]]:
    """
    Genera los segmentos ideales de recorrido entre targets consecutivos
    usando interpolación lineal.

    :param trial: TMTTrial con la lista de stimuli en orden de ejecución.
    :param num_interpolation_points: número de puntos a interpolar por segmento
                                     (incluyendo los extremos).
    :return: Lista de segmentos; cada segmento es una lista de (x, y) interpolados.
    """
    segments: List[List[Tuple[float, float]]] = []
    stimuli = trial.stimuli

    # Si hay menos de 2 targets, no hay segmentos
    if len(stimuli) < 2:
        return segments

    # Para cada par consecutivo de targets
    for start_tgt, end_tgt in zip(stimuli[:-1], stimuli[1:]):
        x0, y0 = start_tgt.position.x, start_tgt.position.y
        x1, y1 = end_tgt.position.x, end_tgt.position.y

        # Generar valores lineales entre x0→x1 y y0→y1
        xs = np.linspace(x0, x1, num=num_interpolation_points)
        ys = np.linspace(y0, y1, num=num_interpolation_points)

        # Combinar en lista de tuplas y guardarla
        segment = [(float(x), float(y)) for x, y in zip(xs, ys)]
        segments.append(segment)

    return segments


class LineDeviation(BaseMetricCalculator):
    def add_metrics(
            self,
            metrics: Dict[str, Any],
            trial: TMTTrial,
            **params
    ) -> Dict[str, Any]:
