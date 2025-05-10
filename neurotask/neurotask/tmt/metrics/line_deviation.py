import math
from typing import Dict, Any
from typing import List, Tuple

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.metrics import get_correct_and_incorrect_segments
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, TMTTarget, TrialType


def get_ideal_target_segments(
        trial: TMTTrial,
        num_interpolation_points: int = 20
) -> List[Tuple[TMTTarget, CursorInfo, CursorInfo]]:
    """
    Genera segmentos “ideales” entre cada par de targets consecutivos,
    en la misma estructura que los segmentos de toque:
      List of (target, start_cursor_info, end_cursor_info)
    donde los CursorInfo tienen tiempo=None y posición interpolada.

    :param trial: TMTTrial con la lista de stimuli en orden.
    :param num_interpolation_points: puntos a interpolar en cada segmento.
    :return: lista de tuplas (TMTTarget, CursorInfo, CursorInfo).
    """
    segments: List[Tuple[TMTTarget, CursorInfo, CursorInfo]] = []
    stimuli = trial.stimuli
    if len(stimuli) < 2:
        return segments

    # Para cada par consecutivo de targets
    for start_tgt, end_tgt in zip(stimuli[:-1], stimuli[1:]):
        x0, y0 = start_tgt.position.x, start_tgt.position.y
        x1, y1 = end_tgt.position.x, end_tgt.position.y

        # Generamos interpolación lineal
        xs = np.linspace(x0, x1, num=num_interpolation_points)
        ys = np.linspace(y0, y1, num=num_interpolation_points)

        # Creamos CursorInfo “ideales” con time=None en cada extremo
        start_ci = CursorInfo(position=start_tgt.position, time=None)
        end_ci = CursorInfo(position=end_tgt.position, time=None)

        # Devolvemos solo el par de extremos (match a estructura real)
        segments.append((end_tgt, start_ci, end_ci))

    return segments



class LineDeviation(BaseMetricCalculator):
    def add_metrics(
        self,
        metrics: Dict[str, Any],
        trial: TMTTrial,
        **params
    ) -> Dict[str, Any]:
        """
        Calcula la desviación media de la trayectoria real respecto
        a la línea ideal entre cada par de targets consecutivos
        correctamente tocados, reutilizando la función de segmentos ideales.

        :param trial: TMTTrial con stimuli, cursor_trail, trial_type, etc.
        :param target_radius: radio para considerar un touch válido.
        :param num_interpolation_points: resolución al generar segmentos ideales.
        :returns: metrics con 'line_deviation'.
        """
        # Sólo aplicable a PART_B

        num_interpolation_points: int = 20

        target_radius: float = params.get('subject').target_radius

        # Obtengo segmentos correctos (target, start_ci, end_ci)
        correct_segments, _ = get_correct_and_incorrect_segments(trial, target_radius)

        # Genero los segmentos ideales
        ideal_segments = get_ideal_target_segments(trial, num_interpolation_points)

        # Traza completa del cursor
        full_trail = trial.get_cursor_trail_from_start()

        comparisons = []

        # Para cada par consecutivo de segmentos correctos
        # (el i-ésimo correct_segments corresponde al ideal_segments[i])
        for i in range(len(correct_segments) - 1):
            # Sacar los tiempos de fin del segmento correcto i y inicio de i+1
            _, _, end_ci_i = correct_segments[i]
            _, start_ci_j, _ = correct_segments[i + 1]

            # Filtrar puntos reales en ese intervalo temporal
            segment_points = [
                ci.position
                for ci in full_trail
                if end_ci_i.time <= ci.time <= start_ci_j.time
            ]
            if not segment_points or i >= len(ideal_segments):
                continue

            # Usar los extremos del segmento ideal correspondiente
            ideal_seg = ideal_segments[i]

            comparison_metric = compare_ideal_vs_real(ideal_seg, segment_points)
            comparisons.append(comparison_metric)

        metrics['line_deviation'] = float(np.mean(deviations)) if deviations else np.nan
        return metrics

