from typing import Dict, Any, List

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.distance_calculation import calculate_distance, calculate_total_distance_from_segment
from neurotask.tmt.metrics.targets_touch_calculator import get_all_trails_between_targets
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, TMTTarget, Coordinate
from scipy.interpolate import interp1d


def build_ideal_trail_segment(segment: List[CursorInfo]) -> List[CursorInfo]:
    """
    Dado un segmento real de CursorInfo, devuelve un segmento 'ideal'
    de la misma longitud, interpolando linealmente posición (x, y)
    y tiempo entre el primer y último punto.

    :param segment: lista de CursorInfo reales
    :return: lista de CursorInfo interpolados (misma longitud que `segment`)
    """
    n = len(segment)
    # Casos triviales
    if n == 0:
        return []
    if n == 1:
        # Solo un punto: retornamos copia del mismo
        return [
            CursorInfo(
                position=Coordinate(x=segment[0].position.x, y=segment[0].position.y),
                time=segment[0].time
            )
        ]

    # Extraer arrays de tiempos y posiciones de los puntos reales
    times = np.array([ci.time for ci in segment], dtype=float)
    xs = np.array([ci.position.x for ci in segment], dtype=float)
    ys = np.array([ci.position.y for ci in segment], dtype=float)

    # Creamos funciones de interpolación lineal solo entre el primer y último punto
    f_x = interp1d([times[0], times[-1]], [xs[0], xs[-1]])
    f_y = interp1d([times[0], times[-1]], [ys[0], ys[-1]])
    f_t = interp1d([0, n - 1], [times[0], times[-1]])  # si queremos uniformizar índices a tiempo

    # Usamos los tiempos reales para x,y y reconstruimos tiempos lineales para el ideal
    ideal_times = f_t(np.arange(n))
    ideal_x_positions = f_x(ideal_times)
    ideal_y_positions = f_y(ideal_times)

    # Construir la lista de CursorInfo ideales
    ideal_segment: List[CursorInfo] = []
    for t_i, x_i, y_i in zip(ideal_times, ideal_x_positions, ideal_y_positions):
        pos = Coordinate(x=float(x_i), y=float(y_i))
        ci = CursorInfo(position=pos, time=float(t_i))
        ideal_segment.append(ci)

    return ideal_segment


class DifferenceFromIdealDistance(BaseMetricCalculator):

    def add_metrics(
            self,
            metrics: Dict[str, Any],
            trial: TMTTrial,
            **params
    ) -> Dict[str, Any]:
        subject = params.get('subject')

        trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]] = (
            get_all_trails_between_targets(trial, subject.target_radius))

        ideal_distances = []
        for trail in trails_between_targets:
            target, cursor_trail = trail
            if target is None:
                continue

            # Calculate the ideal distance
            ideal_distance = self.calculate_distance_difference_from_ideal(cursor_trail)
            ideal_distances.append(ideal_distance)

        metrics['distance_difference_from_ideal'] = np.mean(ideal_distances)

        return metrics

    def calculate_distance_difference_from_ideal(self, segment: list[CursorInfo]) -> float:

        segment_distance = calculate_total_distance_from_segment(segment)
        ideal_distance = calculate_distance(segment[0].position, segment[-1].position)

        return abs(segment_distance - ideal_distance)
