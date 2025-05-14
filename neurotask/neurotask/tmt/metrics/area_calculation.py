from typing import List, Dict, Any

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.targets_touch_calculator import get_all_trails_between_targets
from neurotask.tmt.model.tmt_model import CursorInfo, TMTTarget, TMTTrial


def area_between_real_and_ideal(segment: List[CursorInfo]) -> float:
    # 1) Extraer coordenadas (x, y) de los puntos reales
    point_coords = np.array([[cursor_info.position.x, cursor_info.position.y] for cursor_info in segment], dtype=float)

    return area_between_real_and_ideal_points(point_coords)


def area_between_real_and_ideal_points(point_coords: np.ndarray) -> float:
    """
    Integra la distancia perpendicular absoluta entre la trayectoria real
    y la línea recta ideal que une el primer y último punto del segmento.

    :param segment: lista de CursorInfo reales
    :return: área bajo la curva de desviación, >= 0
    """
    num_points = len(point_coords)
    if num_points < 2:
        return 0.0

    # 2) Definir el vector ideal desde el primer al último punto
    start_point = point_coords[0]
    end_point = point_coords[-1]
    ideal_vector = end_point - start_point
    ideal_length_sq = np.dot(ideal_vector, ideal_vector)
    ideal_length = np.sqrt(ideal_length_sq)

    # 3) Proyectar cada punto real sobre la línea ideal
    #    a) desplazamientos desde el inicio
    offsets = point_coords - start_point[np.newaxis, :]
    #    b) factor de proyección t ∈ [0,1] sobre el vector ideal
    projection_factors = (offsets @ ideal_vector) / ideal_length_sq
    #    c) puntos proyectados en la línea ideal
    projection_points = start_point + projection_factors[:, np.newaxis] * ideal_vector

    # 4) Calcular distancia perpendicular de cada punto real a la ideal
    perpendicular_distances = np.linalg.norm(point_coords - projection_points, axis=1)

    # 5) Coordenadas escalarizadas a lo largo de la línea ideal
    #    (distancia desde el inicio a cada proyección)
    line_positions = projection_factors * ideal_length

    # 6) Integración por la regla del trapecio
    area = np.trapz(perpendicular_distances, line_positions)

    return float(area)


class DifferenceFromIdealArea(BaseMetricCalculator):

    def add_metrics(
            self,
            metrics: Dict[str, Any],
            trial: TMTTrial,
            **params
    ) -> Dict[str, Any]:
        subject = params.get('subject')

        trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]] = (
            get_all_trails_between_targets(trial, subject.target_radius))

        areas = []
        for trail in trails_between_targets:
            target, cursor_trail = trail
            if target is None:
                continue
            area = area_between_real_and_ideal(cursor_trail)
            areas.append(area)

        metrics['area_difference_from_ideal'] = float(np.mean(areas))

        return metrics

# def build_ideal_trail_segment(segment: List[CursorInfo]) -> List[CursorInfo]:
#     """
#     Dado un segmento real de CursorInfo, devuelve un segmento 'ideal'
#     de la misma longitud, interpolando linealmente posición (x, y)
#     y tiempo entre el primer y último punto.
#
#     :param segment: lista de CursorInfo reales
#     :return: lista de CursorInfo interpolados (misma longitud que `segment`)
#     """
#     n = len(segment)
#     # Casos triviales
#     if n == 0:
#         return []
#     if n == 1:
#         # Solo un punto: retornamos copia del mismo
#         return [
#             CursorInfo(
#                 position=Coordinate(x=segment[0].position.x, y=segment[0].position.y),
#                 time=segment[0].time
#             )
#         ]
#
#     # Extraer arrays de tiempos y posiciones de los puntos reales
#     times = np.array([ci.time for ci in segment], dtype=float)
#     xs = np.array([ci.position.x for ci in segment], dtype=float)
#     ys = np.array([ci.position.y for ci in segment], dtype=float)
#
#     # Creamos funciones de interpolación lineal solo entre el primer y último punto
#     f_x = interp1d([times[0], times[-1]], [xs[0], xs[-1]])
#     f_y = interp1d([times[0], times[-1]], [ys[0], ys[-1]])
#     f_t = interp1d([0, n - 1], [times[0], times[-1]])  # si queremos uniformizar índices a tiempo
#
#     # Usamos los tiempos reales para x,y y reconstruimos tiempos lineales para el ideal
#     ideal_times = f_t(np.arange(n))
#     ideal_x_positions = f_x(ideal_times)
#     ideal_y_positions = f_y(ideal_times)
#
#     # Construir la lista de CursorInfo ideales
#     ideal_segment: List[CursorInfo] = []
#     for t_i, x_i, y_i in zip(ideal_times, ideal_x_positions, ideal_y_positions):
#         pos = Coordinate(x=float(x_i), y=float(y_i))
#         ci = CursorInfo(position=pos, time=float(t_i))
#         ideal_segment.append(ci)
#
#     return ideal_segment
