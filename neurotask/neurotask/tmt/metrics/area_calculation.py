from typing import List

import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import CursorInfo, TMTTarget, TMTTrial, TMTSubject


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
            metrics: dict,
            trial: TMTTrial,
            subject: TMTSubject,
            trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
            calculate_crosses: bool,
            speed_threshold,
            consecutive_points,
            correct_targets_touches,
            wrong_targets_touches
    ) -> dict:

        areas = []
        for trail in trails_between_targets:
            target, cursor_trail = trail
            if target is None:
                continue
            area = area_between_real_and_ideal(cursor_trail)
            areas.append(area)

        metrics['area_difference_from_ideal'] = float(np.mean(areas))

        return metrics
