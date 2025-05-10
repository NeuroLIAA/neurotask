import logging
import math
from typing import List, Tuple, Optional

from ..model.tmt_model import Coordinate, TMTTrial, TMTTarget, CursorInfo


def calculate_distance(pos1: Coordinate, pos2: Coordinate) -> float:
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    return math.hypot(dx, dy)


def calculate_total_time(trial: TMTTrial) -> float:
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return cursor_trail_from_first_click[-1].time - cursor_trail_from_first_click[0].time


def calculate_total_distance(trial):
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return sum(
        calculate_distance(cursor_trail_from_first_click[i].position, cursor_trail_from_first_click[i + 1].position)
        for i in range(len(cursor_trail_from_first_click) - 1)
    )


def touched_targets_for_every_cursor_point(trial: TMTTrial, target_radius: float) -> List[
    Tuple[Optional[TMTTarget], CursorInfo]]:
    """
    Returns a list of tuples, where each tuple contains a target that was touched by the cursor and the cursor info.
    If no target is touched, the target is None.
    """
    trail_with_targets = []
    last_touched_target = None

    for cursor_info in trial.get_cursor_trail_from_start():
        # Verificar si el último target sigue siendo tocado para evitar buscar en todos los targets nuevamente
        if last_touched_target is not None:
            distance = calculate_distance(last_touched_target.position, cursor_info.position)
            if distance < target_radius:
                trail_with_targets.append((last_touched_target, cursor_info))
                continue

        # Buscar si el cursor actual toca algún nuevo target
        last_touched_target = get_touched_target_or_none(cursor_info, target_radius, trial)

        trail_with_targets.append((last_touched_target, cursor_info))

    return trail_with_targets


def get_touched_target_or_none(cursor_info: CursorInfo, target_radius: float, trial: TMTTrial) -> Optional[TMTTarget]:
    """
    Returns the target that the cursor is touching, if any. Otherwise, returns None.
    """
    for target in trial.stimuli:
        distance = calculate_distance(target.position, cursor_info.position)
        if distance < target_radius:
            return target

    return None


def get_target_touch_segments(
        trial: TMTTrial,
        target_radius: float
) -> List[Tuple[TMTTarget, List[CursorInfo]]]:
    """
    Devuelve una lista de tuplas para cada segmento continuo de toque sobre un target:
      (target, [lista de CursorInfo tocando ese target]).

    :param trial: instancia de TMTTrial con stimuli y cursor_trail.
    :param target_radius: radio para considerar un touch válido.
    :return: lista de (target, lista de puntos de cursor que tocaron ese target).
    """
    # Lista de (target|None, CursorInfo) para cada punto del cursor
    trail_with_targets = touched_targets_for_every_cursor_point(trial, target_radius)

    segments: List[Tuple[TMTTarget, List[CursorInfo]]] = []
    current_target: Optional[TMTTarget] = None
    current_points: List[CursorInfo] = []

    for touched_target, cursor_info in trail_with_targets:
        if touched_target is not None:
            # Si acabamos de empezar a tocar un nuevo target
            if touched_target != current_target:
                # Cerramos el segmento anterior
                if current_target is not None and current_points:
                    segments.append((current_target, current_points))
                # Iniciamos nuevo segmento
                current_target = touched_target
                current_points = [cursor_info]
            else:
                # Seguimos tocando el mismo target
                current_points.append(cursor_info)
        else:
            # Si dejamos de tocar un target, cerramos el segmento actual
            if current_target is not None and current_points:
                segments.append((current_target, current_points))
            current_target = None
            current_points = []

    # Si al final seguimos tocando un target, cerramos ese último segmento
    if current_target is not None and current_points:
        segments.append((current_target, current_points))

    return segments


def get_correct_and_incorrect_segments(
    trial: TMTTrial,
    target_radius: float
) -> Tuple[
    List[Tuple[TMTTarget, List[CursorInfo]]],
    List[Tuple[TMTTarget, List[CursorInfo]]]
]:
    """
    Devuelve dos listas basadas en segmentos de touch-points completos:
    1. correct_segments: segmentos de targets tocados en el orden esperado.
    2. incorrect_segments: segmentos de targets tocados fuera de orden.

    Cada segmento es una tupla (target, [lista de CursorInfo tocando ese target]).
    """
    # Obtener segmentos con todos los puntos intermedios
    segments: List[Tuple[TMTTarget, List[CursorInfo]]] = \
        get_target_touch_segments(trial, target_radius)

    correct_segments: List[Tuple[TMTTarget, List[CursorInfo]]] = []
    incorrect_segments: List[Tuple[TMTTarget, List[CursorInfo]]] = []
    expected_idx = 0

    for target, points in segments:
        # Comprobamos si coincide con el siguiente target esperado
        if expected_idx < len(trial.stimuli) and target == trial.stimuli[expected_idx]:
            correct_segments.append((target, points))
            expected_idx += 1
        else:
            incorrect_segments.append((target, points))

    return correct_segments, incorrect_segments


def number_of_correct_and_incorrect_segments(trial: TMTTrial, target_radius: float) -> Tuple[int, int]:
    """
    Devuelve el número de segmentos de targets correctamente tocados e incorrectamente tocados.
    """
    correct_segments, incorrect_segments = get_correct_and_incorrect_segments(trial, target_radius)
    return len(correct_segments), len(incorrect_segments)
