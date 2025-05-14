from typing import List, Tuple, Optional

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator

from .distance_calculation import calculate_distance
from ..model.tmt_model import TMTTrial, TMTTarget, CursorInfo, TMTSubject


class TargetsTouchesCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_targets_touches, wrong_targets_touches,
                    correct_intervals, wrong_intervals) -> dict:
        metrics['correct_targets_touches'] = correct_targets_touches
        metrics['wrong_targets_touches'] = wrong_targets_touches

        return metrics


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


def get_all_trails_between_targets(
        trial: TMTTrial,
        target_radius: float
) -> List[Tuple[TMTTarget, List[CursorInfo]]]:
    """
    Devuelve, para cada target en `trial.stimuli`, la lista de CursorInfo
    desde el inicio (o desde el toque del target anterior) hasta el momento
    en que se toca ese target.

    Cada tupla es (target, segmento_de_cursor), donde:
      - `target` es el TMTTarget esperado.
      - `segmento_de_cursor` es la lista de CursorInfo desde el corte anterior
        hasta el primer toque de ese target.

    Si algún target esperado no llega a tocarse, se detiene la generación de
    segmentos.

    :param trial:  instancia de TMTTrial
    :param target_radius:  radio para detección de toques por `touched_targets_for_every_cursor_point`
    :return: lista de (TMTTarget, List[CursorInfo]) en orden de aparición
    """
    segments: List[Tuple[TMTTarget, List[CursorInfo]]] = []
    # Secuencia (target o None, cursor_info) para cada punto de cursor
    trail_with_targets = touched_targets_for_every_cursor_point(trial, target_radius)

    # Iterador único sobre la secuencia de toques/puntos
    trail_iter = iter(trail_with_targets)

    # Para cada target esperado, vamos construyendo su segmento
    for expected in trial.stimuli[1:]:
        current_segment: List[CursorInfo] = []
        for touched, cursor_info in trail_iter:
            current_segment.append(cursor_info)
            if touched == expected:
                # Cerramos el segmento al primer toque válido
                segments.append((expected, current_segment.copy()))
                break
        else:
            # No encontramos el target esperado en lo que queda de trail
            break

    return segments


def get_touched_target_or_none(cursor_info: CursorInfo, target_radius: float, trial: TMTTrial) -> Optional[TMTTarget]:
    """
    Returns the target that the cursor is touching, if any. Otherwise, returns None.
    """
    for target in trial.stimuli:
        distance = calculate_distance(target.position, cursor_info.position)
        if distance < target_radius:
            return target

    return None


def get_target_touch_intervals(trial: TMTTrial, target_radius: float) -> List[Tuple[TMTTarget, CursorInfo, CursorInfo]]:
    """
    Devuelve una lista de tuplas que contienen el target tocado,
    el cursor info cuando comenzó a tocar el target y cuando dejó de tocarlo.
    """
    trail_with_targets = touched_targets_for_every_cursor_point(trial, target_radius)

    segments = []
    current_target = None
    start_cursor_info = None

    for touched_target, cursor_info in trail_with_targets:
        if touched_target is not None:
            if current_target is None:
                # Si no se está tocando ningún target, comenzamos un nuevo segmento
                current_target = touched_target
                start_cursor_info = cursor_info
            elif touched_target != current_target:
                # Si se toca un target diferente al actual, cerramos el segmento anterior
                segments.append((current_target, start_cursor_info, cursor_info))
                # Iniciamos un nuevo segmento con el nuevo target
                current_target = touched_target
                start_cursor_info = cursor_info
        elif current_target is not None:
            # Si el cursor dejó de tocar el target actual, cerramos el segmento
            segments.append((current_target, start_cursor_info, cursor_info))
            current_target = None
            start_cursor_info = None

    # Si al final se sigue tocando un target, cerramos el último segmento
    if current_target is not None:
        segments.append((current_target, start_cursor_info, trail_with_targets[-1][1]))

    return segments


def get_correct_and_incorrect_target_touch_intervals(trial: TMTTrial, target_radius: float) -> Tuple[
    List[Tuple[TMTTarget, CursorInfo, CursorInfo]],
    List[Tuple[TMTTarget, CursorInfo, CursorInfo]]
]:
    """
    Devuelve dos listas:
    1. Los segmentos de targets correctamente tocados (en orden).
    2. Los segmentos de targets incorrectamente tocados (fuera de orden).
    """
    # Obtener todos los segmentos de targets tocados
    segments = get_target_touch_intervals(trial, target_radius)

    correct_segments = []
    incorrect_segments = []
    expected_target_index = 0

    for segment in segments:
        target, start_cursor_info, end_cursor_info = segment

        # Verificamos si el target es el esperado
        if expected_target_index < len(trial.stimuli) and target == trial.stimuli[expected_target_index]:
            # Es el target correcto
            correct_segments.append(segment)
            expected_target_index += 1  # Pasamos al siguiente target esperado
        else:
            # Target tocado incorrectamente
            incorrect_segments.append(segment)

    return correct_segments, incorrect_segments


def number_of_correct_and_incorrect_targets_touched(trial: TMTTrial, target_radius: float) -> Tuple[int, int]:
    """
    Devuelve el número de segmentos de targets correctamente tocados e incorrectamente tocados.
    """
    correct_segments, incorrect_segments = get_correct_and_incorrect_target_touch_intervals(trial, target_radius)
    return len(correct_segments), len(incorrect_segments)
