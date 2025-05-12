from typing import List, Tuple, Optional

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from .distance_calculation import calculate_distance
from ..model.tmt_model import TMTTrial, TMTTarget, CursorInfo


class TargetsTouchesCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial: TMTTrial, **params):
        if 'correct_targets_touches' not in params:
            raise ValueError("correct_targets must be provided")

        if 'wrong_targets_touches' not in params:
            raise ValueError("wrong_targets must be provided")

        metrics['correct_targets_touches'] = params.get('correct_targets_touches', 0)
        metrics['wrong_targets_touches'] = params.get('wrong_targets_touches', 0)

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


def get_all_trails_between_targets(trial: TMTTrial, target_radius: float) -> List[Tuple[TMTTarget, List[CursorInfo]]]:
    all_trails = []
    trail_with_targets = touched_targets_for_every_cursor_point(trial, target_radius)

    current_trail: List[CursorInfo] = []
    expected_target_index = 1

    for touched_target, cursor_info in trail_with_targets:
        current_trail.append(cursor_info)
        if expected_target_index < len(trial.stimuli) and touched_target == trial.stimuli[expected_target_index]:
            # Si el target tocado es el esperado, lo añadimos a la lista de trails
            all_trails.append((touched_target, current_trail))
            expected_target_index += 1
            current_trail = []  # Reiniciamos el trail para el siguiente target


    return all_trails


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


def number_of_correct_and_incorrect_segments(trial: TMTTrial, target_radius: float) -> Tuple[int, int]:
    """
    Devuelve el número de segmentos de targets correctamente tocados e incorrectamente tocados.
    """
    correct_segments, incorrect_segments = get_correct_and_incorrect_target_touch_intervals(trial, target_radius)
    return len(correct_segments), len(incorrect_segments)
