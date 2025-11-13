from typing import List, Tuple, Optional

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from .distance_calculation import calculate_distance
from ..model.tmt_model import TMTTrial, TMTTarget, CursorInfo, TMTSubject


class TargetsTouchesCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals) -> dict:
        correct_touches = count_correctly_touched_targets(
            trial, subject.target_radius
        )

        metrics[self.get_metric_name('correct_targets_touches')] = correct_touches
        metrics[self.get_metric_name('wrong_targets_touches')] = count_incorrect_touches(trial, subject.target_radius)

        return metrics


def touched_targets_for_every_cursor_point(trial: TMTTrial, target_radius: float) -> List[
    Tuple[List[TMTTarget], CursorInfo]]:
    """
    Returns a list of tuples, where each tuple contains a list  of targets touched at that cursor point (overlapping)
    and the cursor info.
    If no target is touched, the list will be empty.
    """
    trail_with_targets = []

    for cursor_info in trial.get_cursor_trail_from_start():
        touched_target_list = get_touched_target_list(cursor_info, target_radius, trial)
        trail_with_targets.append((touched_target_list, cursor_info))

    return trail_with_targets


def get_touched_target_list(cursor_info: CursorInfo, target_radius: float, trial: TMTTrial) -> List[TMTTarget]:
    """
    Returns the list of targets that are touched by the cursor at the given cursor_info.
    """
    targets = []
    for target in trial.stimuli:
        distance = calculate_distance(target.position, cursor_info.position)
        if distance < target_radius:
            targets.append(target)

    return targets


def correct_touched_targets_for_every_cursor_point(
        trial: TMTTrial,
        target_radius: float
) -> List[Tuple[Optional[TMTTarget], CursorInfo]]:
    """
    Returns a list of tuples for each cursor point, where each tuple contains:
      - The correct target if the expected target is touched at that point (None otherwise)
      - The cursor info

    A target is considered "correct" if it's the next expected target in the sequence.
    Once a target is correctly touched, the next target in trial.stimuli becomes expected.

    :param trial: instancia de TMTTrial
    :param target_radius: radio para detección de toques
    :return: lista de (Optional[TMTTarget], CursorInfo) para cada punto de cursor
    """
    # Get all touched targets for every cursor point
    trail_with_targets = touched_targets_for_every_cursor_point(trial, target_radius)

    # Result list
    result: List[Tuple[Optional[TMTTarget], CursorInfo]] = []

    expected_idx = 0

    for touched_list, cursor_info in trail_with_targets:
        # Check if we've already touched all targets
        if expected_idx >= len(trial.stimuli):
            result.append((None, cursor_info))
            continue

        expected = trial.stimuli[expected_idx]

        # Check if the expected target is in the touched list
        if expected in touched_list:
            result.append((expected, cursor_info))
            expected_idx += 1
        else:
            result.append((None, cursor_info))

    return result


def count_correctly_touched_targets(
        trial: TMTTrial,
        target_radius: float
) -> int:
    """
    Calcula cuántos targets de 'trial.stimuli' fueron efectivamente tocados
    (en el orden esperado) dentro del radio dado.

    :param trial:       instancia de TMTTrial con su lista de estímulos.
    :param target_radius: radio para detección de toques.
    :return: número de targets tocados correctamente.
    """
    # Obtenemos la lista de targets correctos para cada punto de cursor
    correct_touches = correct_touched_targets_for_every_cursor_point(trial, target_radius)

    # Contamos los targets únicos que fueron tocados correctamente (no None)
    # Usamos una lista para evitar problemas con TMTTarget que no es hashable
    touched_targets = []
    for target, cursor_info in correct_touches:
        if target is not None and target not in touched_targets:
            touched_targets.append(target)

    return len(touched_targets)


def get_incorrect_touches(
        trial: TMTTrial,
        target_radius: float
) -> List[CursorInfo]:
    """
    Identifica los puntos de cursor donde ocurren toques erróneos,
    siguiendo las reglas acordadas.

    :return: lista de CursorInfo incorrectos
    """
    # 1. Obtenemos la secuencia (touched_list, cursor_info)
    trail = touched_targets_for_every_cursor_point(trial, target_radius)

    stim_iter = iter(trial.stimuli)
    expected = next(stim_iter, None)
    previous = None

    incorrect_points: List[CursorInfo] = []
    in_error = False  # true mientras permanezco en un "estado de error"

    # 3. Recorremos cada punto de cursor
    for touched_list, cursor_info in trail:

        # 1) Acierto del esperado → avanzar estado y salir de error
        if expected in touched_list:
            previous = expected
            expected = next(stim_iter, None)
            in_error = False
            continue

        # 3.2 Si no tocó ningún target → ignoramos
        if not touched_list:
            in_error = False
            continue

        # 3) ¿Algún toque se "salva" por solapar con previous or expected?
        def overlaps_prev_or_expected(t):
            overlapping = get_overlapping_targets(t, trial, target_radius)
            return (previous in overlapping) or (expected in overlapping)

        actual_in_error = not any(overlaps_prev_or_expected(t) for t in touched_list)

        # 4) Añadir solo al entrar en error
        if actual_in_error and not in_error:
            incorrect_points.append(cursor_info)
            in_error = True
        elif not actual_in_error:
            in_error = False

    return incorrect_points


def count_incorrect_touches(
        trial: TMTTrial,
        target_radius: float
) -> int:
    """
    Cuenta globalmente la cantidad de toques erróneos,
    siguiendo las reglas acordadas.

    :return: número de toques erróneos
    """
    incorrect_points = get_incorrect_touches(trial, target_radius)
    return len(incorrect_points)


def get_overlapping_targets(
        target: TMTTarget,
        trial: TMTTrial,
        target_radius: float
) -> List[TMTTarget]:
    """
    Recorre todos los targets en `trial.stimuli` y devuelve la lista de
    aquellos que geométricamente solapan con `target`, considerando
    que dos targets solapan si la distancia entre sus centros es
    menor o igual a 2 * target_radius.
    """
    overlapping: List[TMTTarget] = []
    for t in trial.stimuli:
        # calculamos la distancia entre los centros
        dist = calculate_distance(target.position, t.position)
        if dist <= 2 * target_radius:
            overlapping.append(t)
    return overlapping


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
    # Secuencia (lista_de_targets, cursor_info) para cada punto de cursor
    trail_with_targets = touched_targets_for_every_cursor_point(trial, target_radius)

    # Iterador único sobre la secuencia de toques/puntos
    trail_iter = iter(trail_with_targets)

    # Para cada target esperado, vamos construyendo su segmento
    for expected in trial.stimuli[1:]:
        current_segment: List[CursorInfo] = []
        for touched_list, cursor_info in trail_iter:
            current_segment.append(cursor_info)
            # Si el target esperado está en la lista de tocados, lo damos por válido
            if expected in touched_list:
                segments.append((expected, current_segment.copy()))
                break
        else:
            # No encontramos el target esperado en lo que queda de trail
            break

    return segments


def get_all_intervals_between_targets(
        trial: TMTTrial,
        target_radius: float
) -> List[Tuple[TMTTarget, CursorInfo, CursorInfo]]:
    """
    Utiliza get_all_trails_between_targets para devolver, por cada target tocado
    correctamente, una tupla:
      (target, inicio_del_trail, fin_del_trail)

    Donde:
      - inicio_del_trail  = primer CursorInfo del segmento hacia ese target
      - fin_del_trail     = último CursorInfo (justo en el toque del target)

    :param trial:         instancia de TMTTrial con su lista de estímulos.
    :param target_radius: radio para detección de toques.
    :return: lista de (target, CursorInfo_inicio, CursorInfo_fin)
    """
    # 1) Obtengo los segmentos con la función existente
    segments: List[Tuple[TMTTarget, List[CursorInfo]]] = get_all_trails_between_targets(trial, target_radius)

    # 2) Convierto cada segmento en (target, inicio, fin)
    intervals: List[Tuple[TMTTarget, CursorInfo, CursorInfo]] = []
    for target, cursor_segment in segments:
        start_info = cursor_segment[0]
        end_info = cursor_segment[-1]
        intervals.append((target, start_info, end_info))

    return intervals
