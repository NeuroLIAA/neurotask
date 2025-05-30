from typing import List, Tuple

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from .distance_calculation import calculate_distance
from ..model.tmt_model import TMTTrial, TMTTarget, CursorInfo, TMTSubject


class TargetsTouchesCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals, wrong_intervals) -> dict:
        correct_touches, wrong_touches = number_of_correct_and_incorrect_targets_touched(
            trial, subject.target_radius
        )

        metrics[self.get_metric_name('correct_targets_touches')] = correct_touches
        metrics[self.get_metric_name('wrong_targets_touches')] = wrong_touches

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
    # Obtenemos todos los segmentos hasta el contacto de cada target esperado
    segments: List[Tuple[TMTTarget, List[CursorInfo]]] = get_all_trails_between_targets(
        trial,
        target_radius
    )
    # La cantidad de segmentos coincide con la cantidad de targets tocados
    return len(segments) + 1


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


def get_target_intervals(
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


def get_correct_and_incorrect_target_touch_intervals(trial: TMTTrial, target_radius: float) -> Tuple[
    List[Tuple[TMTTarget, CursorInfo, CursorInfo]],
    List[Tuple[TMTTarget, CursorInfo, CursorInfo]]
]:
    return get_target_intervals(trial, target_radius), None


def number_of_correct_and_incorrect_targets_touched(trial: TMTTrial, target_radius: float) -> Tuple[int, int]:
    """
    Devuelve el número de segmentos de targets correctamente tocados e incorrectamente tocados.
    """
    correct = count_correctly_touched_targets(trial, target_radius)
    # TODO GIAN
    return correct, 0
