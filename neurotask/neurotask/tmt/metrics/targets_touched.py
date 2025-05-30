from typing import List, Tuple

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



# INCORRECT TARGETS Touched


def count_incorrect_touches(
    trial: TMTTrial,
    target_radius: float
) -> int:
    """
    Cuenta globalmente la cantidad de toques erróneos,
    siguiendo las reglas acordadas.
    """
    # 1. Obtenemos la secuencia (touched_list, cursor_info)
    trail = touched_targets_for_every_cursor_point(trial, target_radius)

    # 2. Inicializamos índices y estados
    expected_idx = 1
    previous = trial.stimuli[0]
    expected = trial.stimuli[expected_idx]
    error_count = 0
    prev_was_error = False

    # 3. Recorremos cada punto de cursor
    for touched_list, cursor_info in trail:

        # 3.1 Si tocó el siguiente esperado → avanzamos, reseteamos prev_was_error
        if expected in touched_list:
            expected_idx += 1
            previous = expected
            expected = (trial.stimuli[expected_idx]
                        if expected_idx < len(trial.stimuli)
                        else None)
            prev_was_error = False
            continue

        # 3.2 Si no tocó ningún target → ignoramos
        if not touched_list:
            prev_was_error = False
            continue

        # 3.3 Chequeo de errores en este punto:
        #     - Para cada t en touched_list:
        #         * Obtenemos los targets que solapan con t
        #         * Si alguno de ellos ES expected o ES previous → ¡no es error!
        #     - Si ninguno cumple → es error (contamos sólo una vez por punto)
        is_error = True
        for t in touched_list:
            # aquí necesitamos una función auxiliar overlap(t) → List[TMTTarget]
            solapados = get_overlapping_targets(t, trial, target_radius)
            if previous in solapados or expected in solapados:
                is_error = False
                break

        # 3.4 Contar el error sólo al “entrar” en un estado erróneo
        if is_error and not prev_was_error:
            error_count += 1
            prev_was_error = True
        elif not is_error:
            prev_was_error = False

    return error_count



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