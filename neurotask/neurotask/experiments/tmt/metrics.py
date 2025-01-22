import logging
import math
from typing import List, Tuple, Optional

from neurotask.experiments.tmt.model.tmt_model import Coordinate, TMTTrial, TMTTarget, CursorInfo


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


def calculate_speed(current_cursor: CursorInfo, previous_cursor: CursorInfo) -> float:
    if current_cursor.time <= previous_cursor.time:
        raise ValueError("current_cursor.time must be greater than previous_cursor.time")
    distance = calculate_distance(current_cursor.position, previous_cursor.position)
    time = current_cursor.time - previous_cursor.time
    return distance / time


def calculate_acceleration(current_speed: float, previous_speed: float, current_time: float,
                           previous_time: float) -> float:
    if current_time <= previous_time:
        raise ValueError("current_time must be greater than previous_time")
        # TODO GIAN VER en old tmt

    time_diff = current_time - previous_time

    speed_diff = current_speed - previous_speed
    acceleration = speed_diff / time_diff
    return acceleration


def calculate_speeds_between_cursor_positions(trial: TMTTrial) -> List[float]:
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    return calculate_speeds(cursor_trail_from_first_click)


def calculate_speeds(cursor_trail: List[CursorInfo]) -> List[float]:
    if len(cursor_trail) < 2:
        raise ValueError("At least two points are required to calculate velocity")

    speeds = []

    for i in range(1, len(cursor_trail)):
        current_cursor = cursor_trail[i]
        previous_cursor = cursor_trail[i - 1]
        speed = calculate_speed(current_cursor, previous_cursor)
        if speed > 50:  # TODO GIAN PROBAR, ver porque pasa esto seguro es el sampling rate
            logging.warning(f"Speed value of {speed} detected. This may be an error.")
            raise ValueError(f"Speed value of {speed} detected. This may be an error.")
        speeds.append(speed)

    return speeds


def calculate_accelerations_between_cursor_positions(trial: TMTTrial) -> List[float]:
    """
    Calcula la aceleración entre cada posición del cursor.

    Returns:
    - Lista de aceleraciones entre puntos consecutivos.
    """
    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()

    if len(cursor_trail_from_first_click) < 3:
        print(cursor_trail_from_first_click)
        raise ValueError("At least three points are required to calculate acceleration")

    accelerations = []

    # Calculamos las velocidades primero
    speeds = calculate_speeds_between_cursor_positions(trial)

    # Ahora calculamos la aceleración entre las velocidades
    for i in range(1, len(speeds)):
        current_cursor = cursor_trail_from_first_click[i + 1]  # i+1 porque estamos viendo del tercer punto en adelante
        previous_cursor = cursor_trail_from_first_click[i]
        current_speed = speeds[i]
        previous_speed = speeds[i - 1]

        acceleration = calculate_acceleration(
            current_speed, previous_speed, current_cursor.time, previous_cursor.time
        )
        accelerations.append(acceleration)

    return accelerations


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


def get_target_touch_segments(trial: TMTTrial, target_radius: float) -> List[Tuple[TMTTarget, CursorInfo, CursorInfo]]:
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


def get_correct_and_incorrect_segments(trial: TMTTrial, target_radius: float) -> Tuple[
    List[Tuple[TMTTarget, CursorInfo, CursorInfo]],
    List[Tuple[TMTTarget, CursorInfo, CursorInfo]]
]:
    """
    Devuelve dos listas:
    1. Los segmentos de targets correctamente tocados (en orden).
    2. Los segmentos de targets incorrectamente tocados (fuera de orden).
    """
    # Obtener todos los segmentos de targets tocados
    segments = get_target_touch_segments(trial, target_radius)

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
    correct_segments, incorrect_segments = get_correct_and_incorrect_segments(trial, target_radius)
    return len(correct_segments), len(incorrect_segments)
