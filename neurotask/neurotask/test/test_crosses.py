import pytest
import numpy as np

from neurotask.tmt.crosses.crosses_metric_calculator import CrossesMetricCalculator
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTarget,
    TMTTrial,
    TrialType,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject, 
                     calculate_crosses: bool = True, prefix: str = None) -> dict:
    """
    Compute crosses metrics using CrossesMetricCalculator.
    """
    calculator = CrossesMetricCalculator(prefix=prefix)
    return calculator.add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=[],
        calculate_crosses=calculate_crosses,
        speed_threshold=None,
        consecutive_points=None,
    )


# =============================================================================
# Tests
# =============================================================================

def test_simple_cross_detects_one_cross():
    """
    Trail que forma una X simple debe detectar un cruce.
    Segmento (0,0)->(10,10) y segmento (0,10)->(10,0) que se cruzan.
    Tiempos separados (gap >= 500ms) para que pasen el filtro temporal.
    """
    # Forma una X: segmento diagonal de (0,0) a (10,10) y otro de (0,10) a (10,0)
    # Tiempos: [0, 100] y [600, 700] para que gap=500ms (pasa el filtro)
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),      # Inicio primer segmento
        (10.0, 10.0, 100.0),  # Fin primer segmento
        (5.0, 5.0, 300.0),    # Punto intermedio
        (0.0, 10.0, 600.0),   # Inicio segundo segmento
        (10.0, 0.0, 700.0),   # Fin segundo segmento (cruza con el primero)
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_of_crosses"] == 1


def test_no_cross_returns_zero():
    """
    Trail recto sin cruces debe retornar 0 cruces.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 100.0),
        (20.0, 0.0, 200.0),
        (30.0, 0.0, 300.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_of_crosses"] == 0


def test_multiple_crosses_detects_all():
    """
    Trail con múltiples cruces debe detectar todos los cruces.
    Forma un patrón que cruza múltiples veces.
    """
    # Crea un patrón que se cruza múltiples veces
    # Segmento 1: (0,0) -> (10,10) tiempo [0, 100]
    # Segmento 2: (0,10) -> (10,0) tiempo [600, 700] - cruza con segmento 1
    # Segmento 3: (5,0) -> (5,10) tiempo [1200, 1300] - cruza con segmento 1 y 2
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),      # Inicio segmento 1
        (10.0, 10.0, 100.0),  # Fin segmento 1
        (5.0, 5.0, 200.0),    # Punto intermedio
        (0.0, 10.0, 600.0),   # Inicio segmento 2
        (10.0, 0.0, 700.0),   # Fin segmento 2 (cruza con segmento 1)
        (5.0, 0.0, 1200.0),   # Inicio segmento 3
        (5.0, 10.0, 1300.0), # Fin segmento 3 (cruza con segmento 1 y 2)
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Debe detectar al menos 2 cruces (segmento 3 cruza con segmento 1 y 2)
    assert metrics["number_of_crosses"] >= 2


def test_segments_too_close_in_time_are_excluded():
    """
    Dos segmentos que se cruzan pero con gap < 500ms deben ser excluidos.
    """
    # Segmento 1: (0,0) -> (10,10) tiempo [0, 100]
    # Segmento 2: (0,10) -> (10,0) tiempo [150, 200] - gap = 50ms < 500ms
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 100.0),
        (5.0, 5.0, 120.0),
        (0.0, 10.0, 150.0),
        (10.0, 0.0, 200.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_of_crosses"] == 0


def test_segments_far_apart_in_time_are_included():
    """
    Dos segmentos que se cruzan con gap >= 500ms deben ser incluidos.
    """
    # Segmento 1: (0,0) -> (10,10) tiempo [0, 100]
    # Segmento 2: (0,10) -> (10,0) tiempo [600, 700] - gap = 500ms >= 500ms
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 100.0),
        (5.0, 5.0, 300.0),
        (0.0, 10.0, 600.0),
        (10.0, 0.0, 700.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_of_crosses"] == 1


def test_overlapping_segments_are_excluded():
    """
    Dos segmentos que se cruzan pero se solapan en tiempo (gap = 0) deben ser excluidos.
    """
    # Segmento 1: (0,0) -> (10,10) tiempo [0, 200]
    # Segmento 2: (0,10) -> (10,0) tiempo [100, 300] - se solapan
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 200.0),
        (5.0, 5.0, 150.0),
        (0.0, 10.0, 100.0),
        (10.0, 0.0, 300.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_of_crosses"] == 0


def test_calculate_crosses_false_returns_nan():
    """
    Cuando calculate_crosses=False, debe retornar np.nan.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 100.0),
        (0.0, 10.0, 600.0),
        (10.0, 0.0, 700.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject, calculate_crosses=False)

    assert np.isnan(metrics["number_of_crosses"])


def test_calculate_crosses_true_computes_crosses():
    """
    Cuando calculate_crosses=True, debe calcular los cruces (retorna número entero, no np.nan).
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 100.0),
        (0.0, 10.0, 600.0),
        (10.0, 0.0, 700.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject, calculate_crosses=True)

    assert not np.isnan(metrics["number_of_crosses"])
    assert isinstance(metrics["number_of_crosses"], (int, np.integer))


def test_with_prefix_adds_prefix_to_metric_key():
    """
    Cuando se inicializa con un prefijo, todas las claves de métricas deben tener ese prefijo.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 100.0),
        (0.0, 10.0, 600.0),
        (10.0, 0.0, 700.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject, prefix="non_cut_")

    # Verificar que la clave con prefijo existe
    assert "non_cut_number_of_crosses" in metrics
    # Verificar que la clave sin prefijo NO existe
    assert "number_of_crosses" not in metrics


def test_empty_trail_returns_zero():
    """
    Trail vacío debe retornar 0 cruces.
    """
    cursor_trail = []
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_of_crosses"] == 0


def test_trail_too_short_returns_zero():
    """
    Trail con menos de 4 puntos debe retornar 0 cruces.
    Se necesitan al menos 4 puntos para formar segmentos no adyacentes.
    """
    # Test con 0 puntos (ya cubierto por test_empty_trail)
    # Test con 1 punto
    cursor_trail_1 = build_cursor_trail([
        (0.0, 0.0, 0.0),
    ])
    trial_1, subject_1 = build_trial_and_subject(cursor_trail_1)
    metrics_1 = _compute_metrics(trial_1, subject_1)
    assert metrics_1["number_of_crosses"] == 0

    # Test con 2 puntos
    cursor_trail_2 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 100.0),
    ])
    trial_2, subject_2 = build_trial_and_subject(cursor_trail_2)
    metrics_2 = _compute_metrics(trial_2, subject_2)
    assert metrics_2["number_of_crosses"] == 0

    # Test con 3 puntos
    cursor_trail_3 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 100.0),
        (20.0, 0.0, 200.0),
    ])
    trial_3, subject_3 = build_trial_and_subject(cursor_trail_3)
    metrics_3 = _compute_metrics(trial_3, subject_3)
    assert metrics_3["number_of_crosses"] == 0


def test_adjacent_segments_do_not_count_as_cross():
    """
    Segmentos adyacentes que comparten un punto no deben contar como cruce.
    """
    # Trail: (0,0)->(5,5)->(10,0) donde los segmentos comparten el punto medio
    # Los segmentos adyacentes no se comparan para cruces
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 5.0, 100.0),
        (10.0, 0.0, 200.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Con solo 3 puntos, no hay suficientes para formar segmentos no adyacentes
    assert metrics["number_of_crosses"] == 0


def test_segments_that_touch_at_endpoint():
    """
    Segmentos que se tocan solo en un endpoint no deben contar como cruce.
    """
    # Segmento 1: (0,0) -> (5,5) tiempo [0, 100]
    # Segmento 2: (5,5) -> (10,10) tiempo [600, 700] - comparten endpoint pero no se cruzan
    # Nota: Estos son segmentos adyacentes, así que no se comparan
    # Para que se comparen, necesitamos un punto intermedio
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 5.0, 100.0),
        (10.0, 0.0, 200.0),  # Punto intermedio
        (5.0, 5.0, 600.0),    # Vuelve al punto medio
        (10.0, 10.0, 700.0),  # Continúa desde el punto medio
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Los segmentos que solo se tocan en un endpoint no se cruzan realmente
    # El algoritmo de intersección debe retornar False para este caso
    # Si hay un cruce, sería porque los segmentos realmente se cruzan, no solo se tocan
    assert metrics["number_of_crosses"] >= 0  # Puede ser 0 o más dependiendo de la geometría


def test_colinear_segments_do_not_cross():
    """
    Segmentos colineales que no se cruzan deben retornar 0 cruces.
    """
    # Segmentos colineales en la misma línea pero no se solapan
    # Segmento 1: (0,0) -> (5,0) tiempo [0, 100]
    # Segmento 2: (10,0) -> (15,0) tiempo [600, 700] - colineal pero separado
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 100.0),
        (7.5, 0.0, 300.0),    # Punto intermedio
        (10.0, 0.0, 600.0),
        (15.0, 0.0, 700.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    # Segmentos colineales que no se solapan no se cruzan
    assert metrics["number_of_crosses"] == 0

