import pytest

from neurotask.tmt.metrics.area_calculation import (
    DifferenceFromIdealArea,
)
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTarget,
    TMTTrial,
    TrialType,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _build_trails_between_targets(
        targets_and_segments: list[tuple[TMTTarget, list[CursorInfo]]]
) -> list[tuple[TMTTarget, list[CursorInfo]]]:
    """
    Build trails_between_targets manually for testing.
    """
    return targets_and_segments


def _compute_metrics(
        trial: TMTTrial,
        subject: TMTSubject,
        trails_between_targets: list,
        prefix: str = None
) -> dict:
    """
    Compute area difference metrics using DifferenceFromIdealArea.
    """
    calculator = DifferenceFromIdealArea(prefix=prefix)
    return calculator.add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=trails_between_targets,
        calculate_crosses=False,
        speed_threshold=None,
        consecutive_points=None,
    )


# =============================================================================
# Tests
# =============================================================================

def test_straight_line_segment_returns_zero_area():
    """
    Segmento en línea recta debe retornar área 0.
    """
    # Straight line: (0,0) -> (5,0) -> (10,0)
    # All points are colinear, so area = 0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(10.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["area_difference_from_ideal"] == pytest.approx(0.0, abs=1e-8)


def test_deviated_path_returns_correct_area():
    """
    Segmento con desviación debe calcular área correcta.
    """
    # Path: (0,0) -> (1,1) -> (2,0)
    # This forms a triangle deviation from baseline (0,0) -> (2,0)
    # Known area from test_area.py: 1.0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(2.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Known value from test_area.py: triangle deviation = 1.0
    assert metrics["area_difference_from_ideal"] == pytest.approx(1.0, rel=1e-12)


def test_multiple_segments_returns_average_area():
    """
    Múltiples segmentos deben retornar el promedio.
    """
    # Segment 1: Straight line (area = 0)
    segment1 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    
    # Segment 2: Triangle deviation (area = 1.0)
    segment2 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 0.0, 2.0),
    ])
    
    target1 = TMTTarget("1", Coordinate(10.0, 0.0))
    target2 = TMTTarget("2", Coordinate(2.0, 0.0))
    trails_between_targets = _build_trails_between_targets([
        (target1, segment1),
        (target2, segment2),
    ])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Average: (0 + 1.0) / 2 = 0.5
    assert metrics["area_difference_from_ideal"] == pytest.approx(0.5, rel=1e-12)


def test_triangle_deviation_returns_correct_area():
    """
    Segmento con desviación triangular debe retornar área conocida.
    """
    # Triangle: (0,0) -> (1,1) -> (2,0)
    # Known area from test_area.py: 1.0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(2.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["area_difference_from_ideal"] == pytest.approx(1.0, rel=1e-12)


def test_empty_trails_between_targets_raises_error():
    """
    Lista vacía debe lanzar ValueError con mensaje descriptivo.
    """
    trial, subject = build_trial_and_subject([])
    trails_between_targets = []
    
    with pytest.raises(ValueError, match="Cannot calculate area_difference_from_ideal"):
        _compute_metrics(trial, subject, trails_between_targets)


def test_all_targets_none_raises_error():
    """
    Todos los targets son None, debe lanzar ValueError.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
    ])
    trails_between_targets = [(None, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    with pytest.raises(ValueError, match="Cannot calculate area_difference_from_ideal"):
        _compute_metrics(trial, subject, trails_between_targets)


def test_segment_with_single_point():
    """
    Segmento con un solo punto debe retornar 0.0.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
    ])
    target = TMTTarget("1", Coordinate(0.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["area_difference_from_ideal"] == pytest.approx(0.0)


def test_segment_with_two_points():
    """
    Segmento con dos puntos debe retornar 0.0 (no hay desviación).
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
    ])
    target = TMTTarget("1", Coordinate(5.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["area_difference_from_ideal"] == pytest.approx(0.0)


def test_returns_area_difference_from_ideal_metric():
    """
    Verificar que retorna la métrica con la clave correcta.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(10.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert "area_difference_from_ideal" in metrics
    assert isinstance(metrics["area_difference_from_ideal"], float)


def test_with_prefix_adds_prefix_to_metric_key():
    """
    Con prefijo, la clave debe tener el prefijo.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(10.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets, prefix="non_cut_")
    
    # Check that prefixed key exists
    assert "non_cut_area_difference_from_ideal" in metrics
    
    # Check that non-prefixed key does not exist
    assert "area_difference_from_ideal" not in metrics


def test_returns_float_type():
    """
    Verificar que el resultado es siempre un float.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(2.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert isinstance(metrics["area_difference_from_ideal"], float)


def test_square_deviation_returns_correct_area():
    """
    Segmento con desviación cuadrada debe retornar área conocida.
    """
    # Square deviation: (0,0) -> (0,2) -> (2,2) -> (2,0)
    # Known area from test_area.py: 4.0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 1.0),
        (2.0, 2.0, 2.0),
        (2.0, 0.0, 3.0),
    ])
    target = TMTTarget("1", Coordinate(2.0, 0.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Known value from test_area.py: square deviation = 4.0
    assert metrics["area_difference_from_ideal"] == pytest.approx(4.0, rel=1e-12)


def test_colinear_points_returns_zero():
    """
    Puntos colineales deben retornar área 0.
    """
    # Colinear points: (0,0) -> (1,1) -> (2,2)
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(2.0, 2.0))
    trails_between_targets = _build_trails_between_targets([(target, segment)])
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["area_difference_from_ideal"] == pytest.approx(0.0, abs=1e-8)


def test_mixed_valid_and_none_targets():
    """
    Mezcla de targets válidos y None, solo debe contar los válidos.
    """
    # Valid segment (triangle, area = 1.0)
    segment1 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 0.0, 2.0),
    ])
    
    # Segment with None target (should be skipped)
    segment2 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
    ])
    
    target1 = TMTTarget("1", Coordinate(2.0, 0.0))
    trails_between_targets = [
        (target1, segment1),
        (None, segment2),  # This should be skipped
    ]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Should only count segment1: area = 1.0
    assert metrics["area_difference_from_ideal"] == pytest.approx(1.0, rel=1e-12)

