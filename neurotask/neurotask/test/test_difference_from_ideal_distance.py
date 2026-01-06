import pytest

from neurotask.tmt.metrics.difference_from_ideal_distance import (
    DifferenceFromIdealDistance,
)
from neurotask.tmt.metrics.targets_touched import get_all_trails_between_targets
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTarget,
    TMTTrial,
    TrialType,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _compute_metrics(
        trial: TMTTrial,
        subject: TMTSubject,
        trails_between_targets: list,
        prefix: str = None
) -> dict:
    """
    Compute distance difference metrics using DifferenceFromIdealDistance.
    """
    calculator = DifferenceFromIdealDistance(prefix=prefix)
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

def test_straight_line_segment_returns_zero_difference():
    """
    Segmento en línea recta, diferencia debe ser 0.0.
    """
    # Straight line: (0,0) -> (5,0) -> (10,0)
    # Actual distance = 5 + 5 = 10
    # Ideal distance = 10 (straight line)
    # Difference = |10 - 10| = 0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(10.0, 0.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["distance_difference_from_ideal"] == pytest.approx(0.0)


def test_deviated_path_returns_correct_difference():
    """
    Segmento con desviación, verificar diferencia calculada correctamente.
    """
    # Path: (0,0) -> (3,0) -> (3,4) -> (0,4)
    # Actual distance = 3 + 4 + 3 = 10
    # Ideal distance = sqrt((0-0)² + (4-0)²) = 4
    # Difference = |10 - 4| = 6
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),   # distance = 3
        (3.0, 4.0, 2.0),   # distance = 4
        (0.0, 4.0, 3.0),   # distance = 3
    ])
    target = TMTTarget("1", Coordinate(0.0, 4.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Actual: 3 + 4 + 3 = 10, Ideal: 4, Difference: |10 - 4| = 6
    assert metrics["distance_difference_from_ideal"] == pytest.approx(6.0)


def test_multiple_segments_returns_average_difference():
    """
    Múltiples segmentos, verificar que retorna el promedio.
    """
    # Segment 1: (0,0) -> (5,0) -> (10,0) -> (15,0)
    # Actual: 5 + 5 + 5 = 15, Ideal: 15, Difference: 0
    segment1 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
        (15.0, 0.0, 3.0),
    ])
    
    # Segment 2: (0,0) -> (3,0) -> (3,4)
    # Actual: 3 + 4 = 7, Ideal: 5, Difference: |7 - 5| = 2
    segment2 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),
        (3.0, 4.0, 2.0),
    ])
    
    target1 = TMTTarget("1", Coordinate(15.0, 0.0))
    target2 = TMTTarget("2", Coordinate(3.0, 4.0))
    trails_between_targets = [
        (target1, segment1),
        (target2, segment2),
    ]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Average: (0 + 2) / 2 = 1.0
    assert metrics["distance_difference_from_ideal"] == pytest.approx(1.0)


def test_empty_trails_between_targets_raises_error():
    """
    Lista vacía debe lanzar ValueError con mensaje descriptivo.
    """
    trial, subject = build_trial_and_subject([])
    trails_between_targets = []
    
    with pytest.raises(ValueError, match="Cannot calculate distance_difference_from_ideal"):
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
    
    with pytest.raises(ValueError, match="Cannot calculate distance_difference_from_ideal"):
        _compute_metrics(trial, subject, trails_between_targets)


def test_segment_with_single_point():
    """
    Segmento con un solo punto (debe funcionar, diferencia = 0).
    """
    # Single point: actual distance = 0, ideal distance = 0, difference = 0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
    ])
    target = TMTTarget("1", Coordinate(0.0, 0.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["distance_difference_from_ideal"] == pytest.approx(0.0)


def test_segment_with_two_points():
    """
    Segmento con dos puntos en línea recta (diferencia = 0).
    """
    # Two points in straight line: (0,0) -> (5,0)
    # Actual: 5, Ideal: 5, Difference: 0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
    ])
    target = TMTTarget("1", Coordinate(5.0, 0.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["distance_difference_from_ideal"] == pytest.approx(0.0)


def test_returns_distance_difference_from_ideal_metric():
    """
    Verificar que retorna la métrica con la clave correcta.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(10.0, 0.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert "distance_difference_from_ideal" in metrics
    assert isinstance(metrics["distance_difference_from_ideal"], float)


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
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets, prefix="non_cut_")
    
    # Check that prefixed key exists
    assert "non_cut_distance_difference_from_ideal" in metrics
    
    # Check that non-prefixed key does not exist
    assert "distance_difference_from_ideal" not in metrics


def test_zigzag_segment_returns_correct_difference():
    """
    Segmento en zigzag, verificar diferencia correcta.
    """
    # Zigzag: (0,0) -> (1,1) -> (2,0) -> (3,1) -> (4,0)
    # Actual distance = 4 * sqrt(2) ≈ 5.657
    # Ideal distance = 4 (straight line from (0,0) to (4,0))
    # Difference = |5.657 - 4| ≈ 1.657
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),   # distance = sqrt(2)
        (2.0, 0.0, 2.0),   # distance = sqrt(2)
        (3.0, 1.0, 3.0),   # distance = sqrt(2)
        (4.0, 0.0, 4.0),   # distance = sqrt(2)
    ])
    target = TMTTarget("1", Coordinate(4.0, 0.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    actual_distance = 4 * (2.0 ** 0.5)  # 4 * sqrt(2)
    ideal_distance = 4.0
    expected_difference = abs(actual_distance - ideal_distance)
    
    assert metrics["distance_difference_from_ideal"] == pytest.approx(expected_difference)


def test_pythagorean_triangle_segment():
    """
    Segmento formando triángulo rectángulo, verificar diferencia.
    """
    # Right triangle: (0,0) -> (3,0) -> (3,4)
    # Actual distance = 3 + 4 = 7
    # Ideal distance = 5 (hypotenuse)
    # Difference = |7 - 5| = 2
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),   # distance = 3
        (3.0, 4.0, 2.0),   # distance = 4
    ])
    target = TMTTarget("1", Coordinate(3.0, 4.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Actual: 7, Ideal: 5, Difference: 2
    assert metrics["distance_difference_from_ideal"] == pytest.approx(2.0)


def test_returns_float_type():
    """
    Verificar que el resultado es siempre un float.
    """
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
    ])
    target = TMTTarget("1", Coordinate(10.0, 0.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert isinstance(metrics["distance_difference_from_ideal"], float)


def test_ideal_path_has_zero_difference():
    """
    Cuando el camino real es ideal, diferencia debe ser 0.
    """
    # Diagonal line: (0,0) -> (3,4) -> (6,8)
    # This is a straight line, so actual = ideal
    # Actual: sqrt(3²+4²) + sqrt(3²+4²) = 5 + 5 = 10
    # Ideal: sqrt(6²+8²) = 10
    # Difference: 0
    segment = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 1.0),   # distance = 5
        (6.0, 8.0, 2.0),   # distance = 5
    ])
    target = TMTTarget("1", Coordinate(6.0, 8.0))
    trails_between_targets = [(target, segment)]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    assert metrics["distance_difference_from_ideal"] == pytest.approx(0.0)


def test_mixed_valid_and_none_targets():
    """
    Mezcla de targets válidos y None, solo debe contar los válidos.
    """
    # Valid segment
    segment1 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
    ])
    
    # Segment with None target (should be skipped)
    segment2 = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 1.0),
    ])
    
    target1 = TMTTarget("1", Coordinate(5.0, 0.0))
    trails_between_targets = [
        (target1, segment1),
        (None, segment2),  # This should be skipped
    ]
    
    trial, subject = build_trial_and_subject([])
    
    metrics = _compute_metrics(trial, subject, trails_between_targets)
    
    # Should only calculate difference for segment1: |5 - 5| = 0
    assert metrics["distance_difference_from_ideal"] == pytest.approx(0.0)

