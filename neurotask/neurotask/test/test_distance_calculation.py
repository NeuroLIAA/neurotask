import pytest

from neurotask.tmt.metrics.distance_calculation import (
    TotalDistanceCalculator,
    calculate_total_distance,
)
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTrial,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject, prefix: str = None) -> dict:
    """
    Compute distance metrics using TotalDistanceCalculator.
    """
    calculator = TotalDistanceCalculator(prefix=prefix)
    return calculator.add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=[],
        calculate_crosses=False,
        speed_threshold=None,
        consecutive_points=None,
        target_radius_multiplier=1.0,
        crosses_time_threshold=500,
    )


# =============================================================================
# Tests
# =============================================================================

def test_straight_line_returns_correct_distance():
    """
    Trail recto horizontal/vertical, verificar distancia total.
    """
    # Horizontal line: 0,0 -> 5,0 -> 10,0 -> 15,0
    # Total distance = 5 + 5 + 5 = 15
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
        (15.0, 0.0, 3.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(15.0)


def test_multiple_points_returns_sum_of_distances():
    """
    Múltiples puntos con cambios de dirección, verificar suma correcta.
    """
    # Path: (0,0) -> (3,0) -> (3,4) -> (0,4)
    # Distances: 3 + 4 + 3 = 10
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),   # distance = 3
        (3.0, 4.0, 2.0),   # distance = 4
        (0.0, 4.0, 3.0),   # distance = 3
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(10.0)


def test_pythagorean_triangle_returns_correct_distance():
    """
    Trail formando un triángulo rectángulo (3-4-5), verificar distancia.
    """
    # Right triangle: (0,0) -> (3,0) -> (3,4)
    # Distances: sqrt(3²+0²) + sqrt(0²+4²) = 3 + 4 = 7
    # Note: This is NOT the hypotenuse (5), but the sum of the two legs
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),   # distance = 3
        (3.0, 4.0, 2.0),   # distance = 4
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(7.0)


def test_empty_trail_returns_zero():
    """
    Trail vacío debe retornar 0.0.
    """
    cursor_trail = []
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(0.0)


def test_single_point_returns_zero():
    """
    Un solo punto debe retornar 0.0 (no hay distancia entre puntos).
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(0.0)


def test_two_points_returns_distance():
    """
    Dos puntos, verificar distancia euclidiana correcta.
    """
    # Distance between (0,0) and (3,4) = sqrt(3² + 4²) = 5
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 1.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(5.0)


def test_with_custom_start_only_counts_from_start():
    """
    Con with_custom_start=True, solo debe contar distancia desde el punto de inicio.
    """
    # Full trail: (0,0) -> (5,0) -> (10,0) -> (15,0)
    # Start at (10,0), so only (10,0) -> (15,0) should count
    # Distance = 5
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),   # Before start
        (5.0, 0.0, 1.0),   # Before start
        (10.0, 0.0, 2.0),  # Start point
        (15.0, 0.0, 3.0),  # After start
    ])
    
    start_point = CursorInfo(Coordinate(10.0, 0.0), 2.0)
    trial, subject = build_trial_and_subject(
        cursor_trail,
        with_custom_start=True,
        start=start_point,
    )

    metrics = _compute_metrics(trial, subject)

    # Should only count distance from start: (10,0) -> (15,0) = 5
    assert metrics["total_distance"] == pytest.approx(5.0)


def test_returns_total_distance_metric():
    """
    Verificar que retorna la métrica con la clave correcta (total_distance).
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 2.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert "total_distance" in metrics
    assert isinstance(metrics["total_distance"], float)


def test_with_prefix_adds_prefix_to_metric_key():
    """
    Con prefijo, la clave debe tener el prefijo (ej: non_cut_total_distance).
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 1.0),
        (4.0, 0.0, 2.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject, prefix="non_cut_")

    # Check that prefixed key exists
    assert "non_cut_total_distance" in metrics

    # Check that non-prefixed key does not exist
    assert "total_distance" not in metrics


def test_zigzag_path_returns_correct_distance():
    """
    Trail con movimiento en zigzag, verificar suma correcta.
    """
    # Zigzag: (0,0) -> (1,1) -> (2,0) -> (3,1) -> (4,0)
    # Distances: sqrt(2) + sqrt(2) + sqrt(2) + sqrt(2) = 4*sqrt(2) ≈ 5.657
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),   # distance = sqrt(2)
        (2.0, 0.0, 2.0),   # distance = sqrt(2)
        (3.0, 1.0, 3.0),   # distance = sqrt(2)
        (4.0, 0.0, 4.0),   # distance = sqrt(2)
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    expected_distance = 4 * (2.0 ** 0.5)  # 4 * sqrt(2)
    assert metrics["total_distance"] == pytest.approx(expected_distance)


def test_returns_float_type():
    """
    Verificar que el resultado es siempre un float.
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert isinstance(metrics["total_distance"], float)


def test_vertical_line_returns_correct_distance():
    """
    Trail vertical, verificar distancia total.
    """
    # Vertical line: (0,0) -> (0,5) -> (0,10) -> (0,15)
    # Total distance = 5 + 5 + 5 = 15
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (0.0, 5.0, 1.0),
        (0.0, 10.0, 2.0),
        (0.0, 15.0, 3.0),
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(15.0)


def test_diagonal_line_returns_correct_distance():
    """
    Trail diagonal, verificar distancia total usando distancia euclidiana.
    """
    # Diagonal: (0,0) -> (3,4) -> (6,8)
    # Distance 1: sqrt(3² + 4²) = 5
    # Distance 2: sqrt(3² + 4²) = 5
    # Total = 10
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 1.0),   # distance = 5
        (6.0, 8.0, 2.0),   # distance = 5
    ])
    trial, subject = build_trial_and_subject(cursor_trail)

    metrics = _compute_metrics(trial, subject)

    assert metrics["total_distance"] == pytest.approx(10.0)

