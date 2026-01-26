import pytest

from neurotask.tmt.segmentation.segmentation_metric import SegmentationMetricCalculator
from neurotask.tmt.model.tmt_model import (
    TMTSubject,
    TMTTrial,
    TMTTarget,
    Coordinate,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial_and_subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject, speed_threshold: float,
                     consecutive_points: int, prefix: str = None) -> dict:
    """
    Compute segmentation metrics using SegmentationMetricCalculator.

    :param trial: TMTTrial object
    :param subject: TMTSubject object
    :param speed_threshold: Speed threshold for segmentation (must be positive)
    :param consecutive_points: Number of consecutive points for state transitions (must be positive)
    :param prefix: Optional prefix for metric names
    :return: Dictionary of segmentation metrics
    """
    calculator = SegmentationMetricCalculator(prefix=prefix)
    return calculator.add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=[],
        calculate_crosses=False,
        speed_threshold=speed_threshold,
        consecutive_points=consecutive_points,
        target_radius_multiplier=1.0,
        crosses_time_threshold=500,
    )


def _build_simple_trial_and_subject():
    """Helper to create a simple trial and subject for basic tests."""
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
        (5.0, 0.0, 2.0),
        (10.0, 0.0, 3.0),
        (10.5, 0.0, 4.0),
        (15.0, 0.0, 5.0),
    ])
    targets = [
        TMTTarget("1", Coordinate(0.0, 0.0)),
        TMTTarget("2", Coordinate(10.0, 0.0)),
    ]
    return build_trial_and_subject(
        cursor_trail=cursor_trail,
        targets=targets,
        target_radius=1.0,
    )


# =============================================================================
# Tests de estructura básica
# =============================================================================

def test_returns_all_expected_metrics():
    """Verifica que las 15 métricas esperadas estén presentes."""
    trial, subject = _build_simple_trial_and_subject()
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2)
    
    expected_keys = [
        "hesitation_time",
        "travel_time",
        "search_time",
        "hesitation_distance",
        "travel_distance",
        "search_distance",
        "hesitation_avg_speed",
        "travel_avg_speed",
        "search_avg_speed",
        "state_transitions",
        "hesitation_ratio",
        "total_hesitations",
        "average_duration",
        "max_duration",
        "hesitation_periods",
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_metrics_types_are_correct():
    """Verifica que los tipos de datos sean correctos."""
    trial, subject = _build_simple_trial_and_subject()
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2)
    
    assert isinstance(metrics["state_transitions"], int)
    assert isinstance(metrics["total_hesitations"], int)
    assert isinstance(metrics["hesitation_periods"], list)
    
    for k in ["hesitation_time", "travel_time", "search_time"]:
        assert isinstance(metrics[k], float), f"{k} should be float"
    for k in ["hesitation_distance", "travel_distance", "search_distance"]:
        assert isinstance(metrics[k], float), f"{k} should be float"
    for k in ["hesitation_avg_speed", "travel_avg_speed", "search_avg_speed"]:
        assert isinstance(metrics[k], float), f"{k} should be float"


def test_prefix_adds_prefix_to_all_keys():
    """Verifica que el prefijo se agregue a todas las métricas."""
    trial, subject = _build_simple_trial_and_subject()
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2, prefix="non_cut_")
    
    assert "non_cut_search_time" in metrics
    assert "non_cut_travel_time" in metrics
    assert "non_cut_hesitation_time" in metrics
    assert "non_cut_state_transitions" in metrics
    assert "non_cut_hesitation_ratio" in metrics
    
    assert "search_time" not in metrics
    assert "travel_time" not in metrics


# =============================================================================
# Tests de validación de errores
# =============================================================================

def test_speed_threshold_none_raises_error():
    """speed_threshold=None debe lanzar ValueError."""
    trial, subject = _build_simple_trial_and_subject()
    
    with pytest.raises(ValueError, match="Speed threshold must be provided"):
        _compute_metrics(trial, subject, speed_threshold=None, consecutive_points=2)


def test_speed_threshold_negative_raises_error():
    """speed_threshold <= 0 debe lanzar ValueError."""
    trial, subject = _build_simple_trial_and_subject()
    
    with pytest.raises(ValueError, match="Speed threshold must be positive"):
        _compute_metrics(trial, subject, speed_threshold=-1.0, consecutive_points=2)


def test_speed_threshold_zero_raises_error():
    """speed_threshold = 0 debe lanzar ValueError."""
    trial, subject = _build_simple_trial_and_subject()
    
    with pytest.raises(ValueError, match="Speed threshold must be positive"):
        _compute_metrics(trial, subject, speed_threshold=0.0, consecutive_points=2)


def test_consecutive_points_none_raises_error():
    """consecutive_points=None debe lanzar ValueError."""
    trial, subject = _build_simple_trial_and_subject()
    
    with pytest.raises(ValueError, match="Consecutive points must be provided"):
        _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=None)


def test_consecutive_points_zero_raises_error():
    """consecutive_points <= 0 debe lanzar ValueError."""
    trial, subject = _build_simple_trial_and_subject()
    
    with pytest.raises(ValueError, match="Number of consecutive points must be positive"):
        _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=0)


def test_consecutive_points_negative_raises_error():
    """consecutive_points < 0 debe lanzar ValueError."""
    trial, subject = _build_simple_trial_and_subject()
    
    with pytest.raises(ValueError, match="Number of consecutive points must be positive"):
        _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=-1)


# =============================================================================
# Tests de invariantes matemáticos
# =============================================================================

def test_time_metrics_sum_equals_total_duration():
    """La suma de tiempos por estado debe ser igual a la duración total del trail."""
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
        (5.0, 0.0, 2.0),
        (10.0, 0.0, 3.0),
        (10.5, 0.0, 4.0),
        (15.0, 0.0, 5.0),
    ])
    targets = [
        TMTTarget("1", Coordinate(0.0, 0.0)),
        TMTTarget("2", Coordinate(10.0, 0.0)),
    ]
    trial, subject = build_trial_and_subject(cursor_trail=cursor_trail, targets=targets, target_radius=1.0)
    
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2)
    
    total_duration = cursor_trail[-1].time - cursor_trail[0].time
    total_state_time = metrics["search_time"] + metrics["travel_time"] + metrics["hesitation_time"]
    
    assert total_state_time == pytest.approx(total_duration, abs=1e-6), \
        f"Sum of state times ({total_state_time}) != trial duration ({total_duration})"


def test_hesitation_ratio_is_consistent():
    """hesitation_ratio debe ser hesitation_time / (travel_time + hesitation_time)."""
    trial, subject = _build_simple_trial_and_subject()
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2)
    
    denom = metrics["travel_time"] + metrics["hesitation_time"]
    if denom > 0:
        expected_ratio = metrics["hesitation_time"] / denom
        assert metrics["hesitation_ratio"] == pytest.approx(expected_ratio, abs=1e-6)
    else:
        assert metrics["hesitation_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_hesitation_periods_consistency():
    """total_hesitations debe ser igual a len(hesitation_periods), y max/avg deben ser consistentes."""
    trial, subject = _build_simple_trial_and_subject()
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2)
    
    hp = metrics["hesitation_periods"]
    assert metrics["total_hesitations"] == len(hp)
    
    if len(hp) == 0:
        assert metrics["average_duration"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["max_duration"] == pytest.approx(0.0, abs=1e-6)
    else:
        assert metrics["max_duration"] == pytest.approx(max(hp), abs=1e-6)
        assert metrics["average_duration"] == pytest.approx(sum(hp) / len(hp), abs=1e-6)


def test_distance_metrics_sum_is_consistent():
    """La suma de distancias por estado debe ser igual a la distancia total recorrida."""
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 1.0),
        (20.0, 0.0, 2.0),
    ])
    targets = [TMTTarget("1", Coordinate(0.0, 0.0))]
    trial, subject = build_trial_and_subject(cursor_trail=cursor_trail, targets=targets, target_radius=1.0)
    
    metrics = _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=2)
    
    total_distance = metrics["search_distance"] + metrics["travel_distance"] + metrics["hesitation_distance"]
    
    expected_total_distance = 20.0
    assert total_distance == pytest.approx(expected_total_distance, abs=1e-6), \
        f"Sum of state distances ({total_distance}) != expected total distance ({expected_total_distance})"


def test_metrics_are_non_negative():
    """Todas las métricas de tiempo, distancia y velocidad deben ser >= 0."""
    trial, subject = _build_simple_trial_and_subject()
    metrics = _compute_metrics(trial, subject, speed_threshold=3.0, consecutive_points=2)
    
    for k in ["hesitation_time", "travel_time", "search_time"]:
        assert metrics[k] >= 0.0, f"{k} should be non-negative"
    for k in ["hesitation_distance", "travel_distance", "search_distance"]:
        assert metrics[k] >= 0.0, f"{k} should be non-negative"
    for k in ["hesitation_avg_speed", "travel_avg_speed", "search_avg_speed"]:
        assert metrics[k] >= 0.0, f"{k} should be non-negative"
    
    assert metrics["state_transitions"] >= 0
    assert metrics["total_hesitations"] >= 0
    assert 0.0 <= metrics["hesitation_ratio"] <= 1.0


# =============================================================================
# Tests de estados específicos
# =============================================================================

def test_all_points_on_target_only_search():
    """Cuando todos los puntos están sobre el target, solo debe haber tiempo en Search."""
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 1.0),
        (0.2, 0.0, 2.0),
        (0.3, 0.0, 3.0),
    ])
    targets = [TMTTarget("1", Coordinate(0.0, 0.0))]
    trial, subject = build_trial_and_subject(cursor_trail=cursor_trail, targets=targets, target_radius=1.0)
    
    metrics = _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=2)
    
    assert metrics["search_time"] > 0.0
    assert metrics["travel_time"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hesitation_time"] == pytest.approx(0.0, abs=1e-6)


def test_all_points_on_target_exact_values():
    """
    Trail donde todos los puntos están sobre el target.
    Estados: Search(0) -> Search(1) -> Search(2) -> Search(3)
    
    Valores esperados calculados manualmente:
    - search_time: 3.0 (de t=0 a t=3, acumulado para estado anterior)
    - search_distance: 0.3 (0.1 + 0.1 + 0.1)
    - search_avg_speed: ~0.1 (promedio de velocidades: 0.1, 0.1, 0.1)
    - travel_time, hesitation_time: 0.0
    - state_transitions: 0
    """
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 1.0),
        (0.2, 0.0, 2.0),
        (0.3, 0.0, 3.0),
    ])
    targets = [TMTTarget("1", Coordinate(0.0, 0.0))]
    trial, subject = build_trial_and_subject(cursor_trail=cursor_trail, targets=targets, target_radius=1.0)
    
    metrics = _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=2)
    
    assert metrics["search_time"] == pytest.approx(3.0, abs=1e-6)
    assert metrics["travel_time"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hesitation_time"] == pytest.approx(0.0, abs=1e-6)
    
    assert metrics["search_distance"] == pytest.approx(0.3, abs=1e-6)
    assert metrics["travel_distance"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hesitation_distance"] == pytest.approx(0.0, abs=1e-6)
    
    assert metrics["search_avg_speed"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["travel_avg_speed"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hesitation_avg_speed"] == pytest.approx(0.0, abs=1e-6)
    
    assert metrics["state_transitions"] == 0
    assert metrics["total_hesitations"] == 0
    assert metrics["hesitation_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_no_hesitation_returns_zero_hesitation_metrics():
    """Sin hesitación, las métricas de hesitación deben ser 0."""
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
        (10.0, 0.0, 2.0),
        (15.0, 0.0, 3.0),
    ])
    targets = [TMTTarget("1", Coordinate(0.0, 0.0))]
    trial, subject = build_trial_and_subject(cursor_trail=cursor_trail, targets=targets, target_radius=1.0)
    
    metrics = _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=2)
    
    assert metrics["total_hesitations"] == 0
    assert metrics["hesitation_periods"] == []
    assert metrics["average_duration"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["max_duration"] == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Tests de boundary cases
# =============================================================================

def test_cursor_trail_with_minimum_points():
    """Trail con 2 puntos (mínimo para calcular velocidad)."""
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 1.0),
    ])
    targets = [TMTTarget("1", Coordinate(0.0, 0.0))]
    trial, subject = build_trial_and_subject(cursor_trail=cursor_trail, targets=targets, target_radius=1.0)
    
    metrics = _compute_metrics(trial, subject, speed_threshold=1.0, consecutive_points=2)
    
    total_duration = cursor_trail[-1].time - cursor_trail[0].time
    total_state_time = metrics["search_time"] + metrics["travel_time"] + metrics["hesitation_time"]
    
    assert total_state_time == pytest.approx(total_duration, abs=1e-6)
    assert metrics["total_hesitations"] >= 0
    assert isinstance(metrics["hesitation_periods"], list)
