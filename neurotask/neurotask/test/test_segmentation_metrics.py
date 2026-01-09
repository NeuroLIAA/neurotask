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
    )


def test_segmentation_metrics_smoke_and_consistency_with_explicit_targets():
    """
    Test "correcto" (robusto) para SegmentationMetricCalculator:

    - No intenta adivinar la clasificación exacta (Search/Travel/Hesitation), porque eso depende de reglas.
    - En cambio:
        1) Fija targets explícitos para que `calculate_over_targets` sea determinístico.
        2) Verifica que estén las 15 métricas.
        3) Verifica invariantes consistentes con las definiciones:
           - tiempos/distancias no negativos
           - suma de tiempos = duración total del trail
           - hesitation_ratio consistente con sus componentes
           - coherencia interna de hesitation_periods / total_hesitations / max / avg
           - tipos esperados (int/list)
    """

    # Trail simple, tiempos monotónicos (en segundos acá, pero el código solo usa diferencias)
    cursor_trail = build_cursor_trail([
        (0.0, 0.0, 0.0),   # cerca del target 0
        (0.5, 0.0, 1.0),   # todavía dentro de radio (target 0)
        (5.0, 0.0, 2.0),   # lejos -> debería salir del target 0
        (10.0, 0.0, 3.0),  # cerca del target 1
        (10.5, 0.0, 4.0),  # todavía dentro del target 1
        (15.0, 0.0, 5.0),  # sale del target 1
    ])

    targets = [
        TMTTarget("1", Coordinate(0.0, 0.0)),
        TMTTarget("2", Coordinate(10.0, 0.0)),
    ]

    trial, subject = build_trial_and_subject(
        cursor_trail=cursor_trail,
        targets=targets,
        target_radius=1.0,
    )

    metrics = _compute_metrics(
        trial=trial,
        subject=subject,
        speed_threshold=3.0,
        consecutive_points=2,
    )

    # --- 1) Keys esperadas ---
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

    # --- 2) Tipos básicos ---
    assert isinstance(metrics["state_transitions"], int)
    assert isinstance(metrics["total_hesitations"], int)
    assert isinstance(metrics["hesitation_periods"], list)

    # --- 3) No-negatividad (invariantes) ---
    for k in ["hesitation_time", "travel_time", "search_time"]:
        assert metrics[k] >= 0.0
    for k in ["hesitation_distance", "travel_distance", "search_distance"]:
        assert metrics[k] >= 0.0
    for k in ["hesitation_avg_speed", "travel_avg_speed", "search_avg_speed"]:
        assert metrics[k] >= 0.0

    # --- 4) Suma de tiempos = duración total del trail ---
    # La implementación acumula dt entre puntos por estado "previo", por lo que la suma
    # de Search/Travel/Hesitation debe ser exactamente (t_last - t_first).
    t0 = cursor_trail[0].time
    tN = cursor_trail[-1].time
    total_duration = tN - t0

    total_state_time = (
        metrics["search_time"] +
        metrics["travel_time"] +
        metrics["hesitation_time"]
    )
    assert total_state_time == pytest.approx(total_duration, abs=1e-6), (
        f"Sum of state times ({total_state_time}) != trial duration ({total_duration})"
    )

    # --- 5) hesitation_ratio coherente con su definición ---
    denom = metrics["travel_time"] + metrics["hesitation_time"]
    if denom > 0:
        assert metrics["hesitation_ratio"] == pytest.approx(
            metrics["hesitation_time"] / denom, abs=1e-6
        )
    else:
        assert metrics["hesitation_ratio"] == pytest.approx(0.0, abs=1e-6)

    # --- 6) Coherencia de hesitation_periods ---
    hp = metrics["hesitation_periods"]
    assert metrics["total_hesitations"] == len(hp)

    if len(hp) == 0:
        assert metrics["average_duration"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["max_duration"] == pytest.approx(0.0, abs=1e-6)
    else:
        assert metrics["max_duration"] == pytest.approx(max(hp), abs=1e-6)
        assert metrics["average_duration"] == pytest.approx(sum(hp) / len(hp), abs=1e-6)

    # --- 7) Prefijo (mini check opcional dentro del mismo test) ---
    metrics_pref = _compute_metrics(
        trial=trial,
        subject=subject,
        speed_threshold=3.0,
        consecutive_points=2,
        prefix="non_cut_",
    )
    assert "non_cut_search_time" in metrics_pref
    assert "search_time" not in metrics_pref
