import pytest

from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTTarget,
    TMTTrial,
    TrialType,
)


# =============================================================================
# Helper functions
# =============================================================================

def _build_cursor_trail(positions_and_times: list[tuple[float, float, float]]) -> list[CursorInfo]:
    """
    Build a cursor trail from a list of (x, y, time) tuples.
    """
    return [
        CursorInfo(Coordinate(x, y), t)
        for x, y, t in positions_and_times
    ]


def _build_trial(cursor_trail: list[CursorInfo], targets: list[TMTTarget] = None) -> TMTTrial:
    """
    Build a minimal trial with the given cursor trail.
    """
    if targets is None:
        targets = [
            TMTTarget("1", Coordinate(0.0, 0.0)),
            TMTTarget("2", Coordinate(50.0, 0.0)),
        ]

    return TMTTrial(
        stimuli=targets,
        cursor_trail=cursor_trail,
        trial_type=TrialType.PART_A,
        id="test_trial",
        order_of_appearance=1,
        rt=cursor_trail[-1].time if cursor_trail else 0.0,
    )


# =============================================================================
# Tests for classify_cursor_positions_with_hesitation
# =============================================================================

class TestClassifyCursorPositionsWithHesitation:
    """Tests for classify_cursor_positions_with_hesitation function."""

    def test_returns_correct_states_with_valid_data(self):
        """
        Verifica que la clasificación de estados sea correcta con datos válidos.
        """
        # Movimiento válido: velocidades de 2 px/ms (< 8.0 threshold)
        cursor_trail = _build_cursor_trail([
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),   # speed = 2
            (4.0, 0.0, 2.0),   # speed = 2
            (6.0, 0.0, 3.0),   # speed = 2
            (8.0, 0.0, 4.0),   # speed = 2
        ])
        trial = _build_trial(cursor_trail)

        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # Verifica estructura correcta
        assert len(result) == len(cursor_trail)
        
        # Cada elemento es (estado, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    def test_handles_invalid_speed_gracefully(self):
        """
        Con velocidades inválidas, no lanza excepción y devuelve clasificación.
        Las velocidades inválidas son marcadas como is_valid=False en SpeedResult.
        """
        # Trial con velocidad inválida (100 px/ms > 8.0)
        cursor_trail = _build_cursor_trail([
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 1.0),  # speed = 100 (invalid)
            (102.0, 0.0, 2.0),  # speed = 2 (valid)
            (104.0, 0.0, 3.0),  # speed = 2 (valid)
        ])
        trial = _build_trial(cursor_trail)

        # No debe lanzar excepción
        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # Verifica estructura correcta
        assert len(result) == len(cursor_trail)
        
        # Cada elemento es (estado, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    def test_handles_non_monotonic_time_gracefully(self):
        """
        Con tiempos no monótonos, no lanza excepción y devuelve clasificación.
        Los puntos con tiempo no monótono son marcados como is_valid=False en SpeedResult.
        """
        # Time: 0 -> 2 -> 1 (retrocede)
        cursor_trail = _build_cursor_trail([
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 2.0),
            (4.0, 0.0, 1.0),  # time goes backwards
            (6.0, 0.0, 3.0),  # valid again
        ])
        trial = _build_trial(cursor_trail)

        # No debe lanzar excepción
        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # Verifica estructura correcta
        assert len(result) == len(cursor_trail)
        
        # Cada elemento es (estado, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    def test_first_point_is_search_on_target(self):
        """
        Verifica que el primer punto sea Search cuando está sobre el target.
        """
        # Cursor empieza sobre el target (0,0)
        cursor_trail = _build_cursor_trail([
            (0.0, 0.0, 0.0),   # sobre target 1
            (2.0, 0.0, 1.0),
            (4.0, 0.0, 2.0),
        ])
        trial = _build_trial(cursor_trail)

        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # El primer punto debe ser Search
        assert result[0][0] == 'Search'
