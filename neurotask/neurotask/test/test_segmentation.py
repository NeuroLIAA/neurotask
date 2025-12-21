import pytest

from neurotask.tmt.metrics.speed_metrics import InvalidSpeedError, NonMonotonicTimeError
from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
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

    def test_raises_invalid_speed_error_when_raise_on_error_true(self):
        """
        Con raise_on_error=True, lanza InvalidSpeedError si velocidad > 8.0 px/ms.
        """
        # Movimiento de 100px en 1ms = 100 px/ms (inválido, > 8.0)
        cursor_trail = _build_cursor_trail([
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 1.0),  # speed = 100 px/ms (invalid)
            (102.0, 0.0, 2.0),  # speed = 2 px/ms (valid)
        ])
        trial = _build_trial(cursor_trail)

        with pytest.raises(InvalidSpeedError, match="exceeds INVALID_SPEED_THRESHOLD"):
            classify_cursor_positions_with_hesitation(
                tmt_trial=trial,
                target_radius=10.0,
                speed_threshold=2.0,
                consecutive_points=2,
                raise_on_error=True
            )

    def test_raises_non_monotonic_error_when_raise_on_error_true(self):
        """
        Con raise_on_error=True, lanza NonMonotonicTimeError si tiempo retrocede.
        """
        # Time: 0 -> 2 -> 1 (retrocede)
        cursor_trail = _build_cursor_trail([
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 2.0),
            (4.0, 0.0, 1.0),  # time goes backwards
        ])
        trial = _build_trial(cursor_trail)

        with pytest.raises(NonMonotonicTimeError, match="current_cursor.time must be greater than previous_cursor.time"):
            classify_cursor_positions_with_hesitation(
                tmt_trial=trial,
                target_radius=10.0,
                speed_threshold=2.0,
                consecutive_points=2,
                raise_on_error=True
            )

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
            consecutive_points=2,
            raise_on_error=True
        )

        # Verifica estructura correcta
        assert len(result) == len(cursor_trail)
        
        # Cada elemento es (estado, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    @pytest.mark.xfail(reason="Bug conocido: con raise_on_error=False y datos inválidos, IndexError por len(speeds) < len(cursor_trail) - 1")
    def test_with_invalid_data_and_raise_on_error_false(self):
        """
        Con raise_on_error=False, no lanza excepción y devuelve clasificación.
        
        NOTA: Este test documenta un bug conocido. Cuando hay velocidades inválidas
        y raise_on_error=False, la lista speeds tiene menos elementos que cursor_trail - 1,
        causando IndexError en speed_increases_over_consecutive_points().
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
            consecutive_points=2,
            raise_on_error=False
        )

        # Verifica estructura correcta
        assert len(result) == len(cursor_trail)
        
        # Cada elemento es (estado, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

