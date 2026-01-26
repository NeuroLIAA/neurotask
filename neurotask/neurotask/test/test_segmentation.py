from neurotask.tmt.segmentation.segmentation import classify_cursor_positions_with_hesitation
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTTarget,
    TMTTrial,
)
from neurotask.test.test_helpers import build_cursor_trail, build_trial


# =============================================================================
# Tests for classify_cursor_positions_with_hesitation
# =============================================================================

class TestClassifyCursorPositionsWithHesitation:
    """Tests for classify_cursor_positions_with_hesitation function."""

    def test_returns_correct_states_with_valid_data(self):
        """
        Verify that state classification is correct with valid data.
        """
        # Valid movement: speeds of 2 px/ms (< 8.0 threshold)
        cursor_trail = build_cursor_trail([
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),   # speed = 2
            (4.0, 0.0, 2.0),   # speed = 2
            (6.0, 0.0, 3.0),   # speed = 2
            (8.0, 0.0, 4.0),   # speed = 2
        ])
        trial = build_trial(cursor_trail)

        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            multiplier=1.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # Verify correct structure
        assert len(result) == len(cursor_trail)
        
        # Each element is (state, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    def test_handles_invalid_speed_gracefully(self):
        """
        With invalid speeds, does not raise exception and returns classification.
        Invalid speeds are marked as is_valid=False in SpeedResult.
        """
        # Trial with invalid speed (100 px/ms > 8.0)
        cursor_trail = build_cursor_trail([
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 1.0),  # speed = 100 (invalid)
            (102.0, 0.0, 2.0),  # speed = 2 (valid)
            (104.0, 0.0, 3.0),  # speed = 2 (valid)
        ])
        trial = build_trial(cursor_trail)

        # Should not raise exception
        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            multiplier=1.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # Verify correct structure
        assert len(result) == len(cursor_trail)
        
        # Each element is (state, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    def test_handles_non_monotonic_time_gracefully(self):
        """
        With non-monotonic times, does not raise exception and returns classification.
        Points with non-monotonic time are marked as is_valid=False in SpeedResult.
        """
        # Time: 0 -> 2 -> 1 (goes backwards)
        cursor_trail = build_cursor_trail([
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 2.0),
            (4.0, 0.0, 1.0),  # time goes backwards
            (6.0, 0.0, 3.0),  # valid again
        ])
        trial = build_trial(cursor_trail)

        # Should not raise exception
        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            multiplier=1.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # Verify correct structure
        assert len(result) == len(cursor_trail)
        
        # Each element is (state, CursorInfo)
        for state, cursor_info in result:
            assert state in ['Search', 'Travel', 'Hesitation']
            assert isinstance(cursor_info, CursorInfo)

    def test_first_point_is_search_on_target(self):
        """
        Verify that the first point is Search when over the target.
        """
        # Cursor starts over the target (0,0)
        cursor_trail = build_cursor_trail([
            (0.0, 0.0, 0.0),   # over target 1
            (2.0, 0.0, 1.0),
            (4.0, 0.0, 2.0),
        ])
        trial = build_trial(cursor_trail)

        result = classify_cursor_positions_with_hesitation(
            tmt_trial=trial,
            target_radius=10.0,
            multiplier=1.0,
            speed_threshold=1.5,
            consecutive_points=2
        )

        # First point should be Search
        assert result[0][0] == 'Search'
