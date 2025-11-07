from neurotask.tmt.metrics.targets_touched import (
    touched_targets_for_every_cursor_point,
    correct_touched_targets_for_every_cursor_point
)
from neurotask.tmt.model.tmt_model import TMTTrial, TMTTarget, CursorInfo, Coordinate, TrialType


class TestTouchedTargetsForEveryCursorPoint:
    """Tests for touched_targets_for_every_cursor_point function"""

    def test_no_targets_touched(self):
        """
        When the cursor never gets close to any target,
        all touched_target lists should be empty.
        """
        # Create targets at (0, 0), (10, 0), (20, 0)
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        # Create cursor trail far from all targets
        cursor_trail = [
            CursorInfo(Coordinate(100.0, 100.0), 0.0),
            CursorInfo(Coordinate(100.0, 101.0), 1.0),
            CursorInfo(Coordinate(100.0, 102.0), 2.0),
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_1",
            order_of_appearance=1,
            rt=2.0
        )

        target_radius = 5.0
        result = touched_targets_for_every_cursor_point(trial, target_radius)

        # Check that we have the correct number of results
        assert len(result) == len(cursor_trail)

        # Check that no targets were touched at any point
        for touched_list, cursor_info in result:
            assert len(touched_list) == 0
            assert isinstance(cursor_info, CursorInfo)

    def test_single_target_touched_at_each_point(self):
        """
        When the cursor moves through targets sequentially,
        each cursor point should touch exactly one target.
        """
        # Create targets at (0, 0), (10, 0), (20, 0)
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        # Create cursor trail that touches each target exactly
        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0
            CursorInfo(Coordinate(10.0, 0.0), 1.0),  # On target 1
            CursorInfo(Coordinate(20.0, 0.0), 2.0),  # On target 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_2",
            order_of_appearance=1,
            rt=2.0
        )

        target_radius = 1.0
        result = touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 3

        # Check each cursor point touches exactly one target
        touched_list_0, cursor_0 = result[0]
        assert len(touched_list_0) == 1
        assert touched_list_0[0] == stimuli[0]
        assert cursor_0.position.x == 0.0

        touched_list_1, cursor_1 = result[1]
        assert len(touched_list_1) == 1
        assert touched_list_1[0] == stimuli[1]
        assert cursor_1.position.x == 10.0

        touched_list_2, cursor_2 = result[2]
        assert len(touched_list_2) == 1
        assert touched_list_2[0] == stimuli[2]
        assert cursor_2.position.x == 20.0

    def test_multiple_overlapping_targets(self):
        """
        When targets are close together and cursor is between them,
        multiple targets should be touched simultaneously.
        """
        # Create targets very close to each other
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(2.0, 0.0)),   # Close to target 0
            TMTTarget("2", Coordinate(10.0, 0.0)),
        ]

        # Cursor at (1.0, 0.0) should touch both target 0 and 1 with radius 2.0
        cursor_trail = [
            CursorInfo(Coordinate(1.0, 0.0), 0.0),
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_3",
            order_of_appearance=1,
            rt=1.0
        )

        target_radius = 2.0
        result = touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 1
        touched_list, cursor_info = result[0]

        # Should touch both target 0 and target 1
        assert len(touched_list) == 2
        assert stimuli[0] in touched_list
        assert stimuli[1] in touched_list
        assert stimuli[2] not in touched_list

    def test_cursor_moving_in_and_out_of_target(self):
        """
        When cursor moves in and out of a target's radius,
        the target should appear only when within radius.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        # Cursor moves from far away, into target 0, and back out
        cursor_trail = [
            CursorInfo(Coordinate(-10.0, 0.0), 0.0),  # Far from target 0
            CursorInfo(Coordinate(-2.0, 0.0), 1.0),   # Getting close
            CursorInfo(Coordinate(0.0, 0.0), 2.0),    # On target 0
            CursorInfo(Coordinate(2.0, 0.0), 3.0),    # Moving away
            CursorInfo(Coordinate(10.0, 0.0), 4.0),   # Far away again
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_4",
            order_of_appearance=1,
            rt=4.0
        )

        target_radius = 1.5
        result = touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 5

        # Point 0: far from target
        assert len(result[0][0]) == 0

        # Point 1: getting close but not touching
        assert len(result[1][0]) == 0

        # Point 2: on target 0
        assert len(result[2][0]) == 1
        assert result[2][0][0] == stimuli[0]

        # Point 3: moved away from target 0
        assert len(result[3][0]) == 0

        # Point 4: on target 1
        assert len(result[4][0]) == 1
        assert result[4][0][0] == stimuli[1]

    def test_empty_cursor_trail(self):
        """
        When there is no cursor trail, the result should be empty.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = []

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_5",
            order_of_appearance=1,
            rt=0.0
        )

        target_radius = 5.0
        result = touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 0

    def test_with_custom_start(self):
        """
        When trial has a custom start point,
        only cursor points after that start should be considered.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        # Cursor trail includes points before and after start
        cursor_trail = [
            CursorInfo(Coordinate(-100.0, 0.0), 0.0),  # Before start
            CursorInfo(Coordinate(-50.0, 0.0), 1.0),   # Before start
            CursorInfo(Coordinate(0.0, 0.0), 2.0),     # Start point - on target 0
            CursorInfo(Coordinate(5.0, 0.0), 3.0),     # After start
            CursorInfo(Coordinate(10.0, 0.0), 4.0),    # After start - on target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_6",
            order_of_appearance=1,
            rt=2.0,
            with_custom_start=True,
            start=CursorInfo(Coordinate(0.0, 0.0), 2.0)
        )

        target_radius = 1.0
        result = touched_targets_for_every_cursor_point(trial, target_radius)

        # Should only include points from time 2.0 onwards
        assert len(result) == 3

        # First result point should be on target 0
        assert len(result[0][0]) == 1
        assert result[0][0][0] == stimuli[0]

        # Last result point should be on target 1
        assert len(result[2][0]) == 1
        assert result[2][0][0] == stimuli[1]

    def test_boundary_case_exactly_at_radius(self):
        """
        Test behavior when cursor is exactly at target_radius distance.
        Should NOT be touched (< not <=).
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
        ]

        # Cursor exactly at radius distance
        target_radius = 5.0
        cursor_trail = [
            CursorInfo(Coordinate(5.0, 0.0), 0.0),  # Exactly at radius
            CursorInfo(Coordinate(4.99, 0.0), 1.0), # Just inside radius
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_7",
            order_of_appearance=1,
            rt=1.0
        )

        result = touched_targets_for_every_cursor_point(trial, target_radius)

        # At exactly radius: should NOT be touched (based on < comparison)
        assert len(result[0][0]) == 0

        # Just inside radius: should be touched
        assert len(result[1][0]) == 1
        assert result[1][0][0] == stimuli[0]


class TestCorrectTouchedTargetsForEveryCursorPoint:
    """Tests for correct_touched_targets_for_every_cursor_point function"""

    def test_correct_sequential_touches(self):
        """
        When targets are touched in the correct order,
        each touch should be marked as correct.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # Touch target 0 (start)
            CursorInfo(Coordinate(5.0, 0.0), 1.0),   # Between targets
            CursorInfo(Coordinate(10.0, 0.0), 2.0),  # Touch target 1 (correct)
            CursorInfo(Coordinate(15.0, 0.0), 3.0),  # Between targets
            CursorInfo(Coordinate(20.0, 0.0), 4.0),  # Touch target 2 (correct)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_c1",
            order_of_appearance=1,
            rt=4.0
        )

        target_radius = 1.0
        result = correct_touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 5

        # Point 0: start target touched
        assert result[0][0] is stimuli[0]

        # Point 1: no target touched
        assert result[1][0] is None

        # Point 2: target 1 correctly touched
        assert result[2][0] == stimuli[1]

        # Point 3: no target touched
        assert result[3][0] is None

        # Point 4: target 2 correctly touched
        assert result[4][0] == stimuli[2]

    def test_wrong_target_touched(self):
        """
        When an incorrect target is touched, it should return None.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0 (start)
            CursorInfo(Coordinate(20.0, 0.0), 1.0),  # On target 2 (WRONG - expected 1)
            CursorInfo(Coordinate(10.0, 0.0), 2.0),  # On target 1 (now correct)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_c2",
            order_of_appearance=1,
            rt=2.0
        )

        target_radius = 1.0
        result = correct_touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 3

        # Point 0: start target touched
        assert result[0][0] is stimuli[0]

        # Point 1: wrong target touched (expected 1, got 2)
        assert result[1][0] is None

        # Point 2: correct target touched
        assert result[2][0] == stimuli[1]

    def test_correct_then_complete(self):
        """
        After all targets are correctly touched,
        remaining points should return None.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0
            CursorInfo(Coordinate(10.0, 0.0), 1.0),  # On target 1 (complete!)
            CursorInfo(Coordinate(10.0, 0.0), 2.0),  # Still on target 1
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # Still on target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_c3",
            order_of_appearance=1,
            rt=3.0
        )

        target_radius = 1.0
        result = correct_touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 4

        # Point 0: start target touched
        assert result[0][0] is stimuli[0]

        # Point 1: target 1 correctly touched
        assert result[1][0] == stimuli[1]

        # Points 2-3: all targets already touched, should be None
        assert result[2][0] is None
        assert result[3][0] is None

    def test_overlapping_targets_correct_one_selected(self):
        """
        When multiple targets overlap and cursor touches both,
        only the correct expected one should be returned.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(12.0, 0.0)),  # Close to target 1
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0
            CursorInfo(Coordinate(11.0, 0.0), 1.0),  # Touches both 1 and 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_c4",
            order_of_appearance=1,
            rt=1.0
        )

        target_radius = 2.0
        result = correct_touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 2

        # Point 0: start target touched
        assert result[0][0] is stimuli[0]

        # Point 1: should return target 1 (the expected one), not target 2
        assert result[1][0] == stimuli[1]

    def test_empty_trail(self):
        """
        Empty cursor trail should return empty result.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = []

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_c5",
            order_of_appearance=1,
            rt=0.0
        )

        target_radius = 1.0
        result = correct_touched_targets_for_every_cursor_point(trial, target_radius)

        assert len(result) == 0

