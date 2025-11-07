from neurotask.tmt.metrics.targets_touched import (
    touched_targets_for_every_cursor_point,
    correct_touched_targets_for_every_cursor_point,
    count_correctly_touched_targets
)
from neurotask.tmt.model.tmt_model import TMTTrial, TMTTarget, CursorInfo, Coordinate, TrialType


class TestTargetTouchFunctions:
    """
    Unified tests for all target touch functions:
    - touched_targets_for_every_cursor_point
    - correct_touched_targets_for_every_cursor_point
    - count_correctly_touched_targets

    Each test verifies the consistency between all three functions.
    """

    def test_no_targets_touched(self):
        """
        When the cursor never gets close to any target,
        all touched_target lists should be empty.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

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

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == len(cursor_trail)
        for touched_list, cursor_info in touched_result:
            assert len(touched_list) == 0
            assert isinstance(cursor_info, CursorInfo)

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == len(cursor_trail)
        for target, cursor_info in correct_result:
            assert target is None

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 0

    def test_all_targets_touched_correctly(self):
        """
        When all targets are touched in correct order.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
            TMTTarget("3", Coordinate(30.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),
            CursorInfo(Coordinate(10.0, 0.0), 1.0),
            CursorInfo(Coordinate(20.0, 0.0), 2.0),
            CursorInfo(Coordinate(30.0, 0.0), 3.0),
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_2",
            order_of_appearance=1,
            rt=3.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 4
        for i, (touched_list, cursor_info) in enumerate(touched_result):
            assert len(touched_list) == 1
            assert touched_list[0] == stimuli[i]

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 4
        for i, (target, cursor_info) in enumerate(correct_result):
            assert target == stimuli[i]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 4

    def test_partial_completion(self):
        """
        When only some targets are touched correctly.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
            TMTTarget("3", Coordinate(30.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),
            CursorInfo(Coordinate(10.0, 0.0), 1.0),
            CursorInfo(Coordinate(15.0, 0.0), 2.0),  # Stop here
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_3",
            order_of_appearance=1,
            rt=2.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 3
        assert len(touched_result[0][0]) == 1  # Target 0 touched
        assert len(touched_result[1][0]) == 1  # Target 1 touched
        assert len(touched_result[2][0]) == 0  # No target touched

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 3
        assert correct_result[0][0] == stimuli[0]
        assert correct_result[1][0] == stimuli[1]
        assert correct_result[2][0] is None

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 2

    def test_targets_touched_out_of_order(self):
        """
        When targets are touched out of order, then corrected.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
            TMTTarget("3", Coordinate(30.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # Touch 0 (correct)
            CursorInfo(Coordinate(20.0, 0.0), 1.0),  # Touch 2 (WRONG - skip 1)
            CursorInfo(Coordinate(30.0, 0.0), 2.0),  # Touch 3 (WRONG - still need 1)
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # Touch 1 (correct)
            CursorInfo(Coordinate(20.0, 0.0), 4.0),  # Touch 2 (correct)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_4",
            order_of_appearance=1,
            rt=4.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 5
        assert stimuli[0] in touched_result[0][0]
        assert stimuli[2] in touched_result[1][0]
        assert stimuli[3] in touched_result[2][0]
        assert stimuli[1] in touched_result[3][0]
        assert stimuli[2] in touched_result[4][0]

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 5
        assert correct_result[0][0] == stimuli[0]  # Correct
        assert correct_result[1][0] is None        # Wrong
        assert correct_result[2][0] is None        # Wrong
        assert correct_result[3][0] == stimuli[1]  # Correct
        assert correct_result[4][0] == stimuli[2]  # Correct

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 3

    def test_multiple_overlapping_targets(self):
        """
        When targets are close together and cursor is between them.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(2.0, 0.0)),   # Close to target 0
            TMTTarget("2", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(1.0, 0.0), 0.0),  # Touches both 0 and 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_5",
            order_of_appearance=1,
            rt=1.0
        )

        target_radius = 2.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 1
        touched_list, cursor_info = touched_result[0]
        assert len(touched_list) == 2
        assert stimuli[0] in touched_list
        assert stimuli[1] in touched_list

        # Test correct_touched_targets_for_every_cursor_point
        # Should only return target 0 (the first expected one)
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 1
        assert correct_result[0][0] == stimuli[0]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 1

    def test_staying_on_target_multiple_points(self):
        """
        When cursor stays on a target for multiple points.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # Touch 0
            CursorInfo(Coordinate(0.0, 0.0), 1.0),   # Still on 0
            CursorInfo(Coordinate(0.0, 0.0), 2.0),   # Still on 0
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # Touch 1
            CursorInfo(Coordinate(10.0, 0.0), 4.0),  # Still on 1
            CursorInfo(Coordinate(20.0, 0.0), 5.0),  # Touch 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_6",
            order_of_appearance=1,
            rt=5.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 6
        # All first 3 points touch target 0
        for i in range(3):
            assert stimuli[0] in touched_result[i][0]
        # Points 3-4 touch target 1
        assert stimuli[1] in touched_result[3][0]
        assert stimuli[1] in touched_result[4][0]
        # Point 5 touches target 2
        assert stimuli[2] in touched_result[5][0]

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 6
        assert correct_result[0][0] == stimuli[0]  # First touch of 0
        assert correct_result[1][0] is None        # Already counted 0
        assert correct_result[2][0] is None        # Already counted 0
        assert correct_result[3][0] == stimuli[1]  # First touch of 1
        assert correct_result[4][0] is None        # Already counted 1
        assert correct_result[5][0] == stimuli[2]  # First touch of 2

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 3

    def test_going_back_and_forth(self):
        """
        When cursor goes back and forth between targets.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # Touch 0
            CursorInfo(Coordinate(5.0, 0.0), 1.0),   # Between
            CursorInfo(Coordinate(0.0, 0.0), 2.0),   # Back to 0
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # Touch 1
            CursorInfo(Coordinate(5.0, 0.0), 4.0),   # Between
            CursorInfo(Coordinate(20.0, 0.0), 5.0),  # Touch 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_7",
            order_of_appearance=1,
            rt=5.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 6

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 6
        assert correct_result[0][0] == stimuli[0]
        assert correct_result[1][0] is None
        assert correct_result[2][0] is None  # Back to 0 but already counted
        assert correct_result[3][0] == stimuli[1]
        assert correct_result[4][0] is None
        assert correct_result[5][0] == stimuli[2]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 3

    def test_empty_cursor_trail(self):
        """
        When there is no cursor trail.
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
            id="test_8",
            order_of_appearance=1,
            rt=0.0
        )

        target_radius = 5.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 0

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 0

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 0

    def test_single_target_trial(self):
        """
        When trial has only one target.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),
            CursorInfo(Coordinate(0.0, 0.0), 1.0),
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_9",
            order_of_appearance=1,
            rt=1.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 2
        assert stimuli[0] in touched_result[0][0]
        assert stimuli[0] in touched_result[1][0]

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 2
        assert correct_result[0][0] == stimuli[0]
        assert correct_result[1][0] is None  # Already counted

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 1

    def test_with_custom_start(self):
        """
        When trial has a custom start point.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(-100.0, 0.0), 0.0),  # Before start
            CursorInfo(Coordinate(-50.0, 0.0), 1.0),   # Before start
            CursorInfo(Coordinate(0.0, 0.0), 2.0),     # Start - touch 0
            CursorInfo(Coordinate(10.0, 0.0), 3.0),    # Touch 1
            CursorInfo(Coordinate(20.0, 0.0), 4.0),    # Touch 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_10",
            order_of_appearance=1,
            rt=2.0,
            with_custom_start=True,
            start=CursorInfo(Coordinate(0.0, 0.0), 2.0)
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 3  # Only from time 2.0 onwards

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 3
        assert correct_result[0][0] == stimuli[0]
        assert correct_result[1][0] == stimuli[1]
        assert correct_result[2][0] == stimuli[2]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 3

    def test_boundary_case_exactly_at_radius(self):
        """
        Test behavior when cursor is exactly at target_radius distance.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
        ]

        target_radius = 5.0
        cursor_trail = [
            CursorInfo(Coordinate(5.0, 0.0), 0.0),  # Exactly at radius
            CursorInfo(Coordinate(4.99, 0.0), 1.0), # Just inside radius
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_11",
            order_of_appearance=1,
            rt=1.0
        )

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result[0][0]) == 0  # Not touched
        assert len(touched_result[1][0]) == 1  # Touched

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert correct_result[0][0] is None
        assert correct_result[1][0] == stimuli[0]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 1

    def test_alternating_correct_incorrect_touches(self):
        """
        When user alternates between correct and incorrect targets.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
            TMTTarget("3", Coordinate(30.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # Touch 0 (correct)
            CursorInfo(Coordinate(30.0, 0.0), 1.0),  # Touch 3 (wrong)
            CursorInfo(Coordinate(20.0, 0.0), 2.0),  # Touch 2 (wrong)
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # Touch 1 (correct)
            CursorInfo(Coordinate(0.0, 0.0), 4.0),   # Back to 0 (wrong)
            CursorInfo(Coordinate(20.0, 0.0), 5.0),  # Touch 2 (correct)
            CursorInfo(Coordinate(30.0, 0.0), 6.0),  # Touch 3 (correct)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_12",
            order_of_appearance=1,
            rt=6.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 7
        assert stimuli[0] in touched_result[0][0]
        assert stimuli[3] in touched_result[1][0]
        assert stimuli[2] in touched_result[2][0]
        assert stimuli[1] in touched_result[3][0]

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 7
        assert correct_result[0][0] == stimuli[0]  # Correct
        assert correct_result[1][0] is None        # Wrong
        assert correct_result[2][0] is None        # Wrong
        assert correct_result[3][0] == stimuli[1]  # Correct
        assert correct_result[4][0] is None        # Wrong
        assert correct_result[5][0] == stimuli[2]  # Correct
        assert correct_result[6][0] == stimuli[3]  # Correct

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 4

    def test_large_radius_overlapping_targets(self):
        """
        When target radius is large and targets overlap.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(5.0, 0.0)),
            TMTTarget("2", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),
            CursorInfo(Coordinate(5.0, 0.0), 1.0),
            CursorInfo(Coordinate(10.0, 0.0), 2.0),
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_13",
            order_of_appearance=1,
            rt=2.0
        )

        target_radius = 10.0  # Large radius

        # Test touched_targets_for_every_cursor_point
        # With large radius, each point may touch multiple targets
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 3

        # Test correct_touched_targets_for_every_cursor_point
        # Should still follow sequential order
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 3
        assert correct_result[0][0] == stimuli[0]
        assert correct_result[1][0] == stimuli[1]
        assert correct_result[2][0] == stimuli[2]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 3

    def test_cursor_moving_in_and_out_of_target(self):
        """
        When cursor moves in and out of a target's radius.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(-10.0, 0.0), 0.0),  # Far from 0
            CursorInfo(Coordinate(-2.0, 0.0), 1.0),   # Getting close
            CursorInfo(Coordinate(0.0, 0.0), 2.0),    # On target 0
            CursorInfo(Coordinate(2.0, 0.0), 3.0),    # Moving away
            CursorInfo(Coordinate(10.0, 0.0), 4.0),   # On target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_14",
            order_of_appearance=1,
            rt=4.0
        )

        target_radius = 1.5

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 5
        assert len(touched_result[0][0]) == 0
        assert len(touched_result[1][0]) == 0
        assert len(touched_result[2][0]) == 1
        assert len(touched_result[3][0]) == 0
        assert len(touched_result[4][0]) == 1

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 5
        assert correct_result[0][0] is None
        assert correct_result[1][0] is None
        assert correct_result[2][0] == stimuli[0]
        assert correct_result[3][0] is None
        assert correct_result[4][0] == stimuli[1]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 2

    def test_two_target_simple_trial(self):
        """
        Simple test with a two-target trial.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),
            CursorInfo(Coordinate(10.0, 0.0), 1.0),
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_15",
            order_of_appearance=1,
            rt=1.0
        )

        target_radius = 1.0

        # Test touched_targets_for_every_cursor_point
        touched_result = touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(touched_result) == 2
        assert stimuli[0] in touched_result[0][0]
        assert stimuli[1] in touched_result[1][0]

        # Test correct_touched_targets_for_every_cursor_point
        correct_result = correct_touched_targets_for_every_cursor_point(trial, target_radius)
        assert len(correct_result) == 2
        assert correct_result[0][0] == stimuli[0]
        assert correct_result[1][0] == stimuli[1]

        # Test count_correctly_touched_targets
        count = count_correctly_touched_targets(trial, target_radius)
        assert count == 2

