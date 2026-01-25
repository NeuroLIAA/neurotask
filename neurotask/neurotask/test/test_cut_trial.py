import pytest
from neurotask.tmt.cut_criteria.cut_implementation import (
    cut_trial_at_minimum_targets,
    cut_trial_at_minimum_correct_targets,
    cut_at_time
)
from neurotask.tmt.model.tmt_model import TMTTrial, TMTTarget, TMTSubject, Coordinate, TrialType, CursorInfo


class TestCutTrialAtMinimumTargets:
    """
    Tests for cut_trial_at_minimum_targets function using TDD approach.

    This function cuts the trial at the point where the minimum number of correct target touches is reached.
    The cut happens when the cursor FIRST TOUCHES the Nth target (not when it leaves it).
    """

    def test_cut_at_first_target(self):
        """
        When cutting at 1 target, trial should be cut when cursor first touches target 0.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(-5.0, 0.0), 0.0),   # Not on target
            CursorInfo(Coordinate(0.0, 0.0), 1.0),    # First touch on target 0 - CUT HERE
            CursorInfo(Coordinate(0.0, 0.0), 2.0),    # Still on target 0
            CursorInfo(Coordinate(5.0, 0.0), 3.0),    # Between targets
            CursorInfo(Coordinate(10.0, 0.0), 4.0),   # On target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_1",
            order_of_appearance=1,
            rt=4.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        cut_trial = cut_trial_at_minimum_targets(1, subject, "subject_1", trial, 1.0)

        # Expected trial cut at time 1.0 (when first touching target 0)
        expected_trial = TMTTrial(
            stimuli=[TMTTarget("0", Coordinate(0.0, 0.0))],
            cursor_trail=[
                CursorInfo(Coordinate(-5.0, 0.0), 0.0),
                CursorInfo(Coordinate(0.0, 0.0), 1.0),
            ],
            trial_type=TrialType.PART_A,
            id="test_1",
            order_of_appearance=1,
            rt=1.0
        )

        assert cut_trial == expected_trial

    def test_cut_at_second_target(self):
        """
        When cutting at 2 targets, trial should be cut when cursor first touches target 1.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),    # On target 0
            CursorInfo(Coordinate(0.0, 0.0), 1.0),    # Still on target 0
            CursorInfo(Coordinate(5.0, 0.0), 2.0),    # Between targets
            CursorInfo(Coordinate(10.0, 0.0), 3.0),   # First touch on target 1 - CUT HERE
            CursorInfo(Coordinate(10.0, 0.0), 4.0),   # Still on target 1
            CursorInfo(Coordinate(20.0, 0.0), 5.0),   # On target 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_2",
            order_of_appearance=1,
            rt=5.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        cut_trial = cut_trial_at_minimum_targets(2, subject, "subject_1", trial, 1.0)

        # Expected trial cut at time 3.0 (when first touching target 1)
        expected_trial = TMTTrial(
            stimuli=[
                TMTTarget("0", Coordinate(0.0, 0.0)),
                TMTTarget("1", Coordinate(10.0, 0.0)),
            ],
            cursor_trail=[
                CursorInfo(Coordinate(0.0, 0.0), 0.0),
                CursorInfo(Coordinate(0.0, 0.0), 1.0),
                CursorInfo(Coordinate(5.0, 0.0), 2.0),
                CursorInfo(Coordinate(10.0, 0.0), 3.0),
            ],
            trial_type=TrialType.PART_A,
            id="test_2",
            order_of_appearance=1,
            rt=3.0
        )

        assert cut_trial == expected_trial

    def test_cut_at_all_targets(self):
        """
        When cutting at N targets where N is the total number of targets,
        should return the full trial (or cut at the last target touch).
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),    # On target 0
            CursorInfo(Coordinate(5.0, 0.0), 1.0),    # Between targets
            CursorInfo(Coordinate(10.0, 0.0), 2.0),   # First touch on target 1 - CUT HERE
            CursorInfo(Coordinate(10.0, 0.0), 3.0),   # Still on target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_3",
            order_of_appearance=1,
            rt=3.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        cut_trial = cut_trial_at_minimum_targets(2, subject, "subject_1", trial, 1.0)

        # Expected trial cut at time 2.0 (when first touching target 1)
        expected_trial = TMTTrial(
            stimuli=[
                TMTTarget("0", Coordinate(0.0, 0.0)),
                TMTTarget("1", Coordinate(10.0, 0.0)),
            ],
            cursor_trail=[
                CursorInfo(Coordinate(0.0, 0.0), 0.0),
                CursorInfo(Coordinate(5.0, 0.0), 1.0),
                CursorInfo(Coordinate(10.0, 0.0), 2.0),
            ],
            trial_type=TrialType.PART_A,
            id="test_3",
            order_of_appearance=1,
            rt=2.0
        )

        assert cut_trial == expected_trial

    def test_insufficient_correct_targets_raises_error(self):
        """
        When trial doesn't have enough correct targets, should raise ValueError.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),    # On target 0
            CursorInfo(Coordinate(5.0, 0.0), 1.0),    # Between targets
            # Never touches target 1 or 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_4",
            order_of_appearance=1,
            rt=1.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Should raise error because trial only touches 1 target but requires 3
        with pytest.raises(ValueError, match="has 1 correct target touches"):
            cut_trial_at_minimum_targets(3, subject, "subject_1", trial, 1.0)

    def test_target_touched_out_of_order_not_counted(self):
        """
        When targets are touched out of order, only correct touches should count.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),    # On target 0 (correct)
            CursorInfo(Coordinate(20.0, 0.0), 1.0),   # On target 2 (WRONG - skipped 1)
            CursorInfo(Coordinate(10.0, 0.0), 2.0),   # On target 1 (correct now)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_5",
            order_of_appearance=1,
            rt=2.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Should raise error because only 2 correct touches (target 2 was out of order)
        with pytest.raises(ValueError, match="has 2 correct target touches"):
            cut_trial_at_minimum_targets(3, subject, "subject_1", trial, 1.0)

    def test_with_custom_start(self):
        """
        When trial has a custom start point, cutting should respect it.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(-100.0, 0.0), 0.0),  # Before start
            CursorInfo(Coordinate(0.0, 0.0), 1.0),     # Start - on target 0
            CursorInfo(Coordinate(0.0, 0.0), 2.0),     # Still on target 0
            CursorInfo(Coordinate(10.0, 0.0), 3.0),    # On target 1 - CUT HERE
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_6",
            order_of_appearance=1,
            rt=2.0,
            with_custom_start=True,
            start=CursorInfo(Coordinate(0.0, 0.0), 1.0)
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        cut_trial = cut_trial_at_minimum_targets(2, subject, "subject_1", trial, 1.0)

        # Expected trial cut at time 3.0 (when first touching target 1)
        expected_trial = TMTTrial(
            stimuli=[
                TMTTarget("0", Coordinate(0.0, 0.0)),
                TMTTarget("1", Coordinate(10.0, 0.0)),
            ],
            cursor_trail=[
                CursorInfo(Coordinate(-100.0, 0.0), 0.0),
                CursorInfo(Coordinate(0.0, 0.0), 1.0),
                CursorInfo(Coordinate(0.0, 0.0), 2.0),
                CursorInfo(Coordinate(10.0, 0.0), 3.0),
            ],
            trial_type=TrialType.PART_A,
            id="test_6",
            order_of_appearance=1,
            rt=3.0,
            with_custom_start=True,
            start=CursorInfo(Coordinate(0.0, 0.0), 1.0)
        )

        assert cut_trial == expected_trial

    def test_multiple_entries_on_same_target(self):
        """
        When cursor enters and exits the same target multiple times,
        only the first entry should count for cutting.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),    # On target 0 - FIRST TOUCH
            CursorInfo(Coordinate(5.0, 0.0), 1.0),    # Leave target 0
            CursorInfo(Coordinate(0.0, 0.0), 2.0),    # Back on target 0 (doesn't count)
            CursorInfo(Coordinate(5.0, 0.0), 3.0),    # Leave again
            CursorInfo(Coordinate(10.0, 0.0), 4.0),   # On target 1 - CUT HERE
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_7",
            order_of_appearance=1,
            rt=4.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        cut_trial = cut_trial_at_minimum_targets(2, subject, "subject_1", trial, 1.0)

        # Expected trial cut at time 4.0 (first touch on target 1)
        expected_trial = TMTTrial(
            stimuli=[
                TMTTarget("0", Coordinate(0.0, 0.0)),
                TMTTarget("1", Coordinate(10.0, 0.0)),
            ],
            cursor_trail=[
                CursorInfo(Coordinate(0.0, 0.0), 0.0),
                CursorInfo(Coordinate(5.0, 0.0), 1.0),
                CursorInfo(Coordinate(0.0, 0.0), 2.0),
                CursorInfo(Coordinate(5.0, 0.0), 3.0),
                CursorInfo(Coordinate(10.0, 0.0), 4.0),
            ],
            trial_type=TrialType.PART_A,
            id="test_7",
            order_of_appearance=1,
            rt=4.0
        )

        assert cut_trial == expected_trial



    def test_large_radius_overlapping_targets(self):
        """
        With large radius where targets overlap, should cut at first touch of correct target.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(5.0, 0.0)),
            TMTTarget("2", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),    # On targets 0 and 1 (expected: 0)
            CursorInfo(Coordinate(5.0, 0.0), 1.0),    # On all targets (expected: 1) - CUT HERE
            CursorInfo(Coordinate(10.0, 0.0), 2.0),   # On targets 1 and 2 (expected: 2)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_8",
            order_of_appearance=1,
            rt=2.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=10.0,  # Large radius
            canvas_size=None
        )

        cut_trial = cut_trial_at_minimum_targets(2, subject, "subject_1", trial, 1.0)

        # Expected trial cut at time 1.0 (first touch on correct target 1)
        expected_trial = TMTTrial(
            stimuli=[
                TMTTarget("0", Coordinate(0.0, 0.0)),
                TMTTarget("1", Coordinate(5.0, 0.0)),
            ],
            cursor_trail=[
                CursorInfo(Coordinate(0.0, 0.0), 0.0),
                CursorInfo(Coordinate(5.0, 0.0), 1.0),
            ],
            trial_type=TrialType.PART_A,
            id="test_8",
            order_of_appearance=1,
            rt=1.0
        )

        assert cut_trial == expected_trial

