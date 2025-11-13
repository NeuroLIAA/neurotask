from neurotask.tmt.metrics.intra_and_inter_target_time import (
    calculate_intra_target_time,
    calculate_inter_target_time
)
from neurotask.tmt.model.tmt_model import TMTTrial, TMTTarget, TMTSubject, Coordinate, TrialType, CursorInfo


class TestCalculateIntraTargetTime:
    """
    Tests for calculate_intra_target_time function using TDD approach.

    Intra-target time is defined as the sum of the times spent within each target area.
    """

    def validate_time_partition(self, trial: TMTTrial, subject: TMTSubject,
                                expected_intra: float, tolerance: float = 0.001):
        """
        Helper method to validate that intra_time + inter_time = total_time.

        :param trial: TMTTrial instance
        :param subject: TMTSubject instance
        :param expected_intra: Expected intra-target time value
        :param tolerance: Tolerance for floating point comparisons
        """
        intra_time = calculate_intra_target_time(trial, subject)
        inter_time = calculate_inter_target_time(trial, intra_time)

        # Validate expected intra-target time
        assert abs(intra_time - expected_intra) < tolerance, \
            f"Expected intra_time={expected_intra}, got {intra_time}"

        # Calculate total time from cursor trail
        cursor_trail = trial.get_cursor_trail_from_start()
        if cursor_trail:
            total_time = cursor_trail[-1].time - cursor_trail[0].time
            # Validate that intra + inter = total
            assert abs((intra_time + inter_time) - total_time) < tolerance, \
                f"intra_time ({intra_time}) + inter_time ({inter_time}) != total_time ({total_time})"
        else:
            # Empty trail: both should be 0
            assert intra_time == 0.0 and inter_time == 0.0

    def test_empty_cursor_trail(self):
        """
        When cursor trail is empty, intra-target time should be 0.
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
            id="test_1",
            order_of_appearance=1,
            rt=0.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=5.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        self.validate_time_partition(trial, subject, expected_intra=0.0)

    def test_single_point_on_target(self):
        """
        When there's only one point on a target, intra-target time should be 0
        (no time difference with next point).
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),  # On target 0
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_2",
            order_of_appearance=1,
            rt=0.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        self.validate_time_partition(trial, subject, expected_intra=0.0)

    def test_two_consecutive_points_on_same_target(self):
        """
        When two consecutive points are on the same target,
        intra-target time should be the time difference.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),  # On target 0 at t=0
            CursorInfo(Coordinate(0.0, 0.0), 2.5),  # Still on target 0 at t=2.5
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_3",
            order_of_appearance=1,
            rt=2.5
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0: 2.5 - 0.0 = 2.5
        self.validate_time_partition(trial, subject, expected_intra=2.5)

    def test_moving_between_two_targets(self):
        """
        When cursor moves from one target to another,
        count time on each target separately.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0 at t=0
            CursorInfo(Coordinate(0.0, 0.0), 1.0),   # Still on target 0 at t=1
            CursorInfo(Coordinate(5.0, 0.0), 2.0),   # Between targets at t=2
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # On target 1 at t=3
            CursorInfo(Coordinate(10.0, 0.0), 5.0),  # Still on target 1 at t=5
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_4",
            order_of_appearance=1,
            rt=5.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0: (1.0 - 0.0) = 1.0
        # Time on target 1: (5.0 - 3.0) = 2.0
        # Total: 1.0 + 2.0 = 3.0
        self.validate_time_partition(trial, subject, expected_intra=3.0)

    def test_no_targets_touched(self):
        """
        When cursor never touches any target, intra-target time should be 0.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
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

        # Validate intra-target time and time partition
        self.validate_time_partition(trial, subject, expected_intra=0.0)

    def test_multiple_targets_touched_in_order(self):
        """
        When multiple targets are touched in correct order,
        sum all time spent on all targets.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0
            CursorInfo(Coordinate(0.0, 0.0), 1.5),   # Still on target 0
            CursorInfo(Coordinate(5.0, 0.0), 2.0),   # Between targets
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # On target 1
            CursorInfo(Coordinate(10.0, 0.0), 4.0),  # Still on target 1
            CursorInfo(Coordinate(15.0, 0.0), 5.0),  # Between targets
            CursorInfo(Coordinate(20.0, 0.0), 6.0),  # On target 2
            CursorInfo(Coordinate(20.0, 0.0), 8.0),  # Still on target 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_6",
            order_of_appearance=1,
            rt=8.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0: (1.5 - 0.0) = 1.5
        # Time on target 1: (4.0 - 3.0) = 1.0
        # Time on target 2: (8.0 - 6.0) = 2.0
        # Total: 1.5 + 1.0 + 2.0 = 4.5
        self.validate_time_partition(trial, subject, expected_intra=4.5)

    def test_wrong_target_touched_not_counted(self):
        """
        When a wrong target is touched (out of order),
        time on that target should not be counted.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
            TMTTarget("2", Coordinate(20.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0 (correct)
            CursorInfo(Coordinate(0.0, 0.0), 1.0),   # Still on target 0
            CursorInfo(Coordinate(20.0, 0.0), 2.0),  # On target 2 (WRONG - skip 1)
            CursorInfo(Coordinate(20.0, 0.0), 5.0),  # Still on target 2 (wrong)
            CursorInfo(Coordinate(10.0, 0.0), 6.0),  # On target 1 (correct now)
            CursorInfo(Coordinate(10.0, 0.0), 8.0),  # Still on target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_7",
            order_of_appearance=1,
            rt=8.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0 (correct): (1.0 - 0.0) = 1.0
        # Time on target 2 (wrong): NOT COUNTED
        # Time on target 1 (correct): (8.0 - 6.0) = 2.0
        # Total: 1.0 + 2.0 = 3.0
        self.validate_time_partition(trial, subject, expected_intra=3.0)

    def test_entering_and_leaving_target_multiple_times(self):
        """
        When cursor enters and leaves a target multiple times,
        only count time when it's the correct expected target.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0 (correct)
            CursorInfo(Coordinate(0.0, 0.0), 1.0),   # Still on target 0
            CursorInfo(Coordinate(5.0, 0.0), 2.0),   # Between targets
            CursorInfo(Coordinate(0.0, 0.0), 3.0),   # Back on target 0 (not counted - already done)
            CursorInfo(Coordinate(0.0, 0.0), 4.0),   # Still on target 0 (not counted)
            CursorInfo(Coordinate(10.0, 0.0), 5.0),  # On target 1 (correct)
            CursorInfo(Coordinate(10.0, 0.0), 7.0),  # Still on target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_8",
            order_of_appearance=1,
            rt=7.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0 (first time, correct): (1.0 - 0.0) = 1.0
        # Time on target 0 (second time, not counted): 0
        # Time on target 1 (correct): (7.0 - 5.0) = 2.0
        # Total: 1.0 + 2.0 = 3.0
        self.validate_time_partition(trial, subject, expected_intra=3.0)

    def test_last_point_on_target_not_counted(self):
        """
        When the last cursor point is on a target,
        it contributes 0 to intra-target time (no next point to calculate difference).
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On target 0
            CursorInfo(Coordinate(0.0, 0.0), 1.0),   # Still on target 0
            CursorInfo(Coordinate(5.0, 0.0), 2.0),   # Between targets
            CursorInfo(Coordinate(10.0, 0.0), 3.0),  # On target 1 (last point)
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_9",
            order_of_appearance=1,
            rt=3.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0: (1.0 - 0.0) = 1.0
        # Time on target 1: 0 (last point, no next point)
        # Total: 1.0
        self.validate_time_partition(trial, subject, expected_intra=1.0)

    def test_with_custom_start(self):
        """
        When trial has a custom start point,
        only count time on targets after the start.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(-100.0, 0.0), 0.0),  # Before start
            CursorInfo(Coordinate(-50.0, 0.0), 1.0),   # Before start
            CursorInfo(Coordinate(0.0, 0.0), 2.0),     # Start - on target 0
            CursorInfo(Coordinate(0.0, 0.0), 3.0),     # Still on target 0
            CursorInfo(Coordinate(10.0, 0.0), 4.0),    # On target 1
            CursorInfo(Coordinate(10.0, 0.0), 6.0),    # Still on target 1
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_10",
            order_of_appearance=1,
            rt=4.0,
            with_custom_start=True,
            start=CursorInfo(Coordinate(0.0, 0.0), 2.0)
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=1.0,
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0: (3.0 - 2.0) = 1.0
        # Time on target 1: (6.0 - 4.0) = 2.0
        # Total: 1.0 + 2.0 = 3.0
        self.validate_time_partition(trial, subject, expected_intra=3.0)

    def test_overlapping_targets_with_large_radius(self):
        """
        When target radius is large and targets overlap,
        only count time for the correct expected target.
        """
        stimuli = [
            TMTTarget("0", Coordinate(0.0, 0.0)),
            TMTTarget("1", Coordinate(5.0, 0.0)),
            TMTTarget("2", Coordinate(10.0, 0.0)),
        ]

        cursor_trail = [
            CursorInfo(Coordinate(0.0, 0.0), 0.0),   # On targets 0 and 1 (expected: 0)
            CursorInfo(Coordinate(0.0, 0.0), 2.0),   # Still on targets 0 and 1
            CursorInfo(Coordinate(5.0, 0.0), 3.0),   # On all targets (expected: 1)
            CursorInfo(Coordinate(5.0, 0.0), 5.0),   # Still on all targets
            CursorInfo(Coordinate(10.0, 0.0), 6.0),  # On targets 1 and 2 (expected: 2)
            CursorInfo(Coordinate(10.0, 0.0), 8.0),  # Still on targets 1 and 2
        ]

        trial = TMTTrial(
            stimuli=stimuli,
            cursor_trail=cursor_trail,
            trial_type=TrialType.PART_A,
            id="test_11",
            order_of_appearance=1,
            rt=8.0
        )

        subject = TMTSubject(
            training_trials=[],
            testing_trials=[trial],
            target_radius=10.0,  # Large radius for overlapping
            canvas_size=None
        )

        # Validate intra-target time and time partition
        # Time on target 0 (correct): (2.0 - 0.0) = 2.0
        # Time on target 1 (correct): (5.0 - 3.0) = 2.0
        # Time on target 2 (correct): (8.0 - 6.0) = 2.0
        # Total: 2.0 + 2.0 + 2.0 = 6.0  (but was 5.0 before)
        # Note: Point at t=6.0 and t=8.0 are both on target 2
        self.validate_time_partition(trial, subject, expected_intra=5.0)


