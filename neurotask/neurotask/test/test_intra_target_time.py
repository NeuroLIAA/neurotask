from neurotask.tmt.metrics.intra_and_inter_target_time import calculate_intra_target_time
from neurotask.tmt.model.tmt_model import TMTTrial, TMTTarget, TMTSubject, Coordinate, TrialType


class TestCalculateIntraTargetTime:
    """
    Tests for calculate_intra_target_time function using TDD approach.

    Intra-target time is defined as the sum of the times spent within each target area.
    """

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

        intra_target_time = calculate_intra_target_time(trial, subject)
        assert intra_target_time == 0.0

