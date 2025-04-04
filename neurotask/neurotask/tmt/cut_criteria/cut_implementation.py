import logging

from neurotask.tmt.cut_criteria.cut_criteria import CutCriteria
from neurotask.tmt.metrics import number_of_correct_and_incorrect_segments, get_correct_and_incorrect_segments
from neurotask.tmt.model.tmt_model import TMTTrial


def cut_trial(trial, correct_targets_minimum, subject, subject_id, cut_criteria):
    if cut_criteria == CutCriteria.MINIMUM_TARGETS:
        if correct_targets_minimum is None:
            raise ValueError("Minimum targets criteria requires a minimum number of correct targets.")
        return cut_trial_at_minimum_targets(correct_targets_minimum, subject, subject_id, trial)

    raise ValueError(f"Invalid cut criteria: {cut_criteria}")


def cut_trial_at_minimum_targets(correct_targets_minimum, subject, subject_id, trial):
    correct_targets_touches, _ = number_of_correct_and_incorrect_segments(
        trial,
        subject.target_radius
    )

    # If the number of correct target touches is at least the required minimum,
    # cut the trial at that number for further analysis.
    # Otherwise, mark the trial as invalid.
    is_valid_number_of_correct_targets = correct_targets_touches >= correct_targets_minimum
    if not is_valid_number_of_correct_targets:
        logging.warning(
            f"Trial {trial.id} of subject {subject_id} has less than {correct_targets_minimum} correct target touches." +
            f"Subject has {correct_targets_touches} correct target touches."
        )
        raise ValueError(f"Trial {trial.id} has less than {correct_targets_minimum} correct targets.")

    cutoff_trial = cut_trial_at_minimum_correct_targets(
        trial,
        correct_targets_minimum,
        subject.target_radius
    )

    return cutoff_trial


def cut_trial_at_minimum_correct_targets(trial: TMTTrial, correct_targets_minimum: int, radius: float) -> TMTTrial:
    """
    Cuts the trial at the minimum number of correct targets.
    """
    correct_segments, _ = get_correct_and_incorrect_segments(trial, radius)
    if len(correct_segments) < correct_targets_minimum:
        raise ValueError(f"Trial {trial.id} has less than {correct_targets_minimum} correct targets.")

    # Sort correct segments by start time
    correct_segments.sort(key=lambda x: x[1].time)

    # Get the last correct segment
    last_correct_segment = correct_segments[correct_targets_minimum - 1]

    # Cut the trial at the end of the last correct segment
    cursor_info_first_entered_target = last_correct_segment[2]

    cut_trial = cut_at_time(trial, cursor_info_first_entered_target.time, correct_targets_minimum)

    return cut_trial


def cut_at_time(trial: TMTTrial, time: float, correct_targets_minimum: int) -> TMTTrial:
    """
    Cuts the trial at the given time.
    """
    new_cursor_trail = [cursor_info for cursor_info in trial.cursor_trail if cursor_info.time <= time]
    # rt is the total time the subject took to complete the trial
    cut_rt = time  # TODO GIAN: ver con gus
    cut_stimuli = trial.stimuli[:correct_targets_minimum]
    new_trial = TMTTrial(
        id=trial.id,
        stimuli=cut_stimuli,
        cursor_trail=new_cursor_trail,
        rt=cut_rt,
        trial_type=trial.trial_type,
        order_of_appearance=trial.order_of_appearance,
        with_custom_start=trial.with_custom_start,
        start=trial.start

    )
    return new_trial
