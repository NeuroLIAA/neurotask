from typing import Optional

from neurotask.tmt.cut_criteria.cut_criteria import CutCriteria
from neurotask.tmt.metrics.intra_and_inter_target_time import get_intra_target_intervals
from neurotask.tmt.metrics.targets_touched import count_correctly_touched_targets
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


def cut_trial(
        trial: TMTTrial,
        correct_targets_minimum: Optional[int],
        subject: TMTSubject,
        subject_id: str,
        cut_criteria: CutCriteria
) -> TMTTrial:
    """
    Process the given trial based on the specified cut criteria.

    Args:
        trial (TMTTrial): The trial to process.
        correct_targets_minimum (Optional[int]): The minimum correct target touches required.
        subject (TMTSubject): The subject associated with the trial.
        subject_id (str): Unique identifier for the subject.
        cut_criteria (CutCriteria): The criteria used to cut the trial.

    Returns:
        TMTTrial: The processed (cut) trial.

    Raises:
        ValueError: If the cut criteria is invalid or if required parameters are missing.
    """
    if cut_criteria == CutCriteria.MINIMUM_TARGETS:
        if correct_targets_minimum is None:
            raise ValueError("Minimum targets criteria requires a minimum number of correct targets.")
        return cut_trial_at_minimum_targets(correct_targets_minimum, subject, subject_id, trial)

    raise ValueError(f"Invalid cut criteria: {cut_criteria}")


def cut_trial_at_minimum_targets(
        correct_targets_minimum: int,
        subject: TMTSubject,
        subject_id: str,
        trial: TMTTrial
) -> TMTTrial:
    """
    Cuts the trial at the point where the minimum number of correct target touches is reached.

    Args:
        correct_targets_minimum (int): The minimum number of correct touches required.
        subject (TMTSubject): The subject associated with the trial.
        subject_id (str): Unique identifier for the subject.
        trial (TMTTrial): The trial to be processed.

    Returns:
        TMTTrial: The trial cut at the minimum correct target touch.

    Raises:
        ValueError: If the trial does not meet the required number of correct target touches.
    """

    correct_targets_touches = count_correctly_touched_targets(trial, subject.target_radius)

    if correct_targets_touches < correct_targets_minimum:
        raise ValueError(
            f"Trial {trial.id} of subject {subject_id} has {correct_targets_touches} correct target touches, "
            f"but the minimum required is {correct_targets_minimum}.")

    return cut_trial_at_minimum_correct_targets(trial, correct_targets_minimum, subject)


def cut_trial_at_minimum_correct_targets(
        trial: TMTTrial,
        correct_targets_minimum: int,
        subject: TMTSubject,
) -> TMTTrial:
    """
    Cuts the trial at the time corresponding to reaching the minimum correct target touches.

    Args:
        trial (TMTTrial): The trial to be cut.
        correct_targets_minimum (int): The required number of correct touches.
        subject (TMTSubject): The subject with target_radius used to determine correct touches.

    Returns:
        TMTTrial: The trial cut at the appropriate time.

    Raises:
        ValueError: If the trial does not contain the required number of correct target segments.
    """
    intra_target_interval: list[tuple[TMTTarget, CursorInfo, CursorInfo]] = get_intra_target_intervals(trial, subject)

    if len(intra_target_interval) < correct_targets_minimum:
        raise ValueError(f"Trial {trial.id} has less than {correct_targets_minimum} correct targets.")

    # Get the Nth interval (where N = correct_targets_minimum)
    # Since correct_touched_targets_for_every_cursor_point ensures targets are touched in order,
    # the Nth interval corresponds to the Nth target in trial.stimuli
    cutoff_interval = intra_target_interval[correct_targets_minimum - 1]

    # Use the start cursor (index 1) to cut when the target is first touched
    cursor_info = cutoff_interval[1]

    return cut_at_time(trial, cursor_info.time, correct_targets_minimum)


def cut_at_time(
        trial: TMTTrial,
        cutoff_time: float,
        correct_targets_minimum: int
) -> TMTTrial:
    """
    Cuts the trial at the specified cutoff time.

    This function creates a new trial instance where the cursor trail is truncated
    at the given cutoff time and only the first 'correct_targets_minimum' stimuli are retained.

    Args:
        trial (TMTTrial): The trial to be cut.
        cutoff_time (float): The time at which to cut the trial.
        correct_targets_minimum (int): The number of stimuli to retain.

    Returns:
        TMTTrial: A new trial instance cut at the specified time.
    """
    new_cursor_trail = [cursor for cursor in trial.cursor_trail if cursor.time <= cutoff_time]
    new_rt = cutoff_time  # Ensure this behavior is as intended.
    new_stimuli = trial.stimuli[:correct_targets_minimum]

    return TMTTrial(
        id=trial.id,
        stimuli=new_stimuli,
        cursor_trail=new_cursor_trail,
        rt=new_rt,
        trial_type=trial.trial_type,
        order_of_appearance=trial.order_of_appearance,
        with_custom_start=trial.with_custom_start,
        start=trial.start
    )
