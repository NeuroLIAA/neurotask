import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .cut_criteria import CutCriteria
from .invalid_cause import InvalidCause
from .metrics import calculate_total_distance, number_of_correct_and_incorrect_segments, \
    calculate_speeds_between_cursor_positions, \
    calculate_accelerations_between_cursor_positions, \
    get_correct_and_incorrect_segments
from .model.tmt_model import TMTExperiment, TMTSubject, TMTTrial
from .segmentation.segmentation import calculate_segmentation_trial_metrics, \
    calculate_speed_threshold_for_all_subjects


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


def generate_rows_for_subject(
        subject_id: str,
        subject: TMTSubject,
        correct_targets_minimum: int,
        speed_threshold: float,
        consecutive_points: int,
        cut_criteria: CutCriteria
) -> List[Dict[str, Any]]:
    """
    Generate a list of row dictionaries, each describing metrics and information
    for valid and invalid trials of a single subject.

    For each trial:
      1. If the trial is invalid from the start, mark it as invalid.
      2. Otherwise, if cut criteria are provided, attempt to cut the trial.
         If cutting fails, mark the trial as invalid.
      3. Compute the number of correct and incorrect target touches.
         If the correct targets count is below the minimum, mark the trial invalid.
      4. If all checks pass, compute trial metrics and combine with general trial info.

    :param subject_id: A unique identifier for the subject.
    :param subject: The subject object containing personal info and trials.
    :param correct_targets_minimum: The minimum number of correct target touches required.
    :param speed_threshold: The speed threshold for calculations.
    :param consecutive_points: The number of consecutive points to consider.
    :param cut_criteria: The criteria to use for cutting trials; if None, no cutting is performed.
    :return: A list of dictionaries, each representing a trial (valid or invalid).
    """
    rows = []

    for trial in subject.testing_trials:
        # Check initial validity.
        if not trial.is_valid():
            logging.warning(f"Trial {trial.id} of subject {subject_id} is not valid from the mapper.")
            rows.append(
                create_invalid_trial_row(
                    subject, subject_id, trial, speed_threshold,
                    invalid_cause=InvalidCause.INVALID_MODEL
                )
            )
            continue

        try:
            processed_trial = trial

            # Apply cut criteria if provided.
            if cut_criteria is not None:
                processed_trial = _attempt_cut_trial(
                    trial, correct_targets_minimum, subject, subject_id, cut_criteria, speed_threshold, rows
                )
                if processed_trial is None:
                    continue

            # Compute target touches.
            correct_touches, wrong_touches = number_of_correct_and_incorrect_segments(
                processed_trial, subject.target_radius
            )

            # Check if the trial meets the minimum correct touches.
            if correct_targets_minimum is not None:
                if correct_touches < correct_targets_minimum:
                    logging.warning(
                        f"Trial {trial.id} of subject {subject_id} has {correct_touches} correct target touches, "
                        f"but the minimum required is {correct_targets_minimum}."
                    )
                    rows.append(
                        create_invalid_trial_row(
                            subject, subject_id, trial, speed_threshold,
                            invalid_cause=InvalidCause.UNDER_CORRECT_TARGETS_MINIMUM
                        )
                    )
                    continue

            # Compute trial metrics.
            trial_metrics = compute_trial_metrics(
                subject, processed_trial, correct_touches, wrong_touches, speed_threshold, consecutive_points
            )

            # Combine general trial info with metrics.
            valid_row = general_trial_info(speed_threshold, subject, subject_id, trial)
            valid_row.update(trial_metrics)
            rows.append(valid_row)

        except Exception as e:
            logging.exception(f"Error processing trial {trial.id} for subject {subject_id}: {e}")
            rows.append(
                create_invalid_trial_row(
                    subject, subject_id, trial, speed_threshold,
                    invalid_cause=InvalidCause.UNKNOWN_ERROR
                )
            )

    return rows


def _attempt_cut_trial(
        trial: Any,
        correct_targets_minimum: int,
        subject: TMTSubject,
        subject_id: str,
        cut_criteria: CutCriteria,
        speed_threshold: float,
        rows: List[Dict[str, Any]]
) -> Optional[Any]:
    """
    Attempt to cut a trial using the specified criteria.

    If cutting fails, the function logs a warning and appends an invalid trial row.

    :return: The processed trial if successful, or None if an error occurred.
    """
    try:
        return cut_trial(trial, correct_targets_minimum, subject, subject_id, cut_criteria)
    except Exception as e:
        logging.exception(f"Cut trial error for trial {trial.id} of subject {subject_id}: {e}")
        rows.append(
            create_invalid_trial_row(
                subject, subject_id, trial, speed_threshold,
                invalid_cause=InvalidCause.CUT_CRITERIA_ERROR
            )
        )
        return None


def general_trial_info(speed_threshold, subject, subject_id, trial):
    trial_row = {
        "subject_id": subject_id,
        "trial_id": trial.id,
        "trial_type": trial.trial_type.name,
        "age": subject.age(),
        "gender": subject.personal_info.gender,
        "is_valid": trial.is_valid(),
        "trial_order_of_appearance": trial.order_of_appearance,
        "speed_threshold": speed_threshold
    }
    return trial_row


def create_invalid_trial_row(
        subject: TMTSubject,
        subject_id: str,
        trial: TMTTrial,
        speed_threshold: float,
        invalid_cause: InvalidCause
) -> Dict[str, Any]:
    if invalid_cause == InvalidCause.INVALID_MODEL:
        if not trial.is_valid_start_configuration():
            invalid_cause = InvalidCause.INVALID_START_CONFIGURATION
        elif not trial.is_valid_length():
            invalid_cause = InvalidCause.INVALID_LENGTH

    return {
        "subject_id": subject_id,
        "trial_id": trial.id,
        "age": subject.age(),
        "gender": subject.personal_info.gender,
        "total_distance": 0,
        "rt": trial.rt,
        "is_valid": False,
        "trial_type": trial.trial_type.name,
        "trial_order_of_appearance": trial.order_of_appearance,
        "correct_targets_touches": np.nan,
        "wrong_targets_touches": np.nan,
        "speed_threshold": speed_threshold,
        "mean_speed": np.nan,
        "std_speed": np.nan,
        "peak_speed": np.nan,
        "mean_acceleration": np.nan,
        "std_acceleration": np.nan,
        "peak_acceleration": np.nan,
        "hesitation_distance": np.nan,
        "hesitation_time": np.nan,
        "invalid_cause": invalid_cause.name
    }


def compute_trial_metrics(
        subject: TMTSubject,
        trial: TMTTrial,
        correct_targets_touches: int,
        wrong_targets_touches: int,
        speed_threshold: float,
        consecutive_points: int
) -> Dict[str, Any]:
    """
    Compute all relevant metrics for a single trial in the TMT experiment.

    This function computes the following metrics:
    - Total distance
    - Reaction time (RT)
    - Number of correct and wrong target touches
    - Speed and acceleration metrics
    - Segmentation metrics
    - Number of crosses

    :param subject: The subject object to which the trial belongs.
    :param trial: The trial object for which to compute metrics.
    :param correct_targets_touches: The number of correct target touches in the trial.
    :param wrong_targets_touches: The number of wrong target touches in the trial.
    :param speed_threshold: The speed threshold for certain calculations.
    :param consecutive_points: The number of consecutive points to consider in the metrics.
    :return: A dictionary containing all computed metrics for the trial.
    """

    # Compute total distance, rt, correct and wrong targets touches
    row_metrics = {
        "total_distance": calculate_total_distance(trial),
        "rt": trial.rt,
        "correct_targets_touches": correct_targets_touches,
        "wrong_targets_touches": wrong_targets_touches
    }

    # Compute speed and acceleration metrics
    speed_and_acc_metrics = compute_speed_and_acceleration_metrics(trial)
    row_metrics.update(speed_and_acc_metrics)

    # Compute segmentation metrics
    segmentation_metrics = calculate_segmentation_trial_metrics(
        trial,
        subject.target_radius,
        speed_threshold,
        consecutive_points
    )
    row_metrics.update(segmentation_metrics)

    # Calculate the number of crosses (This takes too much time)
    number_of_crosses, _ = (0, None)  # TODO GIAN calculate_crosses(trial)
    row_metrics["number_of_crosses"] = number_of_crosses

    return row_metrics


def compute_speed_and_acceleration_metrics(
        trial: TMTTrial
) -> Dict[str, Any]:
    speeds = calculate_speeds_between_cursor_positions(trial)
    accelerations = calculate_accelerations_between_cursor_positions(trial)

    abs_accelerations = np.abs(accelerations)
    negative_accelerations = list(filter(lambda x: x < 0, accelerations))

    # Handle potential empty lists for negative_accelerations
    if len(negative_accelerations) > 0:
        mean_negative_acc = np.mean(negative_accelerations)
        std_negative_acc = np.std(negative_accelerations)
        peak_negative_acc = np.min(negative_accelerations)
    else:
        mean_negative_acc = np.nan
        std_negative_acc = np.nan
        peak_negative_acc = np.nan

    metrics = {

        "mean_speed": np.mean(speeds) if len(speeds) > 0 else np.nan,
        "std_speed": np.std(speeds) if len(speeds) > 0 else np.nan,
        "peak_speed": np.max(speeds) if len(speeds) > 0 else np.nan,

        "mean_acceleration": np.mean(accelerations) if len(accelerations) > 0 else np.nan,
        "std_acceleration": np.std(accelerations) if len(accelerations) > 0 else np.nan,
        "peak_acceleration": np.max(accelerations) if len(accelerations) > 0 else np.nan,

        "mean_abs_acceleration": np.mean(abs_accelerations) if len(abs_accelerations) > 0 else np.nan,
        "std_abs_acceleration": np.std(abs_accelerations) if len(abs_accelerations) > 0 else np.nan,
        "peak_abs_acceleration": np.max(abs_accelerations) if len(abs_accelerations) > 0 else np.nan,

        "mean_negative_acceleration": mean_negative_acc,
        "std_negative_acceleration": std_negative_acc,
        "peak_negative_acceleration": peak_negative_acc
    }

    return metrics


def calculate_and_save_metrics(experiment: TMTExperiment, save_path: str,
                               correct_targets_minimum: int, consecutive_point,
                               cut_criteria: CutCriteria) -> pd.DataFrame:
    rows = []

    speed_threshold_by_subject = calculate_speed_threshold_for_all_subjects(experiment)

    # Iteramos por cada sujeto en las métricas
    for subject_id, subject in experiment.subjects.items():
        try:
            subject_rows = generate_rows_for_subject(subject_id, subject, correct_targets_minimum,
                                                     speed_threshold_by_subject[subject_id], consecutive_point,
                                                     cut_criteria)
            rows.extend(subject_rows)
        except Exception:
            logging.exception(f"Error processing subject {subject_id}")
            continue

    # Convertimos la lista de filas a un DataFrame
    df = pd.DataFrame(rows)

    df.to_csv(save_path, index=False)

    return df


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
