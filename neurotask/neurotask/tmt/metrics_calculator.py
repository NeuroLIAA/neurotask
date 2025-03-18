import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .metrics import calculate_total_distance, number_of_correct_and_incorrect_segments, \
    calculate_speeds_between_cursor_positions, \
    calculate_accelerations_between_cursor_positions, \
    get_correct_and_incorrect_segments
from .model.tmt_model import TMTExperiment, TMTSubject, TMTTrial
from .segmentation.segmentation import calculate_segmentation_trial_metrics, \
    calculate_speed_threshold_for_all_subjects


def generate_rows_for_subject(
        subject_id: str,
        subject: TMTSubject,
        correct_targets_minimum: int,
        speed_threshold: float,
        consecutive_points: int
) -> List[Dict[str, Any]]:
    """
    Generate a list of row dictionaries, each describing metrics and information
    for valid and invalid trials of a single subject.

    Steps:
    1. For each trial in a subject's testing trials:
       - Check if the trial is valid.
       - If valid, attempt to cut the trial at the specified minimum of correct target touches.
         * If successful, compute and update metrics.
         * If not, mark the trial as invalid.
       - If invalid from the start or an error occurs, mark the trial as invalid.
    2. Return a list of dictionaries, each containing relevant trial data and metrics.

    :param subject_id: A unique identifier for the subject.
    :param subject: The subject object containing personal info and trials.
    :param correct_targets_minimum: The minimum number of correct target touches required to consider the trial valid.
    :param speed_threshold: The speed threshold for certain calculations.
    :param consecutive_points: The number of consecutive points to consider in the metrics.
    :return: A list of dictionaries, each representing a trial (valid or invalid).
    """
    rows = []

    for trial in subject.testing_trials:
        # If the trial is not valid from the start, mark it as invalid.
        if not trial.is_valid():
            logging.warning(f"Trial {trial.id} of subject {subject_id} is not valid from the mapper.")
            rows.append(
                create_invalid_trial_row(subject, subject_id, trial, speed_threshold, invalid_cause="INVALID_MODEL"))
            continue

        try:

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
                rows.append(create_invalid_trial_row(subject, subject_id, trial, speed_threshold,
                                                     invalid_cause="MINIMUM_TARGETS"))
                continue

            cutoff_trial = cut_trial_at_minimum_correct_targets(
                trial,
                correct_targets_minimum,
                subject.target_radius
            )

            cutoff_correct_targets_touches, cutoff_wrong_targets_touches = number_of_correct_and_incorrect_segments(
                cutoff_trial,
                subject.target_radius
            )

            # Ensure that the cutoff was successful.
            if cutoff_correct_targets_touches != correct_targets_minimum:
                error_msg = (
                    f"Failed to properly cut trial {trial.id} of subject {subject_id} at "
                    f"{correct_targets_minimum} correct target touches. "
                    f"Obtained {cutoff_correct_targets_touches} correct target touches instead."
                )
                raise ValueError(error_msg)

            # Compute final metrics on the cutoff trial.
            trial_metrics = compute_trial_metrics(
                subject,
                cutoff_trial,
                cutoff_correct_targets_touches,
                cutoff_wrong_targets_touches,
                speed_threshold,
                consecutive_points
            )

            valid_trial_row = general_trial_info(speed_threshold, subject, subject_id, trial)
            valid_trial_row.update(trial_metrics)

            rows.append(valid_trial_row)


        except Exception as e:
            logging.error(f"Error processing trial {trial.id} for subject {subject_id}: {e}")
            logging.warning(f"Trial {trial.id} of subject {subject_id} is not valid because of error.")
            rows.append(create_invalid_trial_row(subject, subject_id, trial, speed_threshold, invalid_cause="ERROR"))

    return rows


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
        invalid_cause: str
) -> Dict[str, Any]:
    if invalid_cause == "INVALID_MODEL":
        if not trial.is_valid_start_configuration():
            invalid_cause = "INVALID_START_CONFIGURATION"
        elif not trial.is_valid_length():
            invalid_cause = "INVALID_LENGTH"

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
        "invalid_cause": invalid_cause
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
                               correct_targets_minimum: int, consecutive_point) -> pd.DataFrame:
    rows = []

    speed_threshold_by_subject = calculate_speed_threshold_for_all_subjects(experiment)

    # Iteramos por cada sujeto en las métricas
    for subject_id, subject in experiment.subjects.items():
        try:
            subject_rows = generate_rows_for_subject(subject_id, subject, correct_targets_minimum,
                                                     speed_threshold_by_subject[subject_id], consecutive_point)
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
