import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .cut_criteria.cut_criteria import CutCriteria
from .cut_criteria.cut_implementation import cut_trial
from .invalid_cause import InvalidCause
from .metrics import calculate_total_distance, number_of_correct_and_incorrect_segments, \
    calculate_speeds_between_cursor_positions, \
    calculate_accelerations_between_cursor_positions
from .model.tmt_model import TMTExperiment, TMTSubject, TMTTrial
from .segmentation.segmentation import calculate_segmentation_trial_metrics, \
    calculate_speed_threshold_for_all_subjects


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
      - Total distance traveled by the cursor.
      - Reaction time (rt) of the trial.
      - Number of correct and wrong target touches.
      - Speed and acceleration statistics.
      - Segmentation metrics.
      - (Stub) Number of crosses.

    Note:
      The number of crosses is currently a stub (set to 0) and should be replaced with an
      actual computation if needed.

    Args:
        subject (TMTSubject): The subject associated with the trial.
        trial (TMTTrial): The trial for which to compute metrics.
        correct_targets_touches (int): Number of correct target touches.
        wrong_targets_touches (int): Number of wrong target touches.
        speed_threshold (float): Speed threshold for segmentation metrics.
        consecutive_points (int): Number of consecutive points for segmentation metrics.

    Returns:
        Dict[str, Any]: A dictionary containing the computed metrics.
    """
    # Base metrics: distance, reaction time, and target touches.
    metrics = {
        "total_distance": calculate_total_distance(trial),
        "rt": trial.rt,
        "correct_targets_touches": correct_targets_touches,
        "wrong_targets_touches": wrong_targets_touches
    }

    # Add speed and acceleration metrics.
    metrics.update(compute_speed_and_acceleration_metrics(trial))

    # Add segmentation metrics.
    segmentation = calculate_segmentation_trial_metrics(
        trial,
        subject.target_radius,
        speed_threshold,
        consecutive_points
    )
    metrics.update(segmentation)

    # TODO GIAN: Replace stub with actual computation for the number of crosses.
    metrics["number_of_crosses"] = 0  # calculate_crosses(trial)

    return metrics


def safe_stats(data, peak_func=np.max) -> Tuple[float, float, float]:
    """
    Compute mean, standard deviation, and a peak value (using the provided peak function)
    for a list of numbers. If the list is empty, returns (np.nan, np.nan, np.nan).

    Args:
        data (Iterable[float]): The data from which to compute statistics.
        peak_func (Callable): Function to compute the peak value (default: np.max).

    Returns:
        Tuple[float, float, float]: (mean, std, peak_value)
    """
    if len(data) > 0:
        return np.mean(data), np.std(data), peak_func(data)
    return np.nan, np.nan, np.nan


def compute_speed_and_acceleration_metrics(trial: TMTTrial) -> Dict[str, Any]:
    """
    Compute and return speed and acceleration statistics from the trial's cursor movements.

    The metrics include:
      - Mean, standard deviation, and peak speed.
      - Mean, standard deviation, and peak acceleration.
      - Mean, standard deviation, and peak of the absolute acceleration.
      - Mean, standard deviation, and peak (minimum) negative acceleration.

    Args:
        trial (TMTTrial): The trial containing cursor movement data.

    Returns:
        Dict[str, Any]: A dictionary with computed speed and acceleration metrics.
    """
    speeds = calculate_speeds_between_cursor_positions(trial)
    accelerations = calculate_accelerations_between_cursor_positions(trial)
    abs_accelerations = np.abs(accelerations)
    negative_accelerations = [acc for acc in accelerations if acc < 0]

    mean_speed, std_speed, peak_speed = safe_stats(speeds)
    mean_acc, std_acc, peak_acc = safe_stats(accelerations)
    mean_abs_acc, std_abs_acc, peak_abs_acc = safe_stats(abs_accelerations)
    # For negative accelerations, use np.min to capture the most negative value.
    mean_neg_acc, std_neg_acc, peak_neg_acc = safe_stats(negative_accelerations, peak_func=np.min)

    return {
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "peak_speed": peak_speed,
        "mean_acceleration": mean_acc,
        "std_acceleration": std_acc,
        "peak_acceleration": peak_acc,
        "mean_abs_acceleration": mean_abs_acc,
        "std_abs_acceleration": std_abs_acc,
        "peak_abs_acceleration": peak_abs_acc,
        "mean_negative_acceleration": mean_neg_acc,
        "std_negative_acceleration": std_neg_acc,
        "peak_negative_acceleration": peak_neg_acc
    }


def calculate_and_save_metrics(experiment: TMTExperiment, save_path: str,
                               correct_targets_minimum: int, consecutive_point,
                               cut_criteria: CutCriteria) -> pd.DataFrame:
    rows = []

    speed_threshold_by_subject = calculate_speed_threshold_for_all_subjects(experiment)

    for subject_id, subject in experiment.subjects.items():
        try:
            subject_rows = generate_rows_for_subject(
                subject_id,
                subject,
                correct_targets_minimum,
                speed_threshold_by_subject[subject_id],
                consecutive_point,
                cut_criteria
            )

            rows.extend(subject_rows)
        except Exception:
            logging.exception(f"Error processing subject {subject_id}")
            continue

    df = pd.DataFrame(rows)

    df.to_csv(save_path, index=False)

    return df
