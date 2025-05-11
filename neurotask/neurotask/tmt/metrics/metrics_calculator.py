import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from neurotask.tmt.crosses.crosses_metric_calculator import CrossesMetricCalculator
from neurotask.tmt.metrics.speed_metrics import SpeedMetricsCalculator
from neurotask.tmt.metrics.zig_zag_amplitud import ZigZagAmplitude
from neurotask.tmt.segmentation.segmentation_metric import SegmentationMetricCalculator
from .base_metric import ReactionTimeCalculator, TotalDistanceCalculator, BaseMetricCalculator

from .targets_touch_calculator import number_of_correct_and_incorrect_segments
from .targets_touch_calculator import TargetsTouchesCalculator
from ..cut_criteria.cut_criteria import CutCriteria
from ..cut_criteria.cut_implementation import cut_trial
from ..invalid_cause import InvalidCause
from ..model.tmt_model import TMTExperiment, TMTSubject, TMTTrial
from ..segmentation.segmentation import calculate_speed_threshold_for_all_subjects


def generate_rows_for_subject(subject_id: str, subject: TMTSubject, correct_targets_minimum: int,
                              speed_threshold: float, consecutive_points: int, cut_criteria: CutCriteria,
                              calculate_crosses: bool) -> List[Dict[str, Any]]:
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
    :param calculate_crosses: Whether to calculate crosses.
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
            metric_calculators = get_metric_calculators()
            trial_metrics = compute_trial_metrics(
                processed_trial,
                metric_calculators,
                correct_targets_touches=correct_touches,
                wrong_targets_touches=wrong_touches,
                speed_threshold=speed_threshold,
                consecutive_points=consecutive_points,
                calculate_crosses=calculate_crosses,
                subject=subject,
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


def get_metric_calculators():
    return [
        ZigZagAmplitude(),
        TotalDistanceCalculator(),
        ReactionTimeCalculator(),
        SpeedMetricsCalculator(),
        SegmentationMetricCalculator(),
        TargetsTouchesCalculator(),
        CrossesMetricCalculator(),
    ]


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
        trial: TMTTrial,
        metric_calculators: List[BaseMetricCalculator],
        **params: Any
) -> Dict[str, Any]:
    """
    Itera sobre cada calculador de métricas y va acumulando
    sus resultados en un único dict.

    :param trial: objeto TMTTrial con la trayectoria y datos del ensayo.
    :param metric_calculators: lista de instancias de clases que implementan add_metrics().
    :param params: parámetros genéricos (ej. speed_threshold, cut_criteria, etc.).
    :return: dict con todas las métricas calculadas para este trial.
    """
    metrics: Dict[str, Any] = {}
    for calculator in metric_calculators:
        metrics = calculator.add_metrics(metrics, trial=trial, **params)
    return metrics


def calculate_and_save_metrics(
        experiment: TMTExperiment,
        save_path: str,
        correct_targets_minimum: int,
        consecutive_points: int,
        cut_criteria: CutCriteria,
        calculate_crosses: bool
) -> pd.DataFrame:
    """
    Calculate metrics for each subject in the experiment, save the results to a CSV file,
    and return the aggregated DataFrame.

    This function computes the speed threshold for each subject, then iterates over all subjects
    to generate trial metric rows using `generate_rows_for_subject`. The resulting rows are aggregated
    into a pandas DataFrame, saved as a CSV file to the specified path, and returned.

    Args:
        experiment (TMTExperiment): The experiment containing subjects and their trials.
        save_path (str): The file path where the CSV should be saved.
        correct_targets_minimum (int): The minimum number of correct target touches required.
        consecutive_points (int): The number of consecutive points for segmentation metrics.
        cut_criteria (CutCriteria): The criteria to use for cutting trials.
        calculate_crosses (bool): Whether to calculate crosses.

    Returns:
        pd.DataFrame: DataFrame containing all computed metrics for the experiment.
    """
    rows: List[Dict[str, Any]] = []

    # Compute speed thresholds for each subject.
    speed_threshold_by_subject = calculate_speed_threshold_for_all_subjects(experiment)

    for subject_id, subject in experiment.subjects.items():
        try:
            threshold = speed_threshold_by_subject.get(subject_id)
            if threshold is None:
                logging.warning(f"Speed threshold not found for subject {subject_id}. Skipping subject.")
                continue

            subject_rows = generate_rows_for_subject(
                subject_id=subject_id,
                subject=subject,
                correct_targets_minimum=correct_targets_minimum,
                speed_threshold=threshold,
                consecutive_points=consecutive_points,
                cut_criteria=cut_criteria,
                calculate_crosses=calculate_crosses
            )
            rows.extend(subject_rows)
        except Exception as e:
            logging.exception(f"Error processing subject {subject_id}: {e}")
            continue

    df = pd.DataFrame(rows)

    try:
        df.to_csv(save_path, index=False)
        logging.info(f"Metrics successfully saved to {save_path}.")
    except Exception as e:
        logging.exception(f"Error saving CSV to {save_path}: {e}")
        raise

    return df
