import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from neurotask.tmt.mapper.mapper import TMTMapper
from neurotask.tmt.metrics.metrics_calculator import calculate_and_save_metrics
from neurotask.tmt.model.tmt_model import TMTTarget, CursorInfo
from .cut_criteria.cut_criteria import CutCriteria


def _segment_to_dict(segment: Tuple['TMTTarget', 'CursorInfo', 'CursorInfo']) -> Dict[str, Any]:
    """
    Converts a segment tuple into a dictionary.
    Each segment tuple is expected to have the form:
    (TMTTarget, start_cursor (CursorInfo), end_cursor (CursorInfo))
    """
    target, start_cursor, end_cursor = segment
    return {
        "target": asdict(target),
        "start_cursor": asdict(start_cursor),
        "end_cursor": asdict(end_cursor)
    }


class TMTAnalyzer:
    """
    Encapsulates the analysis logic. When you instantiate TMTAnalysis, you provide
    the paths and parameters. Then, by calling `run()`, it will map the dataset,
    create the output directory, calculate and save metrics, and store them
    in a pandas DataFrame internally.
    """

    def __init__(
            self,
            mapper: TMTMapper,
            dataset_path: str,
            output_path: str,
    ):
        """
        Parameters
        ----------
        mapper : object
            An object that has a `.map(dataset_path, something)` method
            returning an Experiment-like structure.
        dataset_path : str
            Path to the dataset to be mapped.
        output_path : str
            Directory path where results (metrics.csv) will be stored.
        """
        self.mapper = mapper
        self.dataset_path = dataset_path
        self.output_path = output_path

        # Internally stored
        self.experiment = None
        self.metrics_df: Optional[pd.DataFrame] = None
        self.output_metrics_path: Optional[Path] = None

    def run(self, correct_targets_minimum: Optional[int] = None, consecutive_points: Optional[int] = None,
            cut_criteria: str = None, calculate_crosses=False) -> None:
        """
        Execute the analysis pipeline:
        1. Map the dataset to create an Experiment object.
        2. Create the output directory (if needed).
        3. Calculate and save metrics (csv) and store them in memory.
        """
        # 1. Map dataset -> Experiment
        self.experiment = self.mapper.map(self.dataset_path, None) if self.experiment is None else self.experiment

        logging.info(f"Experiment loaded. Number of subjects: {len(self.experiment.subjects)}")

        # 2. Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.output_metrics_path = Path(self.output_path) / "metrics.csv"

        # 3. Calculate and save metrics
        #    (Suponiendo que esta función retorna un DataFrame con las métricas)

        if consecutive_points is None:
            raise ValueError("consecutive_points must be provided")

        self.metrics_df = calculate_and_save_metrics(
            experiment=self.experiment,
            save_path=self.output_metrics_path,
            correct_targets_minimum=correct_targets_minimum,
            consecutive_points=consecutive_points,
            cut_criteria=CutCriteria(cut_criteria) if cut_criteria else None,
            calculate_crosses=calculate_crosses
        )

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Returns the DataFrame with metrics that was computed in `run()`.
        Raises an error if run() has not been called yet.
        """
        if self.metrics_df is None:
            raise RuntimeError("No metrics have been calculated yet. "
                               "Did you forget to call run()?")

        return self.metrics_df

    def get_experiment(self):
        """
        Returns the Experiment object that was created in `run()`.
        Raises an error if run() has not been called yet.
        """
        if self.experiment is None:
            self.experiment = self.mapper.map(self.dataset_path, None) if self.experiment is None else self.experiment

        return self.experiment

    def get_segments_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves detailed segment data for each trial in the experiment, grouped by subject.

        For each subject, a key is added to the returned dictionary, with its value being a list of
        dictionaries. Each dictionary in the list corresponds to a trial and contains:
          - trial_id: Identifier for the trial.
          - correct_segments: List of segments where correct targets were touched.
          - incorrect_segments: List of segments where incorrect targets were touched.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with subject_id as keys and lists of trial
            segment data as values.
        """
        if self.experiment is None:
            raise RuntimeError("No experiment data available. Did you forget to call run()?")

        segments_data: Dict[str, List[Dict[str, Any]]] = {}

        for subject_id, subject in self.experiment.subjects.items():
            trial_segments_list = []
            for trial in subject.testing_trials:
                try:
                    #TODO GIAN
                    correct_segments, incorrect_segments = [],[]
                    trial_segments = {
                        "trial_id": trial.id,
                        "correct_segments": [_segment_to_dict(seg) for seg in correct_segments],
                        "incorrect_segments": [_segment_to_dict(seg) for seg in incorrect_segments]
                    }
                    trial_segments_list.append(trial_segments)
                except Exception as e:
                    logging.error(f"Error processing trial {trial.id} for subject {subject_id}: {e}")
                    continue

            segments_data[subject_id] = trial_segments_list

        return segments_data

    #function to access by trial id in segments data
    def get_trial_segments_data(self, subject_id: str, trial_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves detailed segment data for a specific trial of a specific subject.

        Parameters:
            subject_id (str): The ID of the subject.
            trial_id (str): The ID of the trial.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the segment data for the specified trial,
            or None if the subject or trial is not found.
        """
        segments_data = self.get_segments_data()
        if subject_id in segments_data:
            for trial in segments_data[subject_id]:
                if trial["trial_id"] == trial_id:
                    return trial
        return None