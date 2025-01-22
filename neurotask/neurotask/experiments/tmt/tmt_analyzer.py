import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from neurotask.experiments.tmt.metrics_calculator import calculate_and_save_metrics


class TMTAnalyzer:
    """
    Encapsulates the analysis logic. When you instantiate TMTAnalysis, you provide
    the paths and parameters. Then, by calling `run()`, it will map the dataset,
    create the output directory, calculate and save metrics, and store them
    in a pandas DataFrame internally.
    """

    def __init__(
            self,
            mapper,
            dataset_path: str,
            output_path: str,
            correct_targets_minimum: int,
            consecutive_points: int
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
        correct_targets_minimum : int
            Parameter to be passed to the metrics calculation function.
        consecutive_points : int
            Parameter to be passed to the metrics calculation function.
        """
        self.mapper = mapper
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.correct_targets_minimum = correct_targets_minimum
        self.consecutive_points = consecutive_points

        # Internally stored
        self.experiment = None
        self.metrics_df: Optional[pd.DataFrame] = None
        self.output_metrics_path: Optional[Path] = None

    def run(self, correct_targets_minimum: Optional[int] = None, consecutive_points: Optional[int] = None) -> None:
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

        ctm = correct_targets_minimum if correct_targets_minimum is not None else self.correct_targets_minimum
        cp = consecutive_points if consecutive_points is not None else self.consecutive_points

        self.metrics_df = calculate_and_save_metrics(
            experiment=self.experiment,
            save_path=self.output_metrics_path,
            correct_targets_minimum=ctm,
            consecutive_point=cp
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
            raise RuntimeError("No experiment has been loaded yet. "
                               "Did you forget to call run()?")
        return self.experiment
