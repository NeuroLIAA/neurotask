from abc import ABC, abstractmethod

from neurotask.tmt.metrics.metrics import calculate_total_distance, calculate_total_time


class BaseMetricCalculator(ABC):
    @abstractmethod
    def add_metrics(self, metrics: dict, trial, **params) -> dict:
        """
        Añade las claves/valores de esta métrica al dict `metrics`.
        `trial` es tu objeto TMTTrial (o las estructuras que uses).
        """
        pass


class TotalDistanceCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial, **params):
        metrics['total_distance'] = calculate_total_distance(trial.path)
        return metrics


class ReactionTimeCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial, **params):
        metrics['rt'] = calculate_total_time(trial.timestamps)
        return metrics


class SpeedMetricsCalculator(BaseMetricCalculator):
    def __init__(self, speed_threshold):
        self.speed_threshold = speed_threshold

    def add_metrics(self, metrics, trial, **params):
        speeds = compute_speeds(trial.path, trial.timestamps)
        metrics['mean_speed'] = speeds.mean()
        metrics['peak_speed'] = speeds.max()
        # …
        return metrics