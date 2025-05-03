from abc import ABC, abstractmethod

from neurotask.tmt.metrics.metrics import calculate_total_distance
from neurotask.tmt.model.tmt_model import TMTTrial


class BaseMetricCalculator(ABC):
    @abstractmethod
    def add_metrics(self, metrics: dict, trial: TMTTrial, **params) -> dict:
        """
        Añade las claves/valores de esta métrica al dict `metrics`.
        `trial` es tu objeto TMTTrial (o las estructuras que uses).
        """
        pass


class TotalDistanceCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial: TMTTrial, **params):
        metrics['total_distance'] = calculate_total_distance(trial)
        return metrics


class ReactionTimeCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial: TMTTrial, **params):
        metrics['rt'] = trial.rt
        return metrics


class TargetsTouchesCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial: TMTTrial, **params):
        if 'correct_targets_touches' not in params:
            raise ValueError("correct_targets must be provided")

        if 'wrong_targets_touches' not in params:
            raise ValueError("wrong_targets must be provided")

        metrics['correct_targets_touches'] = params.get('correct_targets_touches', 0)
        metrics['wrong_targets_touches'] = params.get('wrong_targets_touches', 0)

        return metrics
