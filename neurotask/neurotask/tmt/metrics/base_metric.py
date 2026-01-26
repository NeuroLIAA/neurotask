from abc import ABC, abstractmethod
from typing import Optional

from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class BaseMetricCalculator(ABC):
    def __init__(self, prefix: Optional[str] = None):
        self.prefix = prefix

    def get_metric_name(self, metric_name: str) -> str:
        """
        Devuelve el nombre de la métrica con el prefijo si se ha definido.
        """
        if self.prefix:
            return f'{self.prefix}{metric_name}'
        return metric_name

    @abstractmethod
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, target_radius_multiplier: float,
                    time_threshold: float) -> dict:
        """
        Añade las claves/valores de esta métrica al dict `metrics`.
        `trial` es tu objeto TMTTrial (o las estructuras que uses).
        """
        pass


class ReactionTimeCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, target_radius_multiplier: float,
                    time_threshold: float) -> dict:
        metrics[self.get_metric_name('rt')] = trial.rt
        return metrics
