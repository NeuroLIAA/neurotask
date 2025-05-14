from abc import ABC, abstractmethod

from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class BaseMetricCalculator(ABC):
    @abstractmethod
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
                    calculate_crosses: bool,
                    **params) -> dict:
        """
        Añade las claves/valores de esta métrica al dict `metrics`.
        `trial` es tu objeto TMTTrial (o las estructuras que uses).
        """
        pass


class ReactionTimeCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
                    calculate_crosses: bool,
                    **params) -> dict:
        metrics['rt'] = trial.rt
        return metrics
