from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo
from neurotask.tmt.segmentation.segmentation import calculate_segmentation_trial_metrics


class SegmentationMetricCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
                    **params) -> dict:

        speed_threshold = params.get('speed_threshold')
        if speed_threshold is None:
            raise ValueError("Speed threshold must be provided")
        consecutive_points = params.get('consecutive_points')
        if consecutive_points is None:
            raise ValueError("Consecutive points must be provided")

        segmentation = calculate_segmentation_trial_metrics(
            trial,
            subject.target_radius,
            speed_threshold,
            consecutive_points
        )

        metrics.update(segmentation)
        return metrics
