from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo
from neurotask.tmt.segmentation.segmentation import calculate_segmentation_trial_metrics


class SegmentationMetricCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses,
                    speed_threshold, consecutive_points, correct_intervals, wrong_intervals) -> dict:

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
