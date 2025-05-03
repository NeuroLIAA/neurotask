from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject
from neurotask.tmt.segmentation.segmentation import calculate_segmentation_trial_metrics


class SegmentationMetricCalculator(BaseMetricCalculator):
    def add_metrics(self, metrics, trial: TMTTrial, **params):

        subject: TMTSubject = params.get('subject')
        if subject is None:
            raise ValueError("Subject must be provided")
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
