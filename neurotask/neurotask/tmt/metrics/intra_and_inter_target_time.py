from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.metrics.targets_touched import correct_touched_targets_for_every_cursor_point
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class TargetTime(BaseMetricCalculator):
    def add_metrics(self, metrics: dict, trial: TMTTrial, subject: TMTSubject,
                    trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]], calculate_crosses: bool,
                    speed_threshold, consecutive_points, correct_intervals) -> dict:

        if subject is None:
            raise ValueError("Subject must be provided")

        return metrics


def calculate_intra_target_time(
        trial: TMTTrial,
        subject: TMTSubject
) -> float:
    """
    Calculate the intra-target time for a given trial and subject.
    The intra-target time is defined as the sum of the times spent within each target area.
    """
    # If cursor trail is empty, return 0
    if not trial.cursor_trail:
        return 0.0

    # Get correct touched targets for every cursor point
    correct_touches = correct_touched_targets_for_every_cursor_point(trial, subject.target_radius)

    # Calculate intra-target time
    total_time = 0.0

    for i in range(len(correct_touches)):
        target, cursor_info = correct_touches[i]

        # If a correct target is touched at this point
        if target is not None:
            # Calculate time spent at this point
            if i < len(correct_touches) - 1:
                next_cursor = correct_touches[i + 1][1]
                time_diff = next_cursor.time - cursor_info.time
                total_time += time_diff

    return total_time
