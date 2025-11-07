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


def get_intra_target_intervals(
        trial: TMTTrial,
        subject: TMTSubject
) -> list[tuple[TMTTarget, float, float]]:
    """
    Identify continuous intervals where the cursor is on a correct target.

    Returns a list of tuples (target, start_time, end_time) representing
    continuous periods where the cursor is on the correct expected target.

    An interval starts when the cursor enters a correct target and ends when:
    - The cursor leaves the target (moves to a non-target position)
    - The cursor moves to a different correct target
    - We reach the last cursor point (interval ends at that point's time)

    :param trial: TMTTrial instance
    :param subject: TMTSubject instance with target_radius
    :return: List of (target, start_time, end_time) tuples
    """
    if not trial.cursor_trail:
        return []

    # Get correct touched targets for every cursor point
    correct_touches = correct_touched_targets_for_every_cursor_point(trial, subject.target_radius)

    if not correct_touches:
        return []

    intervals = []
    current_interval_target = None
    current_interval_start = None

    for i in range(len(correct_touches)):
        target, cursor_info = correct_touches[i]

        if target is not None:
            # We're on a correct target
            if current_interval_target is None:
                # Start a new interval
                current_interval_target = target
                current_interval_start = cursor_info.time
            elif current_interval_target != target:
                # Different target - close previous interval at current point and start new one
                intervals.append((current_interval_target, current_interval_start, cursor_info.time))

                # Start new interval
                current_interval_target = target
                current_interval_start = cursor_info.time
        else:
            # Not on a target - close current interval if exists
            if current_interval_target is not None:
                # Close interval at current point (when we left the target)
                intervals.append((current_interval_target, current_interval_start, cursor_info.time))

                current_interval_target = None
                current_interval_start = None

    # Close any open interval at the end
    if current_interval_target is not None:
        # Use the last cursor time as end of interval
        last_cursor = correct_touches[-1][1]
        intervals.append((current_interval_target, current_interval_start, last_cursor.time))

    return intervals


def calculate_intra_target_time(
        trial: TMTTrial,
        subject: TMTSubject
) -> float:
    """
    Calculate the intra-target time for a given trial and subject.
    The intra-target time is defined as the sum of the times spent within each target area.

    This function first identifies continuous intervals where the cursor is on correct targets,
    then sums the duration of all those intervals.

    :param trial: TMTTrial instance
    :param subject: TMTSubject instance with target_radius
    :return: Total time spent on correct targets in seconds
    """
    # Get all intervals where cursor is on correct targets
    intervals = get_intra_target_intervals(trial, subject)

    # Sum the duration of all intervals
    total_time = sum(end_time - start_time for _, start_time, end_time in intervals)

    return total_time
