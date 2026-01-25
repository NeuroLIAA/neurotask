from typing import Dict, Any, List

import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class SamplingPeriodCalculator(BaseMetricCalculator):
    """
    Calculates sampling period (time between samples) metrics from the cursor trail.
    """

    def add_metrics(
        self,
        metrics: dict,
        trial: TMTTrial,
        subject: TMTSubject,
        trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
        calculate_crosses: bool,
        speed_threshold,
        consecutive_points,
        target_radius_multiplier: float
    ) -> dict:
        period_metrics = compute_sampling_period_metrics(trial)

        for key, value in period_metrics.items():
            metrics[self.get_metric_name(key)] = value

        return metrics


def compute_sampling_period_metrics(trial: TMTTrial) -> Dict[str, Any]:
    """
    Computes sampling period statistics based on cursor timestamps.

    Timestamps are assumed to be in milliseconds. The sampling period is the
    time interval between consecutive cursor samples.

    Returns:
        Dict with mean_sampling_period, median_sampling_period, std_sampling_period,
        min_sampling_period, max_sampling_period, sample_count,
        and valid_interval_count.
    """
    cursor_trail = trial.get_cursor_trail_from_start()

    raw_sample_count = len(cursor_trail)

    if raw_sample_count < 2:
        return _empty_metrics(raw_sample_count)

    time_intervals_ms = _calculate_time_intervals_ms(cursor_trail)

    if not time_intervals_ms:
        return _empty_metrics(raw_sample_count)

    return {
        "mean_sampling_period": np.mean(time_intervals_ms),
        "median_sampling_period": np.median(time_intervals_ms),
        "std_sampling_period": np.std(time_intervals_ms),
        "min_sampling_period": np.min(time_intervals_ms),
        "max_sampling_period": np.max(time_intervals_ms),
        "sample_count": raw_sample_count,
        "valid_interval_count": len(time_intervals_ms)
    }


def _calculate_time_intervals_ms(cursor_trail: List[CursorInfo]) -> List[float]:
    """Calculates valid time intervals (in ms) between consecutive samples."""
    intervals = []
    for i in range(1, len(cursor_trail)):
        dt = cursor_trail[i].time - cursor_trail[i - 1].time
        if dt > 0:
            intervals.append(dt)
    return intervals


def _empty_metrics(raw_sample_count: int, valid_count: int = 0) -> Dict[str, Any]:
    """Returns empty metrics when calculation is not possible."""
    return {
        "mean_sampling_period": np.nan,
        "median_sampling_period": np.nan,
        "std_sampling_period": np.nan,
        "min_sampling_period": np.nan,
        "max_sampling_period": np.nan,
        "sample_count": raw_sample_count,
        "valid_interval_count": valid_count
    }
