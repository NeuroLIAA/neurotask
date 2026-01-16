from typing import Dict, Any, List

import numpy as np

from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
from neurotask.tmt.model.tmt_model import TMTTrial, TMTSubject, TMTTarget, CursorInfo


class RefreshRateCalculator(BaseMetricCalculator):
    """
    Calculates refresh rate (sampling rate) metrics from the cursor trail.
    """

    def add_metrics(
        self,
        metrics: dict,
        trial: TMTTrial,
        subject: TMTSubject,
        trails_between_targets: list[tuple[TMTTarget, list[CursorInfo]]],
        calculate_crosses: bool,
        speed_threshold,
        consecutive_points
    ) -> dict:
        refresh_metrics = compute_refresh_rate_metrics(trial)

        for key, value in refresh_metrics.items():
            metrics[self.get_metric_name(key)] = value

        return metrics


def compute_refresh_rate_metrics(trial: TMTTrial) -> Dict[str, Any]:
    """
    Computes refresh rate statistics based on cursor timestamps.

    Timestamps are assumed to be in milliseconds. The refresh rate is calculated
    as 1000 / dt_ms to obtain Hz.

    Returns:
        Dict with mean_refresh_rate, median_refresh_rate, std_refresh_rate,
        min_refresh_rate, max_refresh_rate, sample_count_for_refresh_rate,
        and valid_interval_count_for_refresh_rate.
    """
    cursor_trail = trial.get_cursor_trail_from_start()

    raw_sample_count = len(cursor_trail)

    if raw_sample_count < 2:
        return _empty_metrics(raw_sample_count)

    time_intervals_ms = _calculate_time_intervals_ms(cursor_trail)

    if not time_intervals_ms:
        return _empty_metrics(raw_sample_count)

    # Convert ms to Hz: Hz = 1000 / dt_ms
    refresh_rates = [1000.0 / dt for dt in time_intervals_ms]

    return {
        "mean_refresh_rate": np.mean(refresh_rates),
        "median_refresh_rate": np.median(refresh_rates),
        "std_refresh_rate": np.std(refresh_rates),
        "min_refresh_rate": np.min(refresh_rates),
        "max_refresh_rate": np.max(refresh_rates),
        "sample_count_for_refresh_rate": raw_sample_count,
        "valid_interval_count_for_refresh_rate": len(time_intervals_ms)
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
        "mean_refresh_rate": np.nan,
        "median_refresh_rate": np.nan,
        "std_refresh_rate": np.nan,
        "min_refresh_rate": np.nan,
        "max_refresh_rate": np.nan,
        "sample_count_for_refresh_rate": raw_sample_count,
        "valid_interval_count_for_refresh_rate": valid_count
    }
