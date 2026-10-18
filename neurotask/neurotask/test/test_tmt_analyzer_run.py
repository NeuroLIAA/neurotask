from types import SimpleNamespace

import pytest

import neurotask.tmt.tmt_analyzer as tmt_analyzer_module
from neurotask.tmt.invalid_cause import InvalidCause
from neurotask.tmt.model.tmt_model import TMTTrial, TrialType
from neurotask.tmt.tmt_analyzer import TMTAnalyzer
from neurotask.test.test_helpers import build_cursor_trail


@pytest.fixture
def analyzer(tmp_path, monkeypatch):
    """
    TMTAnalyzer with a preloaded experiment and calculate_and_save_metrics replaced by a stub
    that records the arguments it receives.
    """
    captured = {}

    def fake_calculate_and_save_metrics(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(tmt_analyzer_module, "calculate_and_save_metrics", fake_calculate_and_save_metrics)

    tmt_analyzer = TMTAnalyzer(mapper=None, dataset_path="unused", output_path=str(tmp_path))
    tmt_analyzer.experiment = SimpleNamespace(subjects=[])
    return tmt_analyzer, captured


class TestRunParameters:

    def test_target_radius_multiplier_defaults_to_one(self, analyzer):
        tmt_analyzer, captured = analyzer

        tmt_analyzer.run(consecutive_points=5)

        assert captured["target_radius_multiplier"] == 1.0

    def test_runs_without_crosses_time_threshold_when_crosses_disabled(self, analyzer):
        tmt_analyzer, captured = analyzer

        tmt_analyzer.run(consecutive_points=5, calculate_crosses=False)

        assert captured["crosses_time_threshold"] is None

    def test_crosses_time_threshold_required_when_crosses_enabled(self, analyzer):
        tmt_analyzer, _ = analyzer

        with pytest.raises(ValueError, match="crosses_time_threshold must be provided"):
            tmt_analyzer.run(consecutive_points=5, calculate_crosses=True)

    def test_non_positive_crosses_time_threshold_raises(self, analyzer):
        tmt_analyzer, _ = analyzer

        with pytest.raises(ValueError, match="crosses_time_threshold must be a positive number"):
            tmt_analyzer.run(consecutive_points=5, calculate_crosses=True, crosses_time_threshold=0)

    def test_non_positive_target_radius_multiplier_raises(self, analyzer):
        tmt_analyzer, _ = analyzer

        with pytest.raises(ValueError, match="target_radius_multiplier must be a positive number"):
            tmt_analyzer.run(consecutive_points=5, target_radius_multiplier=0)


def test_trial_with_custom_start_but_no_start_is_invalid_without_raising():
    """
    A trial marked with_custom_start but without a start point (e.g. no first click recorded)
    must be reported as invalid instead of raising.
    """
    trial = TMTTrial(
        stimuli=[],
        cursor_trail=build_cursor_trail([(0, 0, 0), (1, 1, 10), (2, 2, 20)]),
        trial_type=TrialType.PART_A,
        id="no_start",
        order_of_appearance=0,
        rt=20,
        with_custom_start=True,
        start=None,
    )

    assert trial.is_valid() is False
    assert trial.get_invalid_cause() == InvalidCause.INVALID_START_CONFIGURATION
