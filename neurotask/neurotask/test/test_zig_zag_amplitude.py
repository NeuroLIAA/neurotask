import numpy as np
import pytest

from neurotask.tmt.metrics.zig_zag_amplitud import ZigZagAmplitude
from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTarget,
    TMTTrial,
    TrialType,
)


def _build_targets(contents: list[str]) -> list[TMTTarget]:
    return [
        TMTTarget(content, Coordinate(idx * 10.0, 0.0))
        for idx, content in enumerate(contents)
    ]


def _build_trial_and_subject(
        contents: list[str],
        touches: list[tuple[str, float]],
        trial_type: TrialType,
) -> tuple[TMTTrial, TMTSubject]:
    targets = _build_targets(contents)
    position_by_content = {target.content: target.position for target in targets}

    cursor_trail: list[CursorInfo] = []
    for label, timestamp in touches:
        if label == "off":
            cursor_trail.append(CursorInfo(Coordinate(500.0 + timestamp, 500.0), timestamp))
        else:
            cursor_trail.append(CursorInfo(position_by_content[label], timestamp))

    trial = TMTTrial(
        stimuli=targets,
        cursor_trail=cursor_trail,
        trial_type=trial_type,
        id=f"trial_{trial_type.value.lower()}",
        order_of_appearance=1,
        rt=cursor_trail[-1].time if cursor_trail else 0.0,
    )

    subject = TMTSubject(
        training_trials=[],
        testing_trials=[trial],
        target_radius=1.0,
        canvas_size=None,
    )

    return trial, subject


def _compute_metrics(trial: TMTTrial, subject: TMTSubject) -> dict:
    return ZigZagAmplitude().add_metrics(
        metrics={},
        trial=trial,
        subject=subject,
        trails_between_targets=[],
        calculate_crosses=False,
        speed_threshold=None,
        consecutive_points=None,
        target_radius_multiplier=1.0,
        crosses_time_threshold=500,
    )


def test_letter_and_number_latencies_are_computed():
    sequence = ["1", "A", "2", "B", "3", "C"]
    touches = [
        ("1", 0.0), ("1", 1.0), ("off", 1.5),
        ("A", 2.0), ("A", 3.0), ("off", 3.5),
        ("2", 4.0), ("2", 5.0), ("off", 5.5),
        ("B", 6.0), ("B", 7.0), ("off", 7.5),
        ("3", 8.0), ("3", 9.0), ("off", 9.5),
        ("C", 10.0), ("C", 11.0),
    ]

    trial, subject = _build_trial_and_subject(sequence, touches, TrialType.PART_B)

    metrics = _compute_metrics(trial, subject)

    assert metrics["letter_to_number_latency"] == pytest.approx(2.0)
    assert metrics["number_to_letter_latency"] == pytest.approx(2.0)


def test_initial_number_to_letter_path_is_counted():
    sequence = ["1", "A", "2"]
    touches = [
        ("1", 0.0), ("1", 1.0), ("off", 1.5),
        ("A", 2.0), ("A", 3.0), ("off", 3.5),
        ("2", 4.0), ("2", 5.0),
    ]

    trial, subject = _build_trial_and_subject(sequence, touches, TrialType.PART_B)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_to_letter_latency"] == pytest.approx(2.0)
    assert metrics["letter_to_number_latency"] == pytest.approx(2.0)


def test_non_part_b_trials_return_nan():
    sequence = ["1", "A", "2"]
    touches = [
        ("1", 0.0), ("1", 1.0), ("off", 1.5),
        ("A", 2.0), ("A", 3.0), ("off", 3.5),
        ("2", 4.0), ("2", 5.0),
    ]

    trial, subject = _build_trial_and_subject(sequence, touches, TrialType.PART_A)

    metrics = _compute_metrics(trial, subject)

    assert np.isnan(metrics["letter_to_number_latency"])
    assert np.isnan(metrics["number_to_letter_latency"])


def test_initial_number_to_letter_path_is_counted():
    sequence = ["1", "A", "2"]
    touches = [
        ("1", 0.0), ("1", 1.0), ("off", 1.5),
        ("A", 2.0), ("A", 3.0), ("off", 3.5),
        ("2", 4.0), ("2", 5.0),
    ]

    trial, subject = _build_trial_and_subject(sequence, touches, TrialType.PART_B)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_to_letter_latency"] == pytest.approx(2.0)
    assert metrics["letter_to_number_latency"] == pytest.approx(2.0)


def test_invalid_alternation_raises_assertion():
    sequence = ["1", "A", "B"]
    touches = [
        ("1", 0.0), ("1", 1.0), ("off", 1.5),
        ("A", 2.0), ("A", 3.0), ("off", 3.5),
        ("B", 4.0), ("B", 5.0),
    ]

    trial, subject = _build_trial_and_subject(sequence, touches, TrialType.PART_B)

    with pytest.raises(AssertionError):
        _compute_metrics(trial, subject)


def test_initial_number_to_letter_path_is_included():
    sequence = ["1", "A", "2"]
    touches = [
        ("1", 0.0), ("1", 1.0), ("off", 1.5),
        ("A", 3.0), ("A", 4.0), ("off", 4.5),
        ("2", 6.0), ("2", 7.0),
    ]

    trial, subject = _build_trial_and_subject(sequence, touches, TrialType.PART_B)

    metrics = _compute_metrics(trial, subject)

    assert metrics["number_to_letter_latency"] == pytest.approx(3.0)
    assert metrics["letter_to_number_latency"] == pytest.approx(3.0)

