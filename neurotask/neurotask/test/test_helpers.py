"""
Helper functions for test files.

This module provides common utility functions used across multiple test files
to reduce code duplication.
"""

from neurotask.tmt.model.tmt_model import (
    Coordinate,
    CursorInfo,
    TMTSubject,
    TMTTarget,
    TMTTrial,
    TrialType,
)


def build_cursor_trail(positions_and_times: list[tuple[float, float, float]]) -> list[CursorInfo]:
    """
    Build a cursor trail from a list of (x, y, time) tuples.

    :param positions_and_times: List of tuples containing (x, y, time) for each cursor position
    :return: List of CursorInfo objects
    """
    return [
        CursorInfo(Coordinate(x, y), t)
        for x, y, t in positions_and_times
    ]


def build_trial_and_subject(
        cursor_trail: list[CursorInfo],
        targets: list[TMTTarget] = None,
        target_radius: float = 1.0,
        with_custom_start: bool = False,
        start: CursorInfo = None,
        trial_id: str = "test_trial",
        trial_type: TrialType = TrialType.PART_A,
) -> tuple[TMTTrial, TMTSubject]:
    """
    Build a trial and subject with the given cursor trail and optional parameters.

    :param cursor_trail: List of CursorInfo objects representing the cursor movement
    :param targets: List of TMTTarget objects. If None, creates a default target at (0, 0)
    :param target_radius: Radius for target detection (default: 1.0)
    :param with_custom_start: Whether the trial has a custom start point (default: False)
    :param start: Custom start CursorInfo (required if with_custom_start=True)
    :param trial_id: Identifier for the trial (default: "test_trial")
    :param trial_type: Type of trial (default: TrialType.PART_A)
    :return: Tuple of (TMTTrial, TMTSubject)
    """
    if targets is None:
        targets = [TMTTarget("0", Coordinate(0.0, 0.0))]

    trial = TMTTrial(
        stimuli=targets,
        cursor_trail=cursor_trail,
        trial_type=trial_type,
        id=trial_id,
        order_of_appearance=1,
        rt=cursor_trail[-1].time if cursor_trail else 0.0,
        with_custom_start=with_custom_start,
        start=start,
    )

    subject = TMTSubject(
        training_trials=[],
        testing_trials=[trial],
        target_radius=target_radius,
        canvas_size=None,
    )

    return trial, subject


def build_trial(
        cursor_trail: list[CursorInfo],
        targets: list[TMTTarget] = None,
        trial_id: str = "test_trial",
        trial_type: TrialType = TrialType.PART_A,
) -> TMTTrial:
    """
    Build a minimal trial with the given cursor trail (without subject).

    This is useful for tests that only need a trial object.

    :param cursor_trail: List of CursorInfo objects representing the cursor movement
    :param targets: List of TMTTarget objects. If None, creates default targets
    :param trial_id: Identifier for the trial (default: "test_trial")
    :param trial_type: Type of trial (default: TrialType.PART_A)
    :return: TMTTrial object
    """
    if targets is None:
        targets = [
            TMTTarget("1", Coordinate(0.0, 0.0)),
            TMTTarget("2", Coordinate(50.0, 0.0)),
        ]

    return TMTTrial(
        stimuli=targets,
        cursor_trail=cursor_trail,
        trial_type=trial_type,
        id=trial_id,
        order_of_appearance=1,
        rt=cursor_trail[-1].time if cursor_trail else 0.0,
    )

