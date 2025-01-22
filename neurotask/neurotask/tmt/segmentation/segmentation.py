import logging
import math
from typing import List, Tuple, Dict, Optional

import numpy as np

from neurotask.tmt.metrics import calculate_distance, calculate_speeds, calculate_speeds_between_cursor_positions
from neurotask.tmt.model.tmt_model import CursorInfo, TMTTrial, TMTExperiment, Coordinate, TMTSubject, TrialType


def speed_increases_over_consecutive_points(
        speeds: List[float],
        cursor_index: int,
        speed_threshold: float,
        consecutive_points: int
) -> bool:
    """
    Determines if the speed has increased over a specified number of consecutive points beyond a given speed threshold.

    Parameters:
    - speeds: List of speed values between cursor positions.
    - cursor_index: The current index in the cursor trail (starting from 0).
    - speed_threshold: The minimum increase in speed between consecutive points to consider.
    - consecutive_points: Number of consecutive points over which the speed must increase.

    Returns:
    - True if the speed has increased over the required number of consecutive points, False otherwise.
    """
    # Adjust for the fact that speeds list has one less element than cursor_trail
    speed_index = cursor_index - 1  # Speeds are between cursor positions

    if speed_index < consecutive_points:
        return False  # Not enough previous speeds to evaluate

    for i in range(speed_index - consecutive_points + 1, speed_index + 1):
        if i <= 0:
            return False  # Not enough data
        current_speed = speeds[i]
        if current_speed <= speed_threshold:
            return False  # Speed did not increase sufficiently
    return True


def speed_decreases_over_consecutive_points(
        speeds: List[float],
        cursor_index: int,
        speed_threshold: float,
        consecutive_points: int
) -> bool:
    """
    Determines if the speed has decreased over a specified number of consecutive points beyond a given speed threshold.

    Parameters:
    - speeds: List of speed values between cursor positions.
    - cursor_index: The current index in the cursor trail (starting from 0).
    - speed_threshold: The minimum decrease in speed between consecutive points to consider.
    - consecutive_points: Number of consecutive points over which the speed must decrease.

    Returns:
    - True if the speed has decreased over the required number of consecutive points, False otherwise.
    """
    # Adjust for the fact that speeds list has one less element than cursor_trail
    speed_index = cursor_index - 1  # Speeds are between cursor positions

    if speed_index < consecutive_points:
        return False  # Not enough previous speeds to evaluate

    for i in range(speed_index - consecutive_points + 1, speed_index + 1):
        if i <= 0:
            return False  # Not enough data
        current_speed = speeds[i]
        if current_speed > speed_threshold:
            return False  # Speed did not decrease sufficiently
    return True


def classify_cursor_positions_with_hesitation(
        tmt_trial: TMTTrial,
        target_radius: float,
        speed_threshold,
        consecutive_points=5
) -> List[Tuple[str, CursorInfo]]:
    classified_positions = []
    cursor_trail = tmt_trial.get_cursor_trail_from_start()
    speeds = calculate_speeds_between_cursor_positions(tmt_trial)
    over_target_flags = calculate_over_targets(cursor_trail, target_radius, tmt_trial.stimuli)

    current_state = 'Search'
    last_target_position = over_target_flags[0][1]

    for i in range(len(cursor_trail)):
        cursor_info = cursor_trail[i]
        over_target = over_target_flags[i][0]

        if over_target:
            last_target_position = over_target_flags[i][1]
            current_state = 'Search'

        elif speed_increases_over_consecutive_points(speeds, i, speed_threshold, consecutive_points):
            current_state = 'Travel'

        elif (current_state == 'Travel' and
              speed_decreases_over_consecutive_points(speeds, i, speed_threshold, consecutive_points)):
            current_state = 'Hesitation'

        # change search to travel if the cursor is not over the target and is away from the target for a distance of target_radius
        elif current_state == 'Search' and not over_target:
            distance_to_target = calculate_distance(cursor_info.position, last_target_position)
            if distance_to_target > 2 * target_radius:
                current_state = 'Travel'

        else:
            # Maintain the current state
            current_state = current_state

        classified_positions.append((current_state, cursor_info))

    return classified_positions


def calculate_over_targets(cursor_trail, target_radius, stimuli_sequence) -> List[Tuple[bool, Coordinate]]:
    over_target_flags = []
    current_target_index = 0
    on_current_target = False  # Flag to track if cursor is on the current target

    for cursor_info in cursor_trail:
        cursor_pos = cursor_info.position
        over_target = False

        if current_target_index >= len(stimuli_sequence):
            # All targets have been processed
            over_target_flags.append(False)
            continue

        current_target = stimuli_sequence[current_target_index]
        target_pos = current_target.position
        distance_to_target = calculate_distance(cursor_pos, target_pos)

        if on_current_target:
            # Cursor was previously over the target, check if it still is
            if distance_to_target < target_radius:
                over_target = True
            else:
                # Cursor has moved off the target
                over_target = False
                on_current_target = False
                current_target_index += 1  # Move to the next target
        else:
            # Cursor was not over the target, check if it is now
            if distance_to_target < target_radius:
                over_target = True
                on_current_target = True
            else:
                over_target = False

        over_target_flags.append((over_target, target_pos))

    return over_target_flags


def calculate_time_in_states(classified_positions):
    state_times = {'Search': 0.0, 'Travel': 0.0, 'Hesitation': 0.0}
    previous_time = classified_positions[0][1].time
    previous_state = classified_positions[0][0]

    for i in range(1, len(classified_positions)):
        current_time = classified_positions[i][1].time
        current_state = classified_positions[i][0]
        time_diff = current_time - previous_time

        # Accumulate time for the previous state
        state_times[previous_state] += time_diff

        # Update for next iteration
        previous_time = current_time
        previous_state = current_state

    return state_times


def calculate_hesitation_periods(classified_positions):
    hesitation_periods = []
    in_hesitation = False
    start_time = 0.0

    for i in range(len(classified_positions)):
        state = classified_positions[i][0]
        time = classified_positions[i][1].time

        if state == 'Hesitation':
            if not in_hesitation:
                # Start of a hesitation period
                in_hesitation = True
                start_time = time
        else:
            if in_hesitation:
                # End of a hesitation period
                in_hesitation = False
                end_time = time
                duration = end_time - start_time
                hesitation_periods.append(duration)

    # Handle case where the trial ends while still in hesitation
    if in_hesitation:
        end_time = classified_positions[-1][1].time
        duration = end_time - start_time
        hesitation_periods.append(duration)

    total_hesitations = len(hesitation_periods)
    average_duration = sum(hesitation_periods) / total_hesitations if total_hesitations > 0 else 0
    max_duration = max(hesitation_periods) if hesitation_periods else 0

    return {
        'total_hesitations': total_hesitations,
        'average_duration': average_duration,
        'max_duration': max_duration,
        'hesitation_periods': hesitation_periods
    }


def calculate_distance_in_states(classified_positions):
    state_distances = {'Search': 0.0, 'Travel': 0.0, 'Hesitation': 0.0}
    previous_position = classified_positions[0][1].position
    previous_state = classified_positions[0][0]

    for i in range(1, len(classified_positions)):
        current_position = classified_positions[i][1].position
        current_state = classified_positions[i][0]

        # Calculate distance between positions
        distance = math.hypot(
            current_position.x - previous_position.x,
            current_position.y - previous_position.y
        )

        # Accumulate distance for the previous state
        state_distances[previous_state] += distance

        # Update for next iteration
        previous_position = current_position
        previous_state = current_state

    return state_distances


def calculate_average_speed_in_states(classified_positions):
    state_speeds = {'Search': [], 'Travel': [], 'Hesitation': []}
    previous_position = classified_positions[0][1].position
    previous_time = classified_positions[0][1].time
    previous_state = classified_positions[0][0]

    for i in range(1, len(classified_positions)):
        current_position = classified_positions[i][1].position
        current_time = classified_positions[i][1].time
        current_state = classified_positions[i][0]

        # Calculate distance and time difference
        distance = math.hypot(
            current_position.x - previous_position.x,
            current_position.y - previous_position.y
        )
        time_diff = current_time - previous_time

        if time_diff > 0:
            speed = distance / time_diff
            state_speeds[previous_state].append(speed)

        # Update for next iteration
        previous_position = current_position
        previous_time = current_time
        previous_state = current_state

    # Calculate average speeds
    average_speeds = {}
    for state, speeds in state_speeds.items():
        if speeds:
            average_speeds[state] = sum(speeds) / len(speeds)
        else:
            average_speeds[state] = 0.0

    return average_speeds


def calculate_state_transitions(classified_positions):
    transitions = 0
    previous_state = classified_positions[0][0]

    for i in range(1, len(classified_positions)):
        current_state = classified_positions[i][0]
        if current_state != previous_state:
            transitions += 1
        previous_state = current_state

    return transitions


def calculate_hesitation_ratio(classified_positions):
    state_times = calculate_time_in_states(classified_positions)
    travel_time = state_times['Travel'] + state_times['Hesitation']
    hesitation_time = state_times['Hesitation']

    if travel_time > 0:
        hesitation_ratio = hesitation_time / travel_time
    else:
        hesitation_ratio = 0.0

    return hesitation_ratio


def calculate_segmentation_trial_metrics(trial: TMTTrial, target_radius: float, speed_threshold: float,
                                         consecutive_points: int):
    if speed_threshold is None:
        raise ValueError("Speed threshold must be provided.")
    if speed_threshold <= 0:
        raise ValueError("Speed threshold must be positive.")
    if consecutive_points is None:
        raise ValueError("Number of consecutive points must be provided.")
    if consecutive_points <= 0:
        raise ValueError("Number of consecutive points must be positive.")

    classified_positions = classify_cursor_positions_with_hesitation(
        tmt_trial=trial,
        target_radius=target_radius,
        speed_threshold=speed_threshold,
        consecutive_points=consecutive_points
    )

    state_times = calculate_time_in_states(classified_positions)

    state_distances = calculate_distance_in_states(classified_positions)

    average_speeds = calculate_average_speed_in_states(classified_positions)
    state_transitions = calculate_state_transitions(classified_positions)
    hesitation_ratio = calculate_hesitation_ratio(classified_positions)

    metrics = {
        'hesitation_time': state_times['Hesitation'],
        'travel_time': state_times['Travel'],
        'search_time': state_times['Search'],

        'hesitation_distance': state_distances['Hesitation'],
        'travel_distance': state_distances['Travel'],
        'search_distance': state_distances['Search'],

        'hesitation_avg_speed': average_speeds['Hesitation'],
        'travel_avg_speed': average_speeds['Travel'],
        'search_avg_speed': average_speeds['Search'],

        'state_transitions': state_transitions,
        'hesitation_ratio': hesitation_ratio,
    }

    # merge metrics with hesitation_data
    hesitation_data = calculate_hesitation_periods(classified_positions)

    metrics.update(hesitation_data)

    return metrics


# The following functions are used to calculate the speed threshold for all subjects in the experiment.
def calculate_speed_threshold_for_all_subjects(experiment: TMTExperiment) -> Dict[str, float]:
    """
    Calculates the speed threshold for all subjects in the experiment.

    Parameters:
    - experiment: TMTExperiment object.

    Returns:
    - speed_thresholds: dict, where keys are subject IDs and values are the speed thresholds.
    """
    speed_thresholds = {}

    for subject_id, subject in experiment.subjects.items():
        try:
            speed_threshold = calculate_speed_threshold(subject)
            speed_thresholds[subject_id] = speed_threshold
        except ValueError as e:
            # Skip subject if an error occurs
            logging.warning(f"Error calculating speed threshold for subject {subject_id}: {e}")
            continue

    return speed_thresholds


def calculate_target_segments(trial: TMTTrial, target_radius: float, number_of_targets: Optional[int] = None) -> List[
    List[CursorInfo]]:
    """
    Splits the cursor trail into segments between targets.

    Parameters:
    - trial: TMTTrial object containing the cursor_trail and stimuli.
    - target_radius: float, the radius of the targets.

    Returns:
    - segments: List of lists of CursorInfo, where each sublist corresponds to movement between targets.
    """
    if number_of_targets is None:
        number_of_targets = len(trial.stimuli)

    cursor_trail = trial.get_cursor_trail_from_start()
    stimuli = trial.stimuli[: number_of_targets]  # Assuming this is the list of targets in the correct order
    segments = []
    segment_start_index = 0
    cursor_index = 0
    num_cursor_points = len(cursor_trail)

    for target in stimuli:
        target_found = False
        # Iterate through cursor_trail starting from cursor_index
        while cursor_index < num_cursor_points:
            cursor_info = cursor_trail[cursor_index]
            cursor_pos = cursor_info.position
            target_pos = target.position
            # Calculate distance between cursor position and target position
            distance = calculate_distance(cursor_pos, target_pos)
            if distance <= target_radius:
                # Cursor is over the target
                target_found = True
                break
            cursor_index += 1
        if target_found:
            # cursor_index is the index where the cursor is over the target
            segment = cursor_trail[segment_start_index: cursor_index + 1]
            segments.append(segment)
            # Update segment_start_index for next segment
            segment_start_index = cursor_index
            # Move to next cursor point for next iteration
            cursor_index += 1
        else:
            # Target not found in the remaining cursor trail
            # Append the remaining cursor trail as the last segment
            segment = cursor_trail[segment_start_index:]
            segments.append(segment)
            break  # No more targets to process

    return segments


def extract_second_segment(trial: TMTTrial, target_radius: float) -> List[CursorInfo]:
    """
    Calculates the time taken to reach the second target.

    Parameters:
    - trial: TMTTrial object containing the cursor_trail and stimuli.
    - target_radius: float, the radius of the targets.

    Returns:
    - time_to_second_target: float, the time taken to reach the second target.
    """
    segments = calculate_target_segments(trial, target_radius, number_of_targets=2)

    if len(segments) < 2:
        raise ValueError("Only one target found in the trial.")

    second_segment = segments[1]

    return second_segment


def calculate_speed_threshold(subject: TMTSubject) -> float:
    """
    Calculates the median speed threshold for a subject.
    For each trial, the speed threshold is calculated as the median speed during the second segment.
    The second segment is the movement between the first and second target.

    Parameters:
    - trial: TMTTrial object containing the cursor_trail and stimuli.
    - target_radius: float, the radius of the targets.

    Returns:
    - speed: float, the average speed during the second segment.
    """
    trials = subject.testing_trials
    speeds = []
    # iterate over all trials the first that does not fail is the one we use
    for trial in trials:
        if trial.trial_type == TrialType.PART_B:
            continue
        try:
            second_segment = extract_second_segment(trial, subject.target_radius)
            segment_speed = calculate_segment_speed(second_segment)
            speeds.append(segment_speed)
        except ValueError:
            continue

    if len(speeds) == 0:
        raise ValueError("No valid trial found to calculate speed threshold")

    # median of the speeds
    speed_threshold = np.percentile(speeds, 50)
    return speed_threshold


def calculate_segment_speed(segment: List[CursorInfo]) -> float:
    """
    Calculates the average speed of a segment.

    Parameters:
    - segment: List[CursorInfo], the list of cursor positions and times in the segment.

    Returns:
    - average_speed: float, the average speed over the segment.
    """
    speeds = calculate_speeds(segment)

    return np.mean(speeds)
