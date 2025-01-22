from typing import List, Tuple

from neuropsych.experiments.tmt.model.tmt_model import TMTTrial, Coordinate


def calculate_crosses(trial: TMTTrial, time_threshold: float = 500) -> Tuple[
    int, List[Tuple[Tuple[Coordinate, Coordinate], Tuple[Coordinate, Coordinate], float]]]:
    """
    Calculates the number of times the cursor trail crosses itself, excluding segments that are very near in time.

    Parameters:
    - trial: TMTTrial object containing the cursor_trail.
    - time_threshold: float, the minimum time difference between segments to consider them for intersection (in seconds).

    Returns:
    - num_crosses: int, the number of times the cursor trail crosses itself.
    - cross_segments: list of tuples, each containing two crossing segments.
    """
    # Extract positions from the cursor trail
    cursor_trail = trial.get_cursor_trail_from_start()

    # Create list of line segments with time information
    segments = []
    for i in range(len(cursor_trail) - 1):
        p1 = cursor_trail[i].position
        p2 = cursor_trail[i + 1].position
        t1 = cursor_trail[i].time
        t2 = cursor_trail[i + 1].time
        segments.append(((p1, t1), (p2, t2)))

    num_crosses = 0
    cross_segments = []

    # Iterate over all pairs of non-adjacent segments
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            # Skip adjacent segments (they share a point)
            if j == i + 1:
                continue

            # Calculate the minimum time difference between segments
            t_start_i = min(segments[i][0][1], segments[i][1][1])
            t_end_i = max(segments[i][0][1], segments[i][1][1])
            t_start_j = min(segments[j][0][1], segments[j][1][1])
            t_end_j = max(segments[j][0][1], segments[j][1][1])

            # Skip segments that are too close in time
            skip_subject = False
            for t1 in [t_start_i, t_end_i, t_start_j, t_end_j]:
                for t2 in [t_start_i, t_end_i, t_start_j, t_end_j]:
                    if t1 == t2:
                        continue
                    time_diff = t1 - t2
                    if abs(time_diff) < time_threshold:
                        skip_subject = True
                    else:
                        # If at least one pair of points is far enough in time, do not skip
                        skip_subject = False
                        break

            if skip_subject:
                continue  # Segments are too close in time, skip intersection test

            seg1 = (segments[i][0][0], segments[i][1][0])
            seg2 = (segments[j][0][0], segments[j][1][0])

            if segments_intersect(seg1[0], seg1[1], seg2[0], seg2[1]):
                num_crosses += 1
                cross_segments.append((seg1, seg2, time_diff))

    return num_crosses, cross_segments


def orientation(p: Coordinate, q: Coordinate, r: Coordinate) -> int:
    """
    Determines the orientation of the triplet (p, q, r).

    Returns:
    - 0 : Colinear
    - 1 : Clockwise
    - 2 : Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if abs(val) < 1e-10:
        return 0  # Colinear
    elif val > 0:
        return 1  # Clockwise
    else:
        return 2  # Counterclockwise


def on_segment(p: Coordinate, q: Coordinate, r: Coordinate) -> bool:
    """
    Checks if point q lies on segment pr.
    """
    if min(p.x, r.x) <= q.x <= max(p.x, r.x) and \
            min(p.y, r.y) <= q.y <= max(p.y, r.y):
        return True
    return False


def segments_intersect(p1: Coordinate, q1: Coordinate, p2: Coordinate, q2: Coordinate) -> bool:
    """
    Checks if two line segments (p1,q1) and (p2,q2) intersect.
    """
    # Find the four orientations needed for general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False
