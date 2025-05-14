import numpy as np
import pytest
from neurotask.tmt.metrics.area_calculation import area_between_real_and_ideal_points


def test_colinear_points():
    """
    Colinear points along a straight line should yield zero area.
    """
    pts = np.array([[0, 0], [1, 1], [2, 2]])
    assert area_between_real_and_ideal_points(pts) == pytest.approx(0.0, abs=1e-8)


def test_simple_triangle_deviation():
    """
    Three points forming a symmetric 'triangle' deviation from the baseline
    from (0,0) to (2,0) should yield area = 1.0.
    """
    pts = np.array([[0, 0], [1, 1], [2, 0]])
    assert area_between_real_and_ideal_points(pts) == pytest.approx(1.0, rel=1e-12)


def test_return_type_and_non_negative():
    """
    The function should always return a non-negative float.
    """
    # Single point or empty should return 0.0
    assert isinstance(area_between_real_and_ideal_points(np.empty((0, 2))), float)
    assert area_between_real_and_ideal_points(np.array([[0, 0]])) == 0.0

    # Random noise path: area should be >= 0
    random_path = np.random.RandomState(0).randn(10, 2)
    area = area_between_real_and_ideal_points(random_path)
    assert isinstance(area, float)
    assert area >= 0.0


def test_square_deviation():
    """
    Four points making a square of side 2 above the x-axis (baseline from (0,0) to (2,0))
    should yield area = 2*2 = 4.0.
    """
    pts = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 2.0],
        [2.0, 0.0],
    ])
    # baseline is (0,0) → (2,0), so area under the "square bump" is exactly 4.0
    assert area_between_real_and_ideal_points(pts) == pytest.approx(4.0, rel=1e-12)


def test_square_both_sides():
    """
    Path with a 1×1 square above then a 1×1 square below the baseline
    from (0,0) to (2,0) should yield total absolute area = 2.0.
    """
    pts = np.array([
        [0.0, 0.0],
        [0.0, 1.0],  # start square above
        [1.0, 1.0],
        [1.0, 0.0],  # back to baseline at x=1
        [1.0, -1.0],  # start square below
        [2.0, -1.0],
        [2.0, 0.0],  # back to baseline at x=2
    ])
    assert area_between_real_and_ideal_points(pts) == pytest.approx(2.0, rel=1e-12)


def test_two_defined_triangles():
    """
    Dos triángulos consecutivos, uno de (0,0)->(2,0) con pico en (1,1)
    y otro de (2,0)->(4,0) con pico en (3,1), cada uno de área 1 → total = 2.
    """
    pts = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [4.0, 0.0],
    ])
    # Cada triángulo tiene área = (base=2 * altura=1) / 2 = 1 → suma = 2
    assert area_between_real_and_ideal_points(pts) == pytest.approx(2.0, rel=1e-12)
