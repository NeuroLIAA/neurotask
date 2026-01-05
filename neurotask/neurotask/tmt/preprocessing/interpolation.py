import numpy as np
from typing import List, Tuple

def interpolate_trajectory(
    x_coords: List[float], 
    y_coords: List[float], 
    timestamps: List[int], 
    target_freq_hz: int = 60
) -> Tuple[List[float], List[float], List[int]]:
    """
    Interpolates a trajectory (x, y, t) to have a constant sampling rate.
    
    Args:
        x_coords: List of X coordinates.
        y_coords: List of Y coordinates.
        timestamps: List of timestamps (in milliseconds).
        target_freq_hz: Target sampling frequency in Hz (default 60Hz).
        
    Returns:
        A tuple (new_x, new_y, new_t) with the interpolated data.
    """
    # 1. Basic validations
    if len(y_coords) < 2 or len(x_coords) < 2:
        raise ValueError("Too few samples to interpolate")
        
    x = np.array(x_coords, dtype=float)
    y = np.array(y_coords, dtype=float)
    t = np.array(timestamps, dtype=float)
    
    # 2. Handle temporal duplicates
    # Interpolation requires 't' to be strictly increasing.
    # Web browsers sometimes fire two events in the same millisecond.
    t_unique, unique_indices = np.unique(t, return_index=True)
    
    # If there were duplicates, keep only the unique ones
    if len(t_unique) != len(t):
        x = x[unique_indices]
        y = y[unique_indices]
        t = t_unique
        
    # If too few points remain after cleaning, return original (as lists)
    if len(t) < 2:
        raise ValueError("Too few samples to interpolate")

    # 3. Create the new temporal axis
    # Calculate period in ms (e.g., 60Hz -> ~16.66ms)
    period_ms = 1000.0 / target_freq_hz
    
    # Create an array from start to end with constant step
    t_new = np.arange(t[0], t[-1], period_ms)
    
    # Ensure the exact end point is included if it wasn't covered by the step
    if t_new[-1] < t[-1]:
        t_new = np.append(t_new, t[-1])

    # 4. Linear Interpolation
    # numpy.interp(new_x, known_x, known_y)
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    
    # 5. Return as lists (to maintain compatibility with current models)
    return x_new.tolist(), y_new.tolist(), t_new.astype(int).tolist()