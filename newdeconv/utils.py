import numpy as np
from numpy import typing as npt


def _normalize_rows_to_stochastic(x: npt.NDArray[np.int64 | np.float64]):
    """
    Normalizes the rows of a 2D array to make them stochastic.

    This function ensures that the rows of the input 2D array sum to 1, effectively
    turning the array into a row-stochastic matrix. Each row is divided by the sum
    of its elements.

    Args:
        x: A 2D NumPy array of integers or floats. The array must have two dimensions
           (ndim=2).

    Returns:
        A 2D NumPy array where each row has been normalized such that the sum of
        elements in each row equals 1. The output array will have the same shape as
        the input array.

    Raises:
        AssertionError: If the input array is not a 2D array (ndim != 2).
    """
    assert x.ndim == 2, "Input must be a 2D array"
    return x / x.sum(axis=1).reshape((x.shape[0], 1))
