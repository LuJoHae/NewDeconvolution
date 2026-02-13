import numpy as np

import newdeconv.validation
import newdeconv.utils

def _deconvolution(bulk: np.array, reference: np.array, n: int, eps: float) -> np.array:
    """
    Performs a deconvolution process on the provided bulk dataset using a reference matrix.

    This function implements an iterative computational deconvolution to infer proportions
    or contributions of the reference dataset to the bulk dataset.

    Args:
        bulk (np.array): A 1-dimensional array representing the bulk dataset, where
            each element captures an aggregate data point.
        reference (np.array): A 2-dimensional array where each row represents a
            reference signature, and columns indicate features. The shape of this
            array should be (c, g), where 'c' is the number of reference elements
            and 'g' is the number of features/signatures.
        n (int): The number of iterations to perform for the deconvolution process.
        eps (float): A small epsilon value added to avoid division by zero during the
            normalization process.

    Returns:
        np.array: A 2-dimensional array representing the result of the deconvolution
        process. The output matrix has the same shape as the reference matrix, where
        each entry indicates the inferred proportion of contribution.
    """
    c, g = reference.shape

    # Initialize expression matrix
    B = np.ones(shape=(c, g)) / np.repeat(c * g, c * g).reshape((c, g))

    # Iteration scheme
    for i in range(n):
        print()
        B = reference * np.repeat(B.sum(axis=1), g).reshape((c, g))
        B = B / (np.repeat(B.sum(axis=0), c).reshape((g, c)).T + eps)
        B = B * np.repeat(bulk, c).reshape((g, c)).T

    return B
