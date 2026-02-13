import numpy as np
from numpy import typing as npt
from scipy.sparse import csr_matrix
from newdeconv import _logging


def _update_cell_fractions_estimate_by_fixpoint_inplace(cell_fractions: np.ndarray,
                                                        cell_state_gene_expression: np.ndarray) -> None:
    """
    Updates the cell fractions estimate in place by normalizing the sum of gene expression
    values across cell states. This function modifies the `cell_fractions` array directly based
    on calculations derived from `cell_state_gene_expression`.

    Args:
        cell_fractions: A NumPy array that will be updated in place to represent the normalized
            proportions of cell fractions. The values in this array are overwritten based on
            calculations from `cell_state_gene_expression`.
        cell_state_gene_expression: A NumPy 2D array containing gene expression values where
            rows represent genes and columns correspond to different cell states. The function
            calculates the sum across rows for each cell state and uses these values to update
            `cell_fractions`.
    """
    cell_fractions[:] = cell_state_gene_expression.sum(axis=0)
    assert np.all(cell_fractions > 0), str(cell_fractions)
    assert np.all(~np.isnan(cell_fractions)), str(cell_fractions)
    cell_fractions /= cell_fractions.sum()


def _validate_reference(reference: np.ndarray) -> np.ndarray:
    """
    Validates the provided reference data by ensuring it is a numeric 2D array.

    The function converts the input reference data into a NumPy array with
    a float64 data type and checks that it is a two-dimensional array. If the
    validation fails, an assertion error is raised.

    Args:
        reference: The input data array to validate. Must be convertible to a
            NumPy array with numeric data type and must have two dimensions.

    Returns:
        A NumPy array with dtype float64 representing the validated reference data.
    """
    if isinstance(reference, np.ndarray):
        _logging.logger.debug("Validated as array-like")
        assert np.issubdtype(reference.dtype, np.number)
        reference = np.asarray(reference, dtype=np.float64)  # Ensure numeric dtype
    else:
        raise AssertionError("reference must be a numeric array-like object")
    assert reference.ndim == 2, "reference must be a 2D array"
    return reference


def _validate_reference_sparse(reference: np.ndarray) -> np.ndarray:
    """
    Validates the provided reference data by ensuring it is a numeric 2D array.

    The function converts the input reference data into a NumPy array with
    a float64 data type and checks that it is a two-dimensional array. If the
    validation fails, an assertion error is raised.

    Args:
        reference: The input data array to validate. Must be convertible to a
            NumPy array with numeric data type and must have two dimensions.

    Returns:
        A NumPy array with dtype float64 representing the validated reference data.
    """
    if isinstance(reference, csr_matrix):
        _logging.logger.debug("Validated as sparse matrix")
        assert np.issubdtype(reference.dtype, np.number)
        reference = csr_matrix(reference, dtype=np.float64)  # Ensure numeric dtype
    else:
        raise AssertionError("reference must be a csr_matrix object")
    assert reference.ndim == 2, "reference must be a 2D array"
    return reference


def _initialize_deconvolution_arrays(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes and returns arrays used for the deconvolution process.

    This function creates and initializes three arrays necessary for the
    deconvolution process. The cell fractions array is initialized with
    equal values summing to 1, while the probability matrix and
    cell state gene expression matrix are pre-allocated with appropriate
    dimensions based on the reference matrix.

    Args:
        reference (numpy.ndarray): A 2D array representing the reference data,
            where rows correspond to genes, and columns correspond to cells.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: A 1D array representing cell fractions, initialized
              such that each cell has an equal fraction.
            - numpy.ndarray: A 2D array for cell state gene expression, pre-allocated
              based on the dimensions of the reference matrix.
            - numpy.ndarray: A 2D array for the probability matrix, pre-allocated
              based on the reference dimensions.
    """
    gene_number, cell_number = reference.shape  # (G, S)
    cell_fractions = np.full(cell_number, 1.0 / cell_number)
    probability_matrix = np.empty((gene_number, cell_number), dtype=np.float64)
    cell_state_gene_expression = np.empty((gene_number, cell_number), dtype=np.float64)
    return cell_fractions, cell_state_gene_expression, probability_matrix


def _initialize_deconvolution_sparse_matrices(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes and returns arrays used for the deconvolution process.

    This function creates and initializes three arrays necessary for the
    deconvolution process. The cell fractions array is initialized with
    equal values summing to 1, while the probability matrix and
    cell state gene expression matrix are pre-allocated with appropriate
    dimensions based on the reference matrix.

    Args:
        reference (numpy.ndarray): A 2D array representing the reference data,
            where rows correspond to genes, and columns correspond to cells.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: A 1D array representing cell fractions, initialized
              such that each cell has an equal fraction.
            - numpy.ndarray: A 2D array for cell state gene expression, pre-allocated
              based on the dimensions of the reference matrix.
            - numpy.ndarray: A 2D array for the probability matrix, pre-allocated
              based on the reference dimensions.
    """
    _, cell_number = reference.shape  # (G, S)
    cell_fractions = np.full(cell_number, 1.0 / cell_number)
    probability_matrix = csr_matrix((np.ones_like(reference.data), reference.indices.copy(), reference.indptr.copy()), shape=reference.shape, dtype=np.float64)
    cell_state_gene_expression = csr_matrix((np.ones_like(reference.data), reference.indices.copy(), reference.indptr.copy()), shape=reference.shape, dtype=np.float64)
    return cell_fractions, cell_state_gene_expression, probability_matrix


def _validate_bulk(bulk):
    """
    Validates the input bulk data for correct dimensionality and format.

    This function ensures that the input 'bulk' is either a 1D array or a 2D array
    with a single sample. If these conditions are not met, it raises an assertion
    error. The input is converted to a NumPy array and conditioned for further
    processing.

    Args:
        bulk: Input array-like object representing bulk data. It can be of any
            shape and will be validated to ensure it is either a 1D array or
            a 2D array with one sample.

    Returns:
        A 1D NumPy array corresponding to the validated bulk data.

    Raises:
        AssertionError: If the input 'bulk' has dimensions other than 1D, or 2D
            with a single sample.
    """
    bulk = np.asarray(bulk)
    if bulk.ndim == 2:
        assert bulk.shape[0] == 1, "bulk must have a single sample"
        X = bulk.ravel()  # shape (G,)
    elif bulk.ndim == 1:
        X = bulk
    else:
        raise AssertionError("bulk must be 1D or 2D with one sample")
    return X


def _calculate_fractions(x: npt.NDArray[np.int64 | np.float64]):
    """
    Calculates fractions from the input array by summing along the rows and normalizing the result.

    This function takes a NumPy array, computes the sum along its rows, and normalizes these sums
    to create a fraction representation. The output is a 1D array of fractions representing the relative
    contributions of row-wise sums to the overall sum.

    Args:
        x: A NumPy array containing integer or float values. The input array should have at least two dimensions.

    Returns:
        A 1D NumPy array of float values representing the normalized fractions derived from the input array.
    """
    theta = x.sum(axis=1)
    theta = theta / theta.sum()
    return theta


def _update_probability_matrix_inplace(reference: np.array, theta: np.array, probability_matrix: np.array) -> None:
    """
    Updates the probability matrix in-place based on the provided reference and theta arrays.

    This function multiplies the `reference` and `theta` arrays element-wise and updates the
    `probability_matrix` in place. Afterward, the updated `probability_matrix` rows are normalized
    so that the sum of their elements equals 1. It is assumed that no rows in the resulting
    probability matrix will have a sum of zero, thereby avoiding division by zero errors.

    Args:
        reference (np.array): A 2D array representing the reference data.
        theta (np.array): A 2D array with the same shape as `reference`.
        probability_matrix (np.array): A 2D array that will be updated in-place.
    """
    probability_matrix[:] = reference * theta  # (G, S)
    probability_matrix /= probability_matrix.sum(axis=1, keepdims=True)  # no divide-by-zero assumed
