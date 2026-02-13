import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu

from gibbscellprism import _logging
from gibbscellprism._impl.core import _update_cell_fractions_estimate_by_fixpoint_inplace


def _insta_prism_sparse_gpu(
    bulk: cp.ndarray,             # shape (1, G) or (G,)
    reference: csr_matrix_gpu,        # shape (S, G): S cell types × G genes
    n_iter: int = 1000
) -> np.array:
    """
    Performs deconvolution of gene expression data to estimate cell proportions,
    cell state gene expression, and probability matrix.

    This function estimates the contribution of each cell type to the observed
    gene expression data (bulk), using an iterative procedure. It updates the
    probability matrix, cell state gene expression, and cell fractions until
    a specified number of iterations is reached.

    Args:
        bulk: np.ndarray
            Gene expression data for a bulk sample. Can be a 1D array with
            shape (G,) or a 2D array with shape (1, G), where G is the number
            of genes.
        reference: np.ndarray
            Reference gene expression profiles for S cell types. Expected as a
            2D array with shape (S, G), where S is the number of cell types
            and G is the number of genes.
        n_iter: int
            The number of iterations for the iterative estimation process.

    Returns:
        np.ndarray
            Tuple containing three arrays:
            - probability_matrix: Estimates for the probability matrix of dimensions
              reflecting the relationships among genes and cell types.
            - cell_state_gene_expression: Gene expression matrix for each cell state.
            - cell_fractions: Estimated proportions of cell types in the bulk sample.
    """
    bulk = _validate_bulk_gpu(bulk)
    reference = _validate_reference_sparse_gpu(reference)
    reference = csr_matrix_gpu(reference.T)     # (G, S)

    _logging.logger.debug("Initializing deconvolution arrays")
    cell_fractions, cell_state_gene_expression, probability_matrix = _initialize_deconvolution_sparse_matrices_gpu(reference)
    for i in range(n_iter):
        if i <=2 or i % 100 == 0:
            _logging.logger.debug(f"Iteration {i}")
        _update_probability_matrix_inplace_sparse_gpu(reference, cell_fractions, probability_matrix)
        _update_cell_state_gene_expression_by_fixpoint_inplace_sparse_gpu(cell_state_gene_expression, bulk, probability_matrix)
        _update_cell_fractions_estimate_by_fixpoint_inplace(cell_fractions, cell_state_gene_expression)
    return probability_matrix, cell_state_gene_expression, cell_fractions


def _update_cell_state_gene_expression_by_fixpoint_inplace_sparse_gpu(
        cell_state_gene_expression: cp.ndarray,
        bulk: cp.ndarray,
        probability_matrix: cp.ndarray
) -> None:
    """
    Updates cell state gene expression matrix inplace by multiplying the bulk data
    with the probability matrix.

    This function performs element-wise multiplication of the provided bulk dataset
    with the probability matrix, broadcasting as required, and stores the result
    directly into the provided cell_state_gene_expression matrix. The input is
    modified in-place to improve performance and reduce memory overhead.

    Args:
        cell_state_gene_expression: Matrix that stores the final updated cell state
            gene expression values. The matrix is updated in-place.
        bulk: Array containing bulk data which will be used for updating the
            cell state gene expression matrix.
        probability_matrix: Matrix containing probability values for corresponding
            elements that will be used during the update process.

    """
    for i in range(probability_matrix.shape[0]):
        cell_state_gene_expression.data[cell_state_gene_expression.indptr[i]:cell_state_gene_expression.indptr[i + 1]] \
            = probability_matrix.data[probability_matrix.indptr[i]:probability_matrix.indptr[i + 1]] * bulk[i]


def _update_probability_matrix_inplace_sparse_gpu(reference: csr_matrix_gpu, theta: cp.array, probability_matrix: csr_matrix_gpu) -> None:
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

    assert reference.shape[1] == theta.shape[0], str([reference.shape, theta.shape])
    assert theta.ndim == 1, str(theta.shape)
    assert cp.max(reference.indices) < len(theta), str([cp.max(reference.indices), len(theta)])
    probability_matrix.data = reference.data * theta[reference.indices]  # (G, S)
    row_sums = cp.array(probability_matrix.sum(axis=1)).ravel()
    for i in range(row_sums.shape[0]):
        probability_matrix.data[probability_matrix.indptr[i]:probability_matrix.indptr[i + 1]] /= row_sums[i]


def _validate_reference_sparse_gpu(reference: np.ndarray) -> np.ndarray:
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
    if isinstance(reference, csr_matrix_gpu):
        _logging.logger.debug("Validated as sparse matrix")
        assert cp.issubdtype(reference.dtype, cp.number)
        reference = csr_matrix_gpu(reference, dtype=cp.float64)  # Ensure numeric dtype
    else:
        raise AssertionError("reference must be a csr_matrix object")
    assert reference.ndim == 2, "reference must be a 2D array"
    return reference


def _initialize_deconvolution_sparse_matrices_gpu(reference: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
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
    cell_fractions = cp.full(cell_number, 1.0 / cell_number)
    probability_matrix = csr_matrix_gpu((cp.ones_like(reference.data), reference.indices.copy(), reference.indptr.copy()), shape=reference.shape, dtype=cp.float64)
    cell_state_gene_expression = csr_matrix_gpu((cp.ones_like(reference.data), reference.indices.copy(), reference.indptr.copy()), shape=reference.shape, dtype=cp.float64)
    return cell_fractions, cell_state_gene_expression, probability_matrix


def _validate_bulk_gpu(bulk):
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
    bulk = cp.asarray(bulk)
    if bulk.ndim == 2:
        assert bulk.shape[0] == 1, "bulk must have a single sample"
        X = bulk.ravel()  # shape (G,)
    elif bulk.ndim == 1:
        X = bulk
    else:
        raise AssertionError("bulk must be 1D or 2D with one sample")
    return X
