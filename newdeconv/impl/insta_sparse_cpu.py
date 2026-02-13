import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
from gibbscellprism import _logging
from gibbscellprism._impl.core import _update_cell_fractions_estimate_by_fixpoint_inplace, _validate_reference_sparse, \
    _initialize_deconvolution_sparse_matrices, _validate_bulk


def _insta_prism_sparse(
    bulk: np.ndarray,             # shape (1, G) or (G,)
    reference: csr_matrix,        # shape (S, G): S cell types × G genes
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
    bulk = _validate_bulk(bulk)
    reference = _validate_reference_sparse(reference)
    reference = csr_matrix(reference.T)     # (G, S)

    _logging.logger.debug("Initializing deconvolution arrays")
    cell_fractions, cell_state_gene_expression, probability_matrix = _initialize_deconvolution_sparse_matrices(reference)
    for i in range(n_iter):
        if i <=2 or i % 100 == 0:
            _logging.logger.debug(f"Iteration {i}")
        _update_probability_matrix_inplace_sparse(reference, cell_fractions, probability_matrix)
        _update_cell_state_gene_expression_by_fixpoint_inplace_sparse(cell_state_gene_expression, bulk, probability_matrix)
        _update_cell_fractions_estimate_by_fixpoint_inplace(cell_fractions, cell_state_gene_expression)
    return probability_matrix, cell_state_gene_expression, cell_fractions


def _update_cell_state_gene_expression_by_fixpoint_inplace_sparse(
        cell_state_gene_expression: np.ndarray,
        bulk: np.ndarray,
        probability_matrix: np.ndarray
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


def _update_probability_matrix_inplace_sparse(reference: csr_matrix, theta: np.array, probability_matrix: csr_matrix) -> None:
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
    assert np.max(reference.indices) < len(theta), str([np.max(reference.indices), len(theta)])
    probability_matrix.data = reference.data * theta[reference.indices]  # (G, S)
    row_sums = np.array(probability_matrix.sum(axis=1)).ravel()
    for i in range(row_sums.shape[0]):
        probability_matrix.data[probability_matrix.indptr[i]:probability_matrix.indptr[i + 1]] /= row_sums[i]
