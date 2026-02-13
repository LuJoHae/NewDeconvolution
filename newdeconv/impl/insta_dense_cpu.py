import numpy as np

from newdeconv import _logging
from newdeconv.impl.core import _update_cell_fractions_estimate_by_fixpoint_inplace, _validate_reference, \
    _initialize_deconvolution_arrays, _validate_bulk, _update_probability_matrix_inplace


def _insta_prism_dense(
    bulk: np.ndarray,             # shape (1, G) or (G,)
    reference: np.ndarray,        # shape (S, G): S cell types × G genes
    n_iter: int = 1000,
    n_debug: int = 100
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
    reference = _validate_reference(reference).T    # (G, S)

    _logging.logger.debug("Initializing deconvolution arrays")
    cell_fractions, cell_state_gene_expression, probability_matrix = _initialize_deconvolution_arrays(reference)
    for i in range(n_iter):
        assert reference.shape == cell_state_gene_expression.shape
        assert reference.shape == probability_matrix.shape
        assert bulk.shape == reference.shape[0:1]
        assert cell_fractions.shape == reference.shape[1:]
        if i <=10 or i % n_debug == 0:
            _logging.logger.debug(f"Iteration {i}")
        _update_probability_matrix_inplace(reference, cell_fractions, probability_matrix)
        _update_cell_state_gene_expression_by_fixpoint_inplace(cell_state_gene_expression, bulk, probability_matrix)
        _update_cell_fractions_estimate_by_fixpoint_inplace(cell_fractions, cell_state_gene_expression)
    return probability_matrix, cell_state_gene_expression, cell_fractions


def _update_cell_state_gene_expression_by_fixpoint_inplace(
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
    # np.multiply(bulk[:, np.newaxis], probability_matrix, out=cell_state_gene_expression)
    cell_state_gene_expression[:] = probability_matrix * bulk[:, np.newaxis]


