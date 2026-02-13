import numpy as np
from numpy.random import Generator

from newdeconv.impl.core import _validate_reference, _initialize_deconvolution_arrays, _validate_bulk, _update_probability_matrix_inplace


def _bayes_prism(
    bulk: np.ndarray,             # shape (1, G) or (G,)
    reference: np.ndarray,        # shape (S, G): S cell types × G genes
    n_iter: int = 1000,
    alpha: float = 1e-4,
    rng: Generator = np.random.default_rng(0),
):
    """
    Performs Bayesian deconvolution of bulk gene expression data using a reference
    matrix of cell-type-specific gene expression.

    This function applies an iterative approach to estimate two key outputs:
    cell state gene expression and cell type fractions. Using Bayesian updates,
    it refines these estimates based on given input data.

    Args:
        bulk (np.ndarray): A 1-dimensional array of bulk gene expression data with
            shape `(1, G)` or `(G,)`, where `G` is the number of genes.
        reference (np.ndarray): A 2-dimensional reference matrix with shape `(S, G)`,
            where `S` is the number of cell types, and `G` is the number of genes.
        n_iter (int): Number of iterations for the Bayesian updates. Defaults to 1000.
        alpha (float): Dirichlet prior parameter for cell fractions. Defaults to 1e-4.
        rng (Generator): A NumPy random number generator instance for control over
            the random sampling process. Defaults to a generator initialized with
            `np.random.default_rng(0)`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - cell_state_gene_expression (np.ndarray): A matrix representing the
              updated cell state gene expression.
            - cell_fractions (np.ndarray): An array representing the estimated
              proportions of each cell type.
    """
    bulk = _validate_bulk(bulk)
    reference = _validate_reference(reference).T   # (G, S)

    cell_fractions, cell_state_gene_expression, probability_matrix = _initialize_deconvolution_arrays(reference)
    for _ in range(n_iter):
        _update_probability_matrix_inplace(reference, cell_fractions, probability_matrix)
        _update_cell_state_gene_expression_by_sampling_inplace(cell_state_gene_expression, bulk, probability_matrix, rng)
        _update_cell_fractions_estimate_by_sampling_inplace(cell_fractions, cell_state_gene_expression, alpha, rng)
    return cell_state_gene_expression, cell_fractions


def _update_cell_fractions_estimate_by_sampling_inplace(
        cell_fractions: np.array,
        cell_state_gene_expression: np.array, alpha: float,
        rng: Generator
) -> None:
    """
    Updates the cell fractions estimate in place by sampling from a Dirichlet distribution.

    This function modifies the input cell fractions directly, updating its values based on a
    Dirichlet distribution. The parameters of the distribution are derived from the sum of
    the cell state gene expression and the provided alpha value.

    Args:
        cell_fractions: numpy array, modified in place to reflect updated cell fraction
            estimates.
        cell_state_gene_expression: numpy array representing gene expression levels for
            different cell states.
        alpha: float, the concentration parameter added to the summed gene expression to
            form the parameters of the Dirichlet distribution.
        rng: numpy.random.Generator, used to sample from the Dirichlet distribution.
    """
    cell_fractions[:] = rng.dirichlet(cell_state_gene_expression.sum(axis=0) + alpha)


def _update_cell_state_gene_expression_by_sampling_inplace(
        cell_state_gene_expression: np.ndarray,
        bulk: np.ndarray,
        probability_matrix: np.ndarray,
        rng: Generator
) -> None:
    """
    Updates the `cell_state_gene_expression` matrix by sampling using a multinomial distribution.

    This function modifies the `cell_state_gene_expression` in place based on the values of the
    `bulk` vector and the `probability_matrix`. The sampling is performed row-wise in the
    `cell_state_gene_expression` matrix using the multinomial distribution with parameters
    provided in `bulk` and `probability_matrix` for each row. The randomness of the sampling
    process is managed by the provided random number generator `rng`.

    Args:
        cell_state_gene_expression (np.ndarray): A 2D array representing the cell state gene
            expression to be updated in place.
        bulk (np.ndarray): A 1D array containing the total values for each gene to use in the
            multinomial sampling process.
        probability_matrix (np.ndarray): A 2D array representing the probability matrix
            used for the multinomial sampling, where each row corresponds to a gene, and the
            columns correspond to probabilities for each cell state.
        rng (np.random.Generator): A random number generator instance used for multinomial
            sampling to ensure reproducibility across runs.
    """
    for g in range(cell_state_gene_expression.shape[0]):
        cell_state_gene_expression[g, :] = rng.multinomial(bulk[g], probability_matrix[g, :])
