import numpy as np
import gibbscellprism._impl
from scipy.sparse import csr_matrix

import gibbscellprism.utils


def test_deconvolution():
    c = 10
    g = 1000
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    cell_fractions = cell_numbers / cell_numbers.sum()
    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    single_cell_estimate = gibbscellprism._deconvolution(bulk, reference_row_normalized, 100, 0.0)

    fractions_estimate = gibbscellprism._impl.core._calculate_fractions(single_cell_estimate)

    fractions_estimate_error = np.abs(cell_fractions - fractions_estimate)
    print(fractions_estimate_error)
    assert np.all(fractions_estimate_error < 1e-2)


def test_insta_prism():
    c = 10
    g = 1000
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    cell_fractions = cell_numbers / cell_numbers.sum()
    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    B, Z, theta = gibbscellprism._impl.insta_dense_cpu._insta_prism_dense(bulk, reference_row_normalized, 100)

    fractions_estimate_error = np.abs(cell_fractions-theta)
    assert np.all(fractions_estimate_error < 1e-2)


def test_bayes_prism():
    c = 4
    g = 200
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    cell_fractions = cell_numbers / cell_numbers.sum()
    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    Z, theta = gibbscellprism._impl.bayes_dense_cpu._bayes_prism(bulk, reference_row_normalized, 1000)

    fractions_estimate_error = np.abs(cell_fractions-theta)
    print(cell_fractions)
    assert np.all(fractions_estimate_error < 1e-2)


def test_deconvolution_and_insta_prism_equivalent():
    c = 10
    g = 1000
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c,))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    B, Z, theta = gibbscellprism._impl.insta_dense_cpu._insta_prism_dense(bulk, reference_row_normalized, 100)

    single_cell_estimate = gibbscellprism._deconvolution(bulk, reference_row_normalized, 100, 0.0)
    fractions_estimate = gibbscellprism._impl.core._calculate_fractions(single_cell_estimate)

    fractions_estimate_error = np.abs(fractions_estimate-theta)
    assert np.all(fractions_estimate_error < 1e-8)


def test_update_cell_state_gene_expression_by_fixpoint_inplace_dense_and_sparse_equivalent():

    rng = np.random.default_rng(seed=0)
    for i in range(10):
        c = rng.integers(low=10, high=100)
        g = rng.integers(low=10, high=100)

        cell_state_gene_expression = rng.exponential(scale=1, size=(g,c))
        probability_matrix = rng.exponential(scale=1, size=(g, c))
        bulk = rng.integers(low=1, high=100, size=(g,))

        gibbscellprism._impl.insta_dense_cpu._update_cell_state_gene_expression_by_fixpoint_inplace(
            cell_state_gene_expression=cell_state_gene_expression,
            bulk=bulk,
            probability_matrix=probability_matrix
        )

        cell_state_gene_expression_sparse = csr_matrix(cell_state_gene_expression)
        probability_matrix_sparse = csr_matrix(probability_matrix)

        gibbscellprism._impl.insta_sparse_cpu._update_cell_state_gene_expression_by_fixpoint_inplace_sparse(
            cell_state_gene_expression=cell_state_gene_expression_sparse,
            bulk=bulk,
            probability_matrix=probability_matrix_sparse
        )

        assert np.all(probability_matrix_sparse.todense() == probability_matrix)
        assert np.all(cell_state_gene_expression_sparse.todense() == cell_state_gene_expression)


def test_update_probability_matrix_inplace_dense_and_sparse_equivalent():

    rng = np.random.default_rng(seed=0)
    for i in range(10):
        c = rng.integers(low=10, high=100)
        g = rng.integers(low=10, high=100)

        reference = rng.exponential(scale=1, size=(g, c))
        reference_ = reference.copy()
        probability_matrix = rng.exponential(scale=1, size=(g, c))
        probability_matrix_ = probability_matrix.copy()
        theta = np.ones(shape=(c,))
        theta /= theta.sum()

        reference_sparse: csr_matrix = csr_matrix(reference).copy()
        probability_matrix_sparse: csr_matrix = csr_matrix(probability_matrix).copy()

        gibbscellprism._impl.core._update_probability_matrix_inplace(
            reference=reference,
            theta=theta,
            probability_matrix=probability_matrix
        )

        gibbscellprism._impl.insta_sparse_cpu._update_probability_matrix_inplace_sparse(
            reference=reference_sparse,
            theta=theta,
            probability_matrix=probability_matrix_sparse
        )

        assert np.all(reference == reference_)
        assert np.all(reference_sparse.todense() == reference_)
        assert np.all(reference_sparse.todense() == reference)

        assert np.any(probability_matrix != probability_matrix_)
        assert np.any(probability_matrix_sparse.todense() != probability_matrix_)

        assert not np.any(np.isnan(probability_matrix.data))
        assert not np.any(np.isnan(probability_matrix_sparse.data))

        assert np.all(probability_matrix_sparse.indices == reference_sparse.indices)
        assert np.all(probability_matrix_sparse.indptr == reference_sparse.indptr)

        assert probability_matrix_sparse.shape == probability_matrix.shape
        assert np.allclose(probability_matrix_sparse.todense(), probability_matrix, atol=0, rtol=1e-15)
        assert np.allclose(probability_matrix, probability_matrix_sparse.todense(), atol=0, rtol=1e-15)


def test_insta_prism_sparse_and_dense_equivalent():
    rng = np.random.default_rng(seed=0)
    for i in range(10):
        c = rng.integers(low=10, high=100)
        g = rng.integers(low=10, high=100)

        cell_numbers = rng.integers(low=10, high=100, size=(c,))
        reference = rng.integers(low=1, high=1000, size=(c, g))

        reference = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
        bulk = (cell_numbers.reshape(1, c) @ reference).ravel()
        assert bulk.shape == (g,)
        assert bulk.shape == (g,)
        assert reference.shape == (c, g)

        reference_sparse = csr_matrix(reference)

        B, Z, theta = gibbscellprism._impl.insta_dense_cpu._insta_prism_dense(bulk, reference, 10)
        B_sparse, Z_sparse, theta_sparse = gibbscellprism._impl.insta_sparse_cpu._insta_prism_sparse(bulk, csr_matrix(reference_sparse), 10)

        assert np.allclose(B, B_sparse.todense())
        assert np.allclose(Z, Z_sparse.todense())
        assert np.allclose(theta, theta_sparse)



