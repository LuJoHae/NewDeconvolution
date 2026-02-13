import numpy as np
import gibbscellprism._dispatch
import gibbscellprism.utils
from scipy.sparse import csr_matrix
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from cupy import asarray as cp_asarray


def test_insta_prism_cpu_dense():
    c = 10
    g = 1000
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    cell_fractions = cell_numbers / cell_numbers.sum()
    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    B, Z, theta = gibbscellprism._dispatch.insta_prism(bulk, reference_row_normalized, 100, device="cpu", layout="dense")

    assert np.allclose(cell_fractions, theta, atol=1e-2, rtol=1e-8)


def test_insta_prism_cpu_sparse():
    c = 10
    g = 1000
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    cell_fractions = cell_numbers / cell_numbers.sum()
    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    B, Z, theta = gibbscellprism._dispatch.insta_prism(bulk, csr_matrix(reference_row_normalized),
                                                              100, device="cpu", layout="sparse")

    assert np.allclose(cell_fractions, theta, atol=1e-2, rtol=1e-8)


def test_insta_prism_gpu_sparse():
    c = 10
    g = 1000
    rng = np.random.default_rng(seed=0)

    cell_numbers = rng.integers(low=10, high=100, size=(c))
    reference = rng.integers(low=1, high=1000, size=(c, g))

    cell_fractions = cell_numbers / cell_numbers.sum()
    reference_row_normalized = gibbscellprism.utils._normalize_rows_to_stochastic(reference)
    bulk = cell_numbers.reshape(1, c) @ reference

    B, Z, theta = gibbscellprism._dispatch.insta_prism(cp_asarray(bulk), csr_matrix_gpu(csr_matrix(reference_row_normalized)),
                                                              100, device="cuda", layout="sparse")

    assert np.allclose(cell_fractions, theta, atol=1e-2, rtol=1e-8)
