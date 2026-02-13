import numpy as np
import gibbscellprism._impl
from scipy.sparse import csr_matrix
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu

import gibbscellprism.utils


def test_insta_prism_sparse_cpu_and_gpu_equivalent():
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
        reference_sparse_gpu = csr_matrix_gpu(cp.array(reference))
        bulk_gpu = cp.array(bulk)

        B, Z, theta = gibbscellprism._impl.insta_sparse_cpu._insta_prism_sparse(bulk, reference_sparse, 10)
        B_gpu, Z_gpu, theta_gpu = gibbscellprism._impl.insta_sparse_gpu._insta_prism_sparse_gpu(bulk_gpu, reference_sparse_gpu, 10)
        assert isinstance(B_gpu, csr_matrix_gpu)
        B_cpu = B_gpu.get()
        assert isinstance(B_cpu, csr_matrix)
        assert isinstance(B, csr_matrix)
        assert B.shape == B_cpu.shape
        assert np.all(B.indices == B_cpu.indices)
        assert np.all(B.indptr == B_cpu.indptr)
        assert np.allclose(B.data, B_cpu.data, atol=0, rtol=1e-14)