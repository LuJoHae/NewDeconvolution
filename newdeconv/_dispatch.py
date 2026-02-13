import numpy as np
from scipy.sparse import csr_matrix
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from cupy import ndarray as cp_ndarray
from typing import Literal, Callable
from typing import NamedTuple

from gibbscellprism._impl.insta_dense_cpu import _insta_prism_dense
from gibbscellprism._impl.insta_sparse_cpu import _insta_prism_sparse
from gibbscellprism._impl.insta_sparse_gpu import _insta_prism_sparse_gpu

ArraySparseLike = np.ndarray | cp_ndarray | csr_matrix | csr_matrix_gpu
ArrayDenseLike = np.ndarray | cp_ndarray


class Dispatch(NamedTuple):
    function: Callable
    type: ArraySparseLike | ArrayDenseLike


def insta_prism(bulk: ArrayDenseLike,             # shape (1, G) or (G,)
    reference: ArraySparseLike,        # shape (S, G): S cell types × G genes
    n_iter: int = 1000,
    n_debug: int = 100,
    device: Literal["cpu", "cuda"] = "cpu",
    layout: Literal["dense", "sparse"] = "sparse"):

    type_dispatch = {
        ("cpu", "dense"): Dispatch(_insta_prism_dense, ArrayDenseLike),
        ("cpu", "sparse"): Dispatch(_insta_prism_sparse, ArraySparseLike),
        ("cuda", "sparse"): Dispatch(_insta_prism_sparse_gpu, ArraySparseLike)
    }

    if not isinstance(reference, type_dispatch[(device, layout)].type):
        raise ValueError(f"Unsupported dispatch type: {type_dispatch[(device, layout)].type}! It")
    return type_dispatch[(device, layout)].function(bulk, reference, n_iter)
