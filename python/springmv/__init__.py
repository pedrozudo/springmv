"""JAX bindings for SpringMV's CUDA semiring CSR kernels."""

from ._logsumexp import csr_logsumexp as csr_logsumexp
from ._ops import csr_maxplus as csr_maxplus
from ._ops import csr_minplus as csr_minplus
from ._spmv import csr_spmv as csr_spmv
from ._structure import CsrStructure as CsrStructure
from ._transpose import csr_transpose as csr_transpose

__all__ = [
    "CsrStructure",
    "csr_logsumexp",
    "csr_maxplus",
    "csr_minplus",
    "csr_spmv",
    "csr_transpose",
]

__version__ = "0.1.0"
