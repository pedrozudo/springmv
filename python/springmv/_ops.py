"""JAX-facing sparse reduction operations."""

from __future__ import annotations

import functools
from collections.abc import Callable
from numbers import Integral
from typing import Any

import jax
import numpy as np
from jax import ffi as jffi

from . import _ffi as _registrations  # noqa: F401
from ._structure import CsrStructure
from ._validation import as_float_vector


_SUBWARP_SIZES = frozenset((0, 1, 2, 4, 8, 16, 32))


def _validate_subwarp_size(subwarp_size: int) -> np.int32:
    if isinstance(subwarp_size, bool) or not isinstance(subwarp_size, Integral):
        raise TypeError("subwarp_size must be an integer")
    subwarp_size = int(subwarp_size)
    if subwarp_size not in _SUBWARP_SIZES:
        raise ValueError("subwarp_size must be 0, 1, 2, 4, 8, 16, or 32")
    return np.int32(subwarp_size)


@functools.cache
def _reduction_call(target: str, num_rows: int) -> Callable[..., jax.Array]:
    return jffi.ffi_call(
        target,
        jax.ShapeDtypeStruct((num_rows,), np.dtype(np.float32)),
    )


def _reduce(
    operation: str,
    structure: CsrStructure,
    x: Any,
    edge_values: Any | None,
    *,
    edge_values_name: str,
    subwarp_size: int,
) -> jax.Array:
    if not isinstance(structure, CsrStructure):
        raise TypeError("structure must be a CsrStructure")
    x_array = as_float_vector(x, "x")
    if x_array.shape[0] != structure.num_cols:
        raise ValueError("x length must equal structure.num_cols")

    max_row = np.int32(structure.max_nnz_per_row)
    subwarp = _validate_subwarp_size(subwarp_size)
    if edge_values is None:
        target = f"springmv_csr_{operation}_unweighted"
        return _reduction_call(target, structure.num_rows)(
            structure.row_offsets,
            structure.column_indices,
            x_array,
            max_nnz_per_row=max_row,
            subwarp_size=subwarp,
        )

    edge_values_array = as_float_vector(edge_values, edge_values_name)
    if edge_values_array.shape != structure.column_indices.shape:
        raise ValueError(
            f"{edge_values_name} and column_indices must have equal shapes"
        )
    target = f"springmv_csr_{operation}_weighted"
    return _reduction_call(target, structure.num_rows)(
        structure.row_offsets,
        structure.column_indices,
        edge_values_array,
        x_array,
        max_nnz_per_row=max_row,
        subwarp_size=subwarp,
    )


def csr_spmv(
    structure: CsrStructure,
    x: Any,
    *,
    values: Any | None = None,
    subwarp_size: int = 0,
) -> jax.Array:
    """Compute an arithmetic float32/int32 CSR matrix-vector product.

    Omitting ``values`` computes the unweighted row sums without storing or
    reading an edge-value array. The structure owns static scheduling metadata.
    """
    return _reduce(
        "spmv",
        structure,
        x,
        values,
        edge_values_name="values",
        subwarp_size=subwarp_size,
    )


def csr_logsumexp(
    structure: CsrStructure,
    log_x: Any,
    *,
    log_values: Any | None = None,
    subwarp_size: int = 0,
) -> jax.Array:
    """Compute a float32/int32 CSR product over the log semiring."""
    return _reduce(
        "logsumexp",
        structure,
        log_x,
        log_values,
        edge_values_name="log_values",
        subwarp_size=subwarp_size,
    )


def csr_minplus(
    structure: CsrStructure,
    x: Any,
    *,
    edge_values: Any | None = None,
    subwarp_size: int = 0,
) -> jax.Array:
    """Compute a float32/int32 CSR product over the min-plus semiring."""
    return _reduce(
        "minplus",
        structure,
        x,
        edge_values,
        edge_values_name="edge_values",
        subwarp_size=subwarp_size,
    )


def csr_maxplus(
    structure: CsrStructure,
    x: Any,
    *,
    edge_values: Any | None = None,
    subwarp_size: int = 0,
) -> jax.Array:
    """Compute a float32/int32 CSR product over the max-plus semiring."""
    return _reduce(
        "maxplus",
        structure,
        x,
        edge_values,
        edge_values_name="edge_values",
        subwarp_size=subwarp_size,
    )
