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


def _raw_reduction(
    operation: str,
    row_offsets: jax.Array,
    column_indices: jax.Array,
    x: jax.Array,
    edge_values: jax.Array | None,
    *,
    num_rows: int,
    max_nnz_per_row: int,
    subwarp_size: int,
) -> jax.Array:
    max_row = np.int32(max_nnz_per_row)
    subwarp = np.int32(subwarp_size)
    suffix = "weighted" if edge_values is not None else "unweighted"
    call = _reduction_call(
        f"springmv_csr_{operation}_{suffix}", num_rows
    )
    if edge_values is None:
        return call(
            row_offsets,
            column_indices,
            x,
            max_nnz_per_row=max_row,
            subwarp_size=subwarp,
        )
    return call(
        row_offsets,
        column_indices,
        edge_values,
        x,
        max_nnz_per_row=max_row,
        subwarp_size=subwarp,
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

    subwarp = int(_validate_subwarp_size(subwarp_size))
    if edge_values is None:
        return _raw_reduction(
            operation,
            structure.row_offsets,
            structure.column_indices,
            x_array,
            None,
            num_rows=structure.num_rows,
            max_nnz_per_row=structure.max_nnz_per_row,
            subwarp_size=subwarp,
        )

    edge_values_array = as_float_vector(edge_values, edge_values_name)
    if edge_values_array.shape != structure.column_indices.shape:
        raise ValueError(
            f"{edge_values_name} and column_indices must have equal shapes"
        )
    return _raw_reduction(
        operation,
        structure.row_offsets,
        structure.column_indices,
        x_array,
        edge_values_array,
        num_rows=structure.num_rows,
        max_nnz_per_row=structure.max_nnz_per_row,
        subwarp_size=subwarp,
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
