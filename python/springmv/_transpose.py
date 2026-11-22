"""Eager-only CSR transpose preprocessing."""

from __future__ import annotations

import functools
from collections.abc import Callable
from numbers import Integral
from typing import Any

import jax
import numpy as np
from jax import ffi as jffi

from . import _ffi as _registrations  # noqa: F401
from ._validation import as_float_vector, as_structure_array, require_eager_cuda


@functools.cache
def _transpose_call(
    num_cols: int,
    num_nonzeros: int,
    weighted: bool,
    return_permutation: bool,
) -> Callable[..., tuple[jax.Array, ...]]:
    outputs: list[jax.ShapeDtypeStruct] = [
        jax.ShapeDtypeStruct((num_cols + 1,), np.dtype(np.int32)),
        jax.ShapeDtypeStruct((num_nonzeros,), np.dtype(np.int32)),
    ]
    if weighted:
        outputs.append(
            jax.ShapeDtypeStruct((num_nonzeros,), np.dtype(np.float32))
        )
    if return_permutation:
        outputs.append(
            jax.ShapeDtypeStruct((num_nonzeros,), np.dtype(np.int32))
        )
    suffix = "weighted" if weighted else "unweighted"
    if return_permutation:
        suffix += "_with_permutation"
    return jffi.ffi_call(
        f"springmv_csr_transpose_{suffix}",
        tuple(outputs),
    )


def csr_transpose(
    row_offsets: Any,
    column_indices: Any,
    *,
    num_cols: int,
    values: Any | None = None,
    return_permutation: bool = False,
) -> tuple[jax.Array, ...]:
    """Transpose a CUDA CSR matrix outside ``jax.jit``.

    The unweighted form returns ``(row_offsets, column_indices)``. Supplying
    ``values`` adds transposed values to the result. When
    ``return_permutation`` is true, the final result is the permutation that
    maps each transposed entry to its input nonzero position.
    """
    if isinstance(num_cols, bool) or not isinstance(num_cols, Integral):
        raise TypeError("num_cols must be an integer")
    num_cols = int(num_cols)
    if num_cols < 0 or num_cols > np.iinfo(np.int32).max:
        raise ValueError("num_cols must be a nonnegative int32 value")
    if not isinstance(return_permutation, bool):
        raise TypeError("return_permutation must be a boolean")

    row_offsets_array = as_structure_array(row_offsets, "row_offsets")
    column_indices_array = as_structure_array(
        column_indices, "column_indices"
    )
    if row_offsets_array.shape[0] == 0:
        raise ValueError("row_offsets must contain at least one entry")

    values_array = None
    inputs = [row_offsets_array, column_indices_array]
    if values is not None:
        values_array = as_float_vector(values, "values")
        if values_array.shape != column_indices_array.shape:
            raise ValueError("values and column_indices must have equal shapes")
        inputs.append(values_array)
    require_eager_cuda(*inputs, operation="csr_transpose")

    outputs = _transpose_call(
        num_cols,
        column_indices_array.shape[0],
        values_array is not None,
        return_permutation,
    )(*inputs, num_cols=np.int32(num_cols))
    return tuple(outputs)
