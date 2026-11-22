"""JAX-facing sparse reduction operations."""

from __future__ import annotations

import functools
from collections.abc import Callable
from numbers import Integral
from typing import Any

import jax
import numpy as np
from jax import ffi as jffi
from jax.extend import core
from jax.interpreters import ad, batching, mlir

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
        vmap_method="sequential",
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


@functools.cache
def _batched_reduction_call(
    target: str, batch_size: int, num_rows: int
) -> Callable[..., jax.Array]:
    return jffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(
            (batch_size, num_rows), np.dtype(np.float32)
        ),
    )


def _raw_batched_reduction(
    operation: str,
    row_offsets: jax.Array,
    column_indices: jax.Array,
    x: jax.Array,
    edge_values: jax.Array | None,
    *,
    batch_size: int,
    num_rows: int,
    max_nnz_per_row: int,
    subwarp_size: int,
    x_batch_stride: int,
    values_batch_stride: int = 0,
) -> jax.Array:
    if batch_size < 0 or batch_size * num_rows > np.iinfo(np.int32).max:
        raise ValueError("batch_size times num_rows must fit in int32")
    suffix = "weighted" if edge_values is not None else "unweighted"
    call = _batched_reduction_call(
        f"springmv_csr_{operation}_batched_{suffix}", batch_size, num_rows
    )
    attributes = dict(
        batch_size=np.int32(batch_size),
        x_batch_stride=np.int32(x_batch_stride),
        max_nnz_per_row=np.int32(max_nnz_per_row),
        subwarp_size=np.int32(subwarp_size),
    )
    if edge_values is None:
        return call(row_offsets, column_indices, x, **attributes)
    return call(
        row_offsets,
        column_indices,
        edge_values,
        x,
        values_batch_stride=np.int32(values_batch_stride),
        **attributes,
    )


def _unweighted_reduction_impl(
    row_offsets,
    column_indices,
    x,
    *,
    operation,
    num_rows,
    num_cols,
    max_nnz_per_row,
    subwarp_size,
):
    del num_cols
    return _raw_reduction(
        operation,
        row_offsets,
        column_indices,
        x,
        None,
        num_rows=num_rows,
        max_nnz_per_row=max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _weighted_reduction_impl(
    edge_values,
    row_offsets,
    column_indices,
    x,
    *,
    operation,
    num_rows,
    num_cols,
    max_nnz_per_row,
    subwarp_size,
):
    del num_cols
    return _raw_reduction(
        operation,
        row_offsets,
        column_indices,
        x,
        edge_values,
        num_rows=num_rows,
        max_nnz_per_row=max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _unweighted_reduction_abstract(
    row_offsets,
    column_indices,
    x,
    *,
    operation,
    num_rows,
    num_cols,
    max_nnz_per_row,
    subwarp_size,
):
    del operation, max_nnz_per_row, subwarp_size
    assert row_offsets.shape == (num_rows + 1,)
    assert x.shape == (num_cols,)
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray((num_rows,), np.dtype(np.float32))


def _weighted_reduction_abstract(
    edge_values,
    row_offsets,
    column_indices,
    x,
    **params,
):
    result = _unweighted_reduction_abstract(
        row_offsets, column_indices, x, **params
    )
    assert edge_values.shape == column_indices.shape
    assert edge_values.dtype == np.dtype(np.float32)
    return result


_unweighted_reduction_p = core.Primitive("springmv_csr_reduction_unweighted")
_unweighted_reduction_p.def_impl(_unweighted_reduction_impl)
_unweighted_reduction_p.def_abstract_eval(_unweighted_reduction_abstract)
mlir.register_lowering(
    _unweighted_reduction_p,
    mlir.lower_fun(_unweighted_reduction_impl, multiple_results=False),
    platform="cuda",
)

_weighted_reduction_p = core.Primitive("springmv_csr_reduction_weighted")
_weighted_reduction_p.def_impl(_weighted_reduction_impl)
_weighted_reduction_p.def_abstract_eval(_weighted_reduction_abstract)
mlir.register_lowering(
    _weighted_reduction_p,
    mlir.lower_fun(_weighted_reduction_impl, multiple_results=False),
    platform="cuda",
)


def _unweighted_reduction_batcher(
    batched_args, batch_dimensions, **params
):
    if batch_dimensions[0] is not None or batch_dimensions[1] is not None:
        raise NotImplementedError(
            "SpringMV reductions require one shared CsrStructure"
        )
    x = jax.numpy.moveaxis(batched_args[2], batch_dimensions[2], 0)
    return _raw_batched_reduction(
        params["operation"],
        batched_args[0],
        batched_args[1],
        x,
        None,
        batch_size=x.shape[0],
        num_rows=params["num_rows"],
        max_nnz_per_row=params["max_nnz_per_row"],
        subwarp_size=params["subwarp_size"],
        x_batch_stride=params["num_cols"],
    ), 0


def _weighted_reduction_batcher(batched_args, batch_dimensions, **params):
    if batch_dimensions[1] is not None or batch_dimensions[2] is not None:
        raise NotImplementedError(
            "SpringMV reductions require one shared CsrStructure"
        )
    values_batched = batch_dimensions[0] is not None
    x_batched = batch_dimensions[3] is not None
    mapped_sizes = [
        argument.shape[dimension]
        for argument, dimension in zip(
            batched_args, batch_dimensions, strict=True
        )
        if dimension is not None
    ]
    if not mapped_sizes or len(set(mapped_sizes)) != 1:
        raise ValueError("mapped SpringMV operands must have equal batch sizes")
    edge_values = (
        jax.numpy.moveaxis(batched_args[0], batch_dimensions[0], 0)
        if values_batched
        else batched_args[0]
    )
    x = (
        jax.numpy.moveaxis(batched_args[3], batch_dimensions[3], 0)
        if x_batched
        else batched_args[3]
    )
    return _raw_batched_reduction(
        params["operation"],
        batched_args[1],
        batched_args[2],
        x,
        edge_values,
        batch_size=mapped_sizes[0],
        num_rows=params["num_rows"],
        max_nnz_per_row=params["max_nnz_per_row"],
        subwarp_size=params["subwarp_size"],
        x_batch_stride=params["num_cols"] if x_batched else 0,
        values_batch_stride=edge_values.shape[-1] if values_batched else 0,
    ), 0


batching.primitive_batchers[
    _unweighted_reduction_p
] = _unweighted_reduction_batcher
batching.primitive_batchers[
    _weighted_reduction_p
] = _weighted_reduction_batcher


def _nondifferentiable_reduction(*args, **kwargs):
    del args, kwargs
    raise ValueError("min-plus and max-plus cannot be differentiated")


ad.primitive_jvps[_unweighted_reduction_p] = _nondifferentiable_reduction
ad.primitive_jvps[_weighted_reduction_p] = _nondifferentiable_reduction


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
        return _unweighted_reduction_p.bind(
            structure.row_offsets,
            structure.column_indices,
            x_array,
            operation=operation,
            num_rows=structure.num_rows,
            num_cols=structure.num_cols,
            max_nnz_per_row=structure.max_nnz_per_row,
            subwarp_size=subwarp,
        )

    edge_values_array = as_float_vector(edge_values, edge_values_name)
    if edge_values_array.shape != structure.column_indices.shape:
        raise ValueError(
            f"{edge_values_name} and column_indices must have equal shapes"
        )
    return _weighted_reduction_p.bind(
        edge_values_array,
        structure.row_offsets,
        structure.column_indices,
        x_array,
        operation=operation,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
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
