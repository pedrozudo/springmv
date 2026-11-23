"""Atomic-free probabilistic sampling for weighted CSR log-sum-exp rows."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi as jffi
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from . import _ffi as _registrations  # noqa: F401
from ._ops import _validate_subwarp_size
from ._structure import CsrStructure
from ._validation import as_bool_vector, as_float_vector


@functools.cache
def _sample_call(num_children: int):
    return jffi.ffi_call(
        "springmv_csr_logsumexp_sample",
        jax.ShapeDtypeStruct((num_children,), np.dtype(np.bool_)),
        vmap_method="sequential",
    )


@functools.cache
def _batched_sample_call(batch_size: int, num_children: int):
    return jffi.ffi_call(
        "springmv_csr_logsumexp_sample_batched",
        jax.ShapeDtypeStruct(
            (batch_size, num_children), np.dtype(np.bool_)
        ),
    )


def _sample_impl(
    log_weights,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    gumbels,
    active_parent,
    *,
    num_rows,
    num_cols,
    num_nonzeros,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
):
    del num_rows, num_nonzeros
    return _sample_call(num_cols)(
        log_weights,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        gumbels,
        active_parent,
        max_nnz_per_row=np.int32(max_nnz_per_row),
        transpose_max_nnz_per_row=np.int32(
            transpose_max_nnz_per_row
        ),
        subwarp_size=np.int32(subwarp_size),
    )


def _sample_abstract(
    log_weights,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    gumbels,
    active_parent,
    *,
    num_rows,
    num_cols,
    num_nonzeros,
    **params,
):
    del num_nonzeros
    del params
    assert row_offsets.shape == (num_rows + 1,)
    assert transpose_row_offsets.shape == (num_cols + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert log_weights.shape == column_indices.shape
    assert gumbels.shape == column_indices.shape
    assert active_parent.shape == (num_rows,)
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == np.dtype(np.int32)
    assert transpose_column_indices.dtype == np.dtype(np.int32)
    assert log_weights.dtype == gumbels.dtype == np.dtype(np.float32)
    assert active_parent.dtype == np.dtype(np.bool_)
    return jax.core.ShapedArray((num_cols,), np.dtype(np.bool_))


def _batched_sample_impl(
    log_weights,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    gumbels,
    active_parent,
    *,
    batch_size,
    weights_batch_stride,
    gumbels_batch_stride,
    active_parent_batch_stride,
    num_rows,
    num_cols,
    num_nonzeros,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
):
    del num_rows, num_nonzeros
    return _batched_sample_call(batch_size, num_cols)(
        log_weights,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        gumbels,
        active_parent,
        batch_size=np.int32(batch_size),
        weights_batch_stride=np.int32(weights_batch_stride),
        gumbels_batch_stride=np.int32(gumbels_batch_stride),
        active_parent_batch_stride=np.int32(active_parent_batch_stride),
        max_nnz_per_row=np.int32(max_nnz_per_row),
        transpose_max_nnz_per_row=np.int32(
            transpose_max_nnz_per_row
        ),
        subwarp_size=np.int32(subwarp_size),
    )


def _batched_sample_abstract(
    log_weights,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    gumbels,
    active_parent,
    *,
    batch_size,
    weights_batch_stride,
    gumbels_batch_stride,
    active_parent_batch_stride,
    num_rows,
    num_cols,
    **params,
):
    del params
    nonzeros = column_indices.shape[0]
    expected_weights = (
        (batch_size, nonzeros)
        if weights_batch_stride
        else (nonzeros,)
    )
    expected_gumbels = (
        (batch_size, nonzeros)
        if gumbels_batch_stride
        else (nonzeros,)
    )
    expected_active = (
        (batch_size, num_rows)
        if active_parent_batch_stride
        else (num_rows,)
    )
    assert log_weights.shape == expected_weights
    assert gumbels.shape == expected_gumbels
    assert active_parent.shape == expected_active
    assert row_offsets.shape == (num_rows + 1,)
    assert transpose_row_offsets.shape == (num_cols + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == np.dtype(np.int32)
    assert transpose_column_indices.dtype == np.dtype(np.int32)
    assert log_weights.dtype == gumbels.dtype == np.dtype(np.float32)
    assert active_parent.dtype == np.dtype(np.bool_)
    return jax.core.ShapedArray(
        (batch_size, num_cols), np.dtype(np.bool_)
    )


_sample_p = core.Primitive("springmv_csr_logsumexp_sample")
_sample_p.def_impl(_sample_impl)
_sample_p.def_abstract_eval(_sample_abstract)
mlir.register_lowering(
    _sample_p,
    mlir.lower_fun(_sample_impl, multiple_results=False),
    platform="cuda",
)

_batched_sample_p = core.Primitive("springmv_csr_logsumexp_sample_batched")
_batched_sample_p.def_impl(_batched_sample_impl)
_batched_sample_p.def_abstract_eval(_batched_sample_abstract)
mlir.register_lowering(
    _batched_sample_p,
    mlir.lower_fun(_batched_sample_impl, multiple_results=False),
    platform="cuda",
)


def _sample_batcher(batched_args, batch_dimensions, **params):
    if any(batch_dimensions[index] is not None for index in range(1, 5)):
        raise NotImplementedError(
            "springmv.csr_logsumexp_sample requires one shared "
            "CsrStructure"
        )
    mapped_indices = [
        index
        for index in (0, 5, 6)
        if batch_dimensions[index] is not None
    ]
    if not mapped_indices:
        return _sample_p.bind(*batched_args, **params), None
    batch_sizes = {
        batched_args[index].shape[batch_dimensions[index]]
        for index in mapped_indices
    }
    if len(batch_sizes) != 1:
        raise ValueError("mapped SpringMV operands must have equal batch sizes")
    batch_size = batch_sizes.pop()
    arguments = list(batched_args)
    for index in mapped_indices:
        arguments[index] = jnp.moveaxis(
            arguments[index], batch_dimensions[index], 0
        )
    weights_batched = 0 in mapped_indices
    gumbels_batched = 5 in mapped_indices
    active_batched = 6 in mapped_indices
    return _batched_sample_p.bind(
        *arguments,
        batch_size=batch_size,
        weights_batch_stride=(
            params["num_nonzeros"] if weights_batched else 0
        ),
        gumbels_batch_stride=(
            params["num_nonzeros"] if gumbels_batched else 0
        ),
        active_parent_batch_stride=(
            params["num_rows"] if active_batched else 0
        ),
        **params,
    ), 0


batching.primitive_batchers[_sample_p] = _sample_batcher


def _nondifferentiable(*args, **kwargs):
    del args, kwargs
    raise ValueError("csr_logsumexp_sample cannot be differentiated")


ad.primitive_jvps[_sample_p] = _nondifferentiable
ad.primitive_jvps[_batched_sample_p] = _nondifferentiable


def csr_logsumexp_sample(
    structure: CsrStructure,
    log_weights: Any,
    gumbels: Any,
    active_parent: Any,
    *,
    subwarp_size: int = 0,
) -> jax.Array:
    """Sample and propagate weighted log-sum-exp CSR choices.

    ``gumbels`` contains caller-generated Gumbel noise, one value per CSR
    edge. Exactly one non-NaN maximum-score edge is selected per nonempty
    parent row; ties choose the lowest original edge index. Active selections
    are propagated to a Boolean child mask through the stored transpose.
    """
    if not isinstance(structure, CsrStructure):
        raise TypeError("structure must be a CsrStructure")
    weights_array = as_float_vector(log_weights, "log_weights")
    gumbels_array = as_float_vector(gumbels, "gumbels")
    active_array = as_bool_vector(active_parent, "active_parent")
    if weights_array.shape != structure.column_indices.shape:
        raise ValueError(
            "log_weights and column_indices must have equal shapes"
        )
    if gumbels_array.shape != structure.column_indices.shape:
        raise ValueError("gumbels and column_indices must have equal shapes")
    if active_array.shape != (structure.num_rows,):
        raise ValueError(
            "active_parent length must equal structure.num_rows"
        )
    subwarp = int(_validate_subwarp_size(subwarp_size))
    return _sample_p.bind(
        weights_array,
        structure.row_offsets,
        structure.column_indices,
        structure.transpose_row_offsets,
        structure.transpose_column_indices,
        gumbels_array,
        active_array,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        num_nonzeros=structure.num_nonzeros,
        max_nnz_per_row=structure.max_nnz_per_row,
        transpose_max_nnz_per_row=structure.transpose_max_nnz_per_row,
        subwarp_size=subwarp,
    )
