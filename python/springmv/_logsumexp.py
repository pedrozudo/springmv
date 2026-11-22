"""First-order differentiable log-sum-exp CSR products."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi as jffi
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ._batching import sequential_primitive_batcher
from ._ops import (
    _raw_batched_reduction,
    _raw_reduction,
    _validate_subwarp_size,
)
from ._structure import CsrStructure
from ._validation import as_float_vector


def _unweighted_impl(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    *,
    num_rows,
    num_cols,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
):
    del transpose_row_offsets, transpose_column_indices, num_cols
    del transpose_max_nnz_per_row
    return _raw_reduction(
        "logsumexp",
        row_offsets,
        column_indices,
        log_x,
        None,
        num_rows=num_rows,
        max_nnz_per_row=max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _weighted_impl(
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    *,
    num_rows,
    num_cols,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
    input_vjp_strategy,
):
    del transpose_row_offsets, transpose_column_indices, transpose_permutation
    del num_cols, transpose_max_nnz_per_row, input_vjp_strategy
    return _raw_reduction(
        "logsumexp",
        row_offsets,
        column_indices,
        log_x,
        log_values,
        num_rows=num_rows,
        max_nnz_per_row=max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _unweighted_abstract(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    *,
    num_rows,
    num_cols,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
):
    del max_nnz_per_row, transpose_max_nnz_per_row, subwarp_size
    assert row_offsets.shape == (num_rows + 1,)
    assert transpose_row_offsets.shape == (num_cols + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert log_x.shape == (num_cols,)
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == transpose_column_indices.dtype
    assert log_x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray((num_rows,), np.dtype(np.float32))


def _weighted_abstract(
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    *,
    num_rows,
    num_cols,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
    input_vjp_strategy,
):
    del max_nnz_per_row, transpose_max_nnz_per_row, subwarp_size
    del input_vjp_strategy
    assert row_offsets.shape == (num_rows + 1,)
    assert transpose_row_offsets.shape == (num_cols + 1,)
    assert log_values.shape == column_indices.shape
    assert log_values.shape == transpose_column_indices.shape
    assert log_values.shape == transpose_permutation.shape
    assert log_x.shape == (num_cols,)
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == transpose_column_indices.dtype
    assert transpose_permutation.dtype == np.dtype(np.int32)
    assert log_values.dtype == log_x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray((num_rows,), np.dtype(np.float32))


_unweighted_logsumexp_p = core.Primitive(
    "springmv_csr_logsumexp_unweighted"
)
_unweighted_logsumexp_p.def_impl(_unweighted_impl)
_unweighted_logsumexp_p.def_abstract_eval(_unweighted_abstract)
mlir.register_lowering(
    _unweighted_logsumexp_p,
    mlir.lower_fun(_unweighted_impl, multiple_results=False),
    platform="cuda",
)


def _unweighted_batched_impl(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    *,
    batch_size,
    x_batch_stride,
    **params,
):
    del transpose_row_offsets, transpose_column_indices
    return _raw_batched_reduction(
        "logsumexp",
        row_offsets,
        column_indices,
        log_x,
        None,
        batch_size=batch_size,
        num_rows=params["num_rows"],
        max_nnz_per_row=params["max_nnz_per_row"],
        subwarp_size=params["subwarp_size"],
        x_batch_stride=x_batch_stride,
    )


def _weighted_batched_impl(
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    *,
    batch_size,
    values_batch_stride,
    x_batch_stride,
    values_batched,
    x_batched,
    **params,
):
    del transpose_row_offsets, transpose_column_indices, transpose_permutation
    del values_batched, x_batched
    return _raw_batched_reduction(
        "logsumexp",
        row_offsets,
        column_indices,
        log_x,
        log_values,
        batch_size=batch_size,
        num_rows=params["num_rows"],
        max_nnz_per_row=params["max_nnz_per_row"],
        subwarp_size=params["subwarp_size"],
        x_batch_stride=x_batch_stride,
        values_batch_stride=values_batch_stride,
    )


def _unweighted_batched_abstract(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    *,
    batch_size,
    x_batch_stride,
    **params,
):
    del x_batch_stride
    assert log_x.shape == (batch_size, params["num_cols"])
    assert row_offsets.shape == (params["num_rows"] + 1,)
    assert transpose_row_offsets.shape == (params["num_cols"] + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert log_x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray(
        (batch_size, params["num_rows"]), np.dtype(np.float32)
    )


def _weighted_batched_abstract(
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    *,
    batch_size,
    values_batch_stride,
    x_batch_stride,
    values_batched,
    x_batched,
    **params,
):
    del values_batch_stride, x_batch_stride
    expected_values = (
        (batch_size, column_indices.shape[0])
        if values_batched
        else column_indices.shape
    )
    expected_x = (
        (batch_size, params["num_cols"])
        if x_batched
        else (params["num_cols"],)
    )
    assert log_values.shape == expected_values
    assert log_x.shape == expected_x
    assert log_values.dtype == log_x.dtype == np.dtype(np.float32)
    assert row_offsets.shape == (params["num_rows"] + 1,)
    assert transpose_row_offsets.shape == (params["num_cols"] + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert column_indices.shape == transpose_permutation.shape
    return jax.core.ShapedArray(
        (batch_size, params["num_rows"]), np.dtype(np.float32)
    )


_unweighted_batched_logsumexp_p = core.Primitive(
    "springmv_csr_logsumexp_unweighted_batched"
)
_unweighted_batched_logsumexp_p.def_impl(_unweighted_batched_impl)
_unweighted_batched_logsumexp_p.def_abstract_eval(
    _unweighted_batched_abstract
)
mlir.register_lowering(
    _unweighted_batched_logsumexp_p,
    mlir.lower_fun(_unweighted_batched_impl, multiple_results=False),
    platform="cuda",
)

_weighted_batched_logsumexp_p = core.Primitive(
    "springmv_csr_logsumexp_weighted_batched"
)
_weighted_batched_logsumexp_p.def_impl(_weighted_batched_impl)
_weighted_batched_logsumexp_p.def_abstract_eval(_weighted_batched_abstract)
mlir.register_lowering(
    _weighted_batched_logsumexp_p,
    mlir.lower_fun(_weighted_batched_impl, multiple_results=False),
    platform="cuda",
)

_weighted_logsumexp_p = core.Primitive("springmv_csr_logsumexp_weighted")
_weighted_logsumexp_p.def_impl(_weighted_impl)
_weighted_logsumexp_p.def_abstract_eval(_weighted_abstract)
mlir.register_lowering(
    _weighted_logsumexp_p,
    mlir.lower_fun(_weighted_impl, multiple_results=False),
    platform="cuda",
)


def _unweighted_batcher(batched_args, batch_dimensions, **params):
    if any(dimension is not None for dimension in batch_dimensions[:4]):
        raise NotImplementedError(
            "springmv.csr_logsumexp requires one shared CsrStructure; "
            "batching CSR topology arrays is not supported"
        )
    log_x = jnp.moveaxis(batched_args[4], batch_dimensions[4], 0)
    batch_size = log_x.shape[0]
    return _unweighted_batched_logsumexp_p.bind(
        *batched_args[:4],
        log_x,
        batch_size=batch_size,
        x_batch_stride=params["num_cols"],
        **params,
    ), 0


def _weighted_batcher(batched_args, batch_dimensions, **params):
    if any(
        batch_dimensions[index] is not None for index in range(1, 6)
    ):
        raise NotImplementedError(
            "springmv.csr_logsumexp requires one shared CsrStructure; "
            "batching CSR topology arrays is not supported"
        )
    values_batched = batch_dimensions[0] is not None
    x_batched = batch_dimensions[6] is not None
    mapped_sizes = [
        argument.shape[dimension]
        for argument, dimension in zip(
            batched_args, batch_dimensions, strict=True
        )
        if dimension is not None
    ]
    if not mapped_sizes or len(set(mapped_sizes)) != 1:
        raise ValueError("mapped SpringMV operands must have equal batch sizes")
    batch_size = mapped_sizes[0]
    log_values = (
        jnp.moveaxis(batched_args[0], batch_dimensions[0], 0)
        if values_batched
        else batched_args[0]
    )
    log_x = (
        jnp.moveaxis(batched_args[6], batch_dimensions[6], 0)
        if x_batched
        else batched_args[6]
    )
    return _weighted_batched_logsumexp_p.bind(
        log_values,
        *batched_args[1:6],
        log_x,
        batch_size=batch_size,
        values_batch_stride=log_values.shape[-1] if values_batched else 0,
        x_batch_stride=params["num_cols"] if x_batched else 0,
        values_batched=values_batched,
        x_batched=x_batched,
        **params,
    ), 0


batching.primitive_batchers[_unweighted_logsumexp_p] = _unweighted_batcher
batching.primitive_batchers[_weighted_logsumexp_p] = _weighted_batcher


def _unweighted_bind(structure: CsrStructure, log_x, subwarp_size: int):
    return _unweighted_logsumexp_p.bind(
        structure.row_offsets,
        structure.column_indices,
        structure.transpose_row_offsets,
        structure.transpose_column_indices,
        log_x,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        max_nnz_per_row=structure.max_nnz_per_row,
        transpose_max_nnz_per_row=structure.transpose_max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _weighted_bind(
    structure: CsrStructure,
    log_values,
    log_x,
    subwarp_size: int,
    input_vjp_strategy: str,
):
    return _weighted_logsumexp_p.bind(
        log_values,
        structure.row_offsets,
        structure.column_indices,
        structure.transpose_row_offsets,
        structure.transpose_column_indices,
        structure.transpose_permutation,
        log_x,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        max_nnz_per_row=structure.max_nnz_per_row,
        transpose_max_nnz_per_row=structure.transpose_max_nnz_per_row,
        subwarp_size=subwarp_size,
        input_vjp_strategy=input_vjp_strategy,
    )


@functools.cache
def _jvp_call(weighted: bool, num_rows: int):
    suffix = "weighted" if weighted else "unweighted"
    return jffi.ffi_call(
        f"springmv_csr_logsumexp_jvp_{suffix}",
        jax.ShapeDtypeStruct((num_rows,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


def _unweighted_jvp_impl(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    output,
    input_tangent,
    *,
    num_rows,
    num_cols,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
):
    del transpose_row_offsets, transpose_column_indices, num_cols
    del transpose_max_nnz_per_row
    return _jvp_call(False, num_rows)(
        row_offsets,
        column_indices,
        log_x,
        output,
        input_tangent,
        max_nnz_per_row=np.int32(max_nnz_per_row),
        subwarp_size=np.int32(subwarp_size),
    )


def _weighted_jvp_impl(
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    output,
    values_tangent,
    input_tangent,
    *,
    num_rows,
    num_cols,
    max_nnz_per_row,
    transpose_max_nnz_per_row,
    subwarp_size,
    input_vjp_strategy,
):
    del transpose_row_offsets, transpose_column_indices, transpose_permutation
    del num_cols, transpose_max_nnz_per_row, input_vjp_strategy
    return _jvp_call(True, num_rows)(
        row_offsets,
        column_indices,
        log_values,
        log_x,
        output,
        values_tangent,
        input_tangent,
        max_nnz_per_row=np.int32(max_nnz_per_row),
        subwarp_size=np.int32(subwarp_size),
    )


def _unweighted_jvp_abstract(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    output,
    input_tangent,
    **params,
):
    result = _unweighted_abstract(
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        log_x,
        **params,
    )
    assert output.shape == result.shape
    assert input_tangent.shape == log_x.shape
    assert output.dtype == input_tangent.dtype == log_x.dtype
    return result


def _weighted_jvp_abstract(
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    output,
    values_tangent,
    input_tangent,
    **params,
):
    result = _weighted_abstract(
        log_values,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        log_x,
        **params,
    )
    assert output.shape == result.shape
    assert values_tangent.shape == log_values.shape
    assert input_tangent.shape == log_x.shape
    assert output.dtype == values_tangent.dtype == input_tangent.dtype
    assert output.dtype == log_x.dtype
    return result


_unweighted_jvp_p = core.Primitive(
    "springmv_csr_logsumexp_unweighted_jvp"
)
_unweighted_jvp_p.def_impl(_unweighted_jvp_impl)
_unweighted_jvp_p.def_abstract_eval(_unweighted_jvp_abstract)
mlir.register_lowering(
    _unweighted_jvp_p,
    mlir.lower_fun(_unweighted_jvp_impl, multiple_results=False),
    platform="cuda",
)

_weighted_jvp_p = core.Primitive("springmv_csr_logsumexp_weighted_jvp")
_weighted_jvp_p.def_impl(_weighted_jvp_impl)
_weighted_jvp_p.def_abstract_eval(_weighted_jvp_abstract)
mlir.register_lowering(
    _weighted_jvp_p,
    mlir.lower_fun(_weighted_jvp_impl, multiple_results=False),
    platform="cuda",
)


def _unweighted_jvp_batcher(batched_args, batch_dimensions, **params):
    return sequential_primitive_batcher(
        _unweighted_jvp_p,
        batched_args,
        batch_dimensions,
        batchable_indices=(4, 5, 6),
        operation_name="springmv.csr_logsumexp JVP",
        **params,
    )


def _weighted_jvp_batcher(batched_args, batch_dimensions, **params):
    return sequential_primitive_batcher(
        _weighted_jvp_p,
        batched_args,
        batch_dimensions,
        batchable_indices=(0, 6, 7, 8, 9),
        operation_name="springmv.csr_logsumexp JVP",
        **params,
    )


batching.primitive_batchers[_unweighted_jvp_p] = _unweighted_jvp_batcher
batching.primitive_batchers[_weighted_jvp_p] = _weighted_jvp_batcher


def _unweighted_jvp_rule(primals, tangents, **params):
    (
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        log_x,
    ) = primals
    input_tangent = ad.instantiate_zeros(tangents[-1])
    output = _unweighted_logsumexp_p.bind(*primals, **params)
    output_tangent = _unweighted_jvp_p.bind(
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        log_x,
        output,
        input_tangent,
        **params,
    )
    return output, output_tangent


def _weighted_jvp_rule(primals, tangents, **params):
    (
        log_values,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        log_x,
    ) = primals
    values_tangent = ad.instantiate_zeros(tangents[0])
    input_tangent = ad.instantiate_zeros(tangents[-1])
    output = _weighted_logsumexp_p.bind(*primals, **params)
    output_tangent = _weighted_jvp_p.bind(
        log_values,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        log_x,
        output,
        values_tangent,
        input_tangent,
        **params,
    )
    return output, output_tangent


ad.primitive_jvps[_unweighted_logsumexp_p] = _unweighted_jvp_rule
ad.primitive_jvps[_weighted_logsumexp_p] = _weighted_jvp_rule


def _unweighted_batched_jvp_rule(primals, tangents, **params):
    (
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        log_x,
    ) = primals
    input_tangent = ad.instantiate_zeros(tangents[-1])
    output = _unweighted_batched_logsumexp_p.bind(*primals, **params)
    single_params = dict(params)
    single_params.pop("batch_size")
    single_params.pop("x_batch_stride")

    def tangent_one(index):
        return _unweighted_jvp_p.bind(
            row_offsets,
            column_indices,
            transpose_row_offsets,
            transpose_column_indices,
            log_x[index],
            output[index],
            input_tangent[index],
            **single_params,
        )

    output_tangent = jax.lax.map(
        tangent_one, jnp.arange(params["batch_size"], dtype=jnp.int32)
    )
    return output, output_tangent


def _weighted_batched_jvp_rule(primals, tangents, **params):
    (
        log_values,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        log_x,
    ) = primals
    values_tangent = ad.instantiate_zeros(tangents[0])
    input_tangent = ad.instantiate_zeros(tangents[-1])
    output = _weighted_batched_logsumexp_p.bind(*primals, **params)
    single_params = dict(params)
    for name in (
        "batch_size",
        "values_batch_stride",
        "x_batch_stride",
        "values_batched",
        "x_batched",
    ):
        single_params.pop(name)

    def tangent_one(index):
        current_values = (
            log_values[index] if params["values_batched"] else log_values
        )
        current_values_tangent = (
            values_tangent[index]
            if params["values_batched"]
            else values_tangent
        )
        current_x = log_x[index] if params["x_batched"] else log_x
        current_input_tangent = (
            input_tangent[index]
            if params["x_batched"]
            else input_tangent
        )
        return _weighted_jvp_p.bind(
            current_values,
            row_offsets,
            column_indices,
            transpose_row_offsets,
            transpose_column_indices,
            transpose_permutation,
            current_x,
            output[index],
            current_values_tangent,
            current_input_tangent,
            **single_params,
        )

    output_tangent = jax.lax.map(
        tangent_one, jnp.arange(params["batch_size"], dtype=jnp.int32)
    )
    return output, output_tangent


ad.primitive_jvps[
    _unweighted_batched_logsumexp_p
] = _unweighted_batched_jvp_rule
ad.primitive_jvps[
    _weighted_batched_logsumexp_p
] = _weighted_batched_jvp_rule


@functools.cache
def _input_vjp_call(weighted: bool, num_cols: int):
    suffix = "weighted" if weighted else "unweighted"
    return jffi.ffi_call(
        f"springmv_csr_logsumexp_input_vjp_{suffix}",
        jax.ShapeDtypeStruct((num_cols,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


@functools.cache
def _input_vjp_scatter_call(num_cols: int):
    return jffi.ffi_call(
        "springmv_csr_logsumexp_input_vjp_scatter",
        jax.ShapeDtypeStruct((num_cols,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


@functools.cache
def _values_vjp_call(num_nonzeros: int):
    return jffi.ffi_call(
        "springmv_csr_logsumexp_values_vjp",
        jax.ShapeDtypeStruct((num_nonzeros,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


def _unweighted_jvp_transpose(
    cotangent,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    log_x,
    output,
    input_tangent,
    **params,
):
    assert not ad.is_undefined_primal(row_offsets)
    assert not ad.is_undefined_primal(column_indices)
    assert not ad.is_undefined_primal(transpose_row_offsets)
    assert not ad.is_undefined_primal(transpose_column_indices)
    assert not ad.is_undefined_primal(log_x)
    assert not ad.is_undefined_primal(output)
    if not ad.is_undefined_primal(input_tangent):
        raise AssertionError("log-sum-exp transpose expected an unknown tangent")
    cotangent = ad.instantiate_zeros(cotangent)
    input_cotangent = _input_vjp_call(False, params["num_cols"])(
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        log_x,
        output,
        cotangent,
        max_nnz_per_row=np.int32(params["transpose_max_nnz_per_row"]),
        subwarp_size=np.int32(params["subwarp_size"]),
    )
    return (None, None, None, None, None, None, input_cotangent)


def _weighted_jvp_transpose(
    cotangent,
    log_values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    log_x,
    output,
    values_tangent,
    input_tangent,
    **params,
):
    assert not ad.is_undefined_primal(log_values)
    assert not ad.is_undefined_primal(row_offsets)
    assert not ad.is_undefined_primal(column_indices)
    assert not ad.is_undefined_primal(transpose_row_offsets)
    assert not ad.is_undefined_primal(transpose_column_indices)
    assert not ad.is_undefined_primal(transpose_permutation)
    assert not ad.is_undefined_primal(log_x)
    assert not ad.is_undefined_primal(output)
    cotangent = ad.instantiate_zeros(cotangent)

    values_cotangent = None
    if ad.is_undefined_primal(values_tangent):
        values_cotangent = _values_vjp_call(column_indices.shape[0])(
            row_offsets,
            column_indices,
            log_values,
            log_x,
            output,
            cotangent,
            max_nnz_per_row=np.int32(params["max_nnz_per_row"]),
            subwarp_size=np.int32(params["subwarp_size"]),
        )

    input_cotangent = None
    if ad.is_undefined_primal(input_tangent):
        if params["input_vjp_strategy"] == "scatter":
            input_cotangent = _input_vjp_scatter_call(params["num_cols"])(
                row_offsets,
                column_indices,
                log_values,
                log_x,
                output,
                cotangent,
                max_nnz_per_row=np.int32(params["max_nnz_per_row"]),
                subwarp_size=np.int32(params["subwarp_size"]),
            )
        else:
            input_cotangent = _input_vjp_call(True, params["num_cols"])(
                row_offsets,
                column_indices,
                transpose_row_offsets,
                transpose_column_indices,
                transpose_permutation,
                log_values,
                log_x,
                output,
                cotangent,
                max_nnz_per_row=np.int32(
                    params["transpose_max_nnz_per_row"]
                ),
                subwarp_size=np.int32(params["subwarp_size"]),
            )

    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        values_cotangent,
        input_cotangent,
    )


ad.primitive_transposes[_unweighted_jvp_p] = _unweighted_jvp_transpose
ad.primitive_transposes[_weighted_jvp_p] = _weighted_jvp_transpose


def csr_logsumexp(
    structure: CsrStructure,
    log_x: Any,
    *,
    log_values: Any | None = None,
    subwarp_size: int = 0,
    input_vjp_strategy: str = "scatter",
) -> jax.Array:
    """Compute a first-order differentiable log-sum-exp CSR product.

    ``input_vjp_strategy`` selects the weighted input-gradient implementation.
    Unweighted input gradients always use transposed CSR.
    """
    if not isinstance(structure, CsrStructure):
        raise TypeError("structure must be a CsrStructure")
    log_x_array = as_float_vector(log_x, "log_x")
    if log_x_array.shape[0] != structure.num_cols:
        raise ValueError("log_x length must equal structure.num_cols")
    subwarp = int(_validate_subwarp_size(subwarp_size))
    if input_vjp_strategy not in ("transpose", "scatter"):
        raise ValueError(
            "input_vjp_strategy must be 'transpose' or 'scatter'"
        )

    if log_values is None:
        return _unweighted_bind(structure, log_x_array, subwarp)
    log_values_array = as_float_vector(log_values, "log_values")
    if log_values_array.shape != structure.column_indices.shape:
        raise ValueError(
            "log_values and column_indices must have equal shapes"
        )
    return _weighted_bind(
        structure,
        log_values_array,
        log_x_array,
        subwarp,
        input_vjp_strategy,
    )
