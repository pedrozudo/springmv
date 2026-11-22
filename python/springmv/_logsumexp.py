"""First-order differentiable log-sum-exp CSR products."""

from __future__ import annotations

import functools
from typing import Any

import jax
import numpy as np
from jax import ffi as jffi
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ._ops import _raw_reduction, _validate_subwarp_size
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

_weighted_logsumexp_p = core.Primitive("springmv_csr_logsumexp_weighted")
_weighted_logsumexp_p.def_impl(_weighted_impl)
_weighted_logsumexp_p.def_abstract_eval(_weighted_abstract)
mlir.register_lowering(
    _weighted_logsumexp_p,
    mlir.lower_fun(_weighted_impl, multiple_results=False),
    platform="cuda",
)


def _vmap_not_supported(batched_args, batch_dimensions, **params):
    del batched_args, batch_dimensions, params
    raise NotImplementedError(
        "springmv.csr_logsumexp does not support jax.vmap"
    )


batching.primitive_batchers[_unweighted_logsumexp_p] = _vmap_not_supported
batching.primitive_batchers[_weighted_logsumexp_p] = _vmap_not_supported


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


@functools.cache
def _input_vjp_call(weighted: bool, num_cols: int):
    suffix = "weighted" if weighted else "unweighted"
    return jffi.ffi_call(
        f"springmv_csr_logsumexp_input_vjp_{suffix}",
        jax.ShapeDtypeStruct((num_cols,), np.dtype(np.float32)),
    )


@functools.cache
def _input_vjp_scatter_call(num_cols: int):
    return jffi.ffi_call(
        "springmv_csr_logsumexp_input_vjp_scatter",
        jax.ShapeDtypeStruct((num_cols,), np.dtype(np.float32)),
    )


@functools.cache
def _values_vjp_call(num_nonzeros: int):
    return jffi.ffi_call(
        "springmv_csr_logsumexp_values_vjp",
        jax.ShapeDtypeStruct((num_nonzeros,), np.dtype(np.float32)),
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
    input_vjp_strategy: str = "transpose",
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
