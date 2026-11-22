"""Differentiable arithmetic CSR matrix-vector products."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi as jffi
from jax.extend import core
from jax.interpreters import ad, batching, mlir

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
    x,
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
        "spmv",
        row_offsets,
        column_indices,
        x,
        None,
        num_rows=num_rows,
        max_nnz_per_row=max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _weighted_impl(
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
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
        "spmv",
        row_offsets,
        column_indices,
        x,
        values,
        num_rows=num_rows,
        max_nnz_per_row=max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _unweighted_abstract(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    x,
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
    assert x.shape == (num_cols,)
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == transpose_column_indices.dtype
    assert x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray((num_rows,), np.dtype(np.float32))


def _weighted_abstract(
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
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
    assert values.shape == column_indices.shape
    assert values.shape == transpose_column_indices.shape
    assert values.shape == transpose_permutation.shape
    assert x.shape == (num_cols,)
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == transpose_column_indices.dtype
    assert transpose_permutation.dtype == np.dtype(np.int32)
    assert values.dtype == x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray((num_rows,), np.dtype(np.float32))


_unweighted_spmv_p = core.Primitive("springmv_csr_spmv_unweighted")
_unweighted_spmv_p.def_impl(_unweighted_impl)
_unweighted_spmv_p.def_abstract_eval(_unweighted_abstract)
mlir.register_lowering(
    _unweighted_spmv_p,
    mlir.lower_fun(_unweighted_impl, multiple_results=False),
    platform="cuda",
)

_weighted_spmv_p = core.Primitive("springmv_csr_spmv_weighted")
_weighted_spmv_p.def_impl(_weighted_impl)
_weighted_spmv_p.def_abstract_eval(_weighted_abstract)
mlir.register_lowering(
    _weighted_spmv_p,
    mlir.lower_fun(_weighted_impl, multiple_results=False),
    platform="cuda",
)


def _unweighted_batched_impl(
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    x,
    *,
    batch_size,
    x_batch_stride,
    **params,
):
    del transpose_row_offsets, transpose_column_indices
    return _raw_batched_reduction(
        "spmv",
        row_offsets,
        column_indices,
        x,
        None,
        batch_size=batch_size,
        num_rows=params["num_rows"],
        max_nnz_per_row=params["max_nnz_per_row"],
        subwarp_size=params["subwarp_size"],
        x_batch_stride=x_batch_stride,
    )


def _weighted_batched_impl(
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
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
        "spmv",
        row_offsets,
        column_indices,
        x,
        values,
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
    x,
    *,
    batch_size,
    x_batch_stride,
    **params,
):
    del x_batch_stride
    assert x.shape == (batch_size, params["num_cols"])
    assert row_offsets.shape == (params["num_rows"] + 1,)
    assert transpose_row_offsets.shape == (params["num_cols"] + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert row_offsets.dtype == column_indices.dtype == np.dtype(np.int32)
    assert transpose_row_offsets.dtype == transpose_column_indices.dtype
    assert x.dtype == np.dtype(np.float32)
    return jax.core.ShapedArray(
        (batch_size, params["num_rows"]), np.dtype(np.float32)
    )


def _weighted_batched_abstract(
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
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
    assert values.shape == expected_values
    assert x.shape == expected_x
    assert values.dtype == x.dtype == np.dtype(np.float32)
    assert row_offsets.shape == (params["num_rows"] + 1,)
    assert transpose_row_offsets.shape == (params["num_cols"] + 1,)
    assert column_indices.shape == transpose_column_indices.shape
    assert column_indices.shape == transpose_permutation.shape
    return jax.core.ShapedArray(
        (batch_size, params["num_rows"]), np.dtype(np.float32)
    )


_unweighted_batched_spmv_p = core.Primitive(
    "springmv_csr_spmv_unweighted_batched"
)
_unweighted_batched_spmv_p.def_impl(_unweighted_batched_impl)
_unweighted_batched_spmv_p.def_abstract_eval(_unweighted_batched_abstract)
mlir.register_lowering(
    _unweighted_batched_spmv_p,
    mlir.lower_fun(_unweighted_batched_impl, multiple_results=False),
    platform="cuda",
)

_weighted_batched_spmv_p = core.Primitive(
    "springmv_csr_spmv_weighted_batched"
)
_weighted_batched_spmv_p.def_impl(_weighted_batched_impl)
_weighted_batched_spmv_p.def_abstract_eval(_weighted_batched_abstract)
mlir.register_lowering(
    _weighted_batched_spmv_p,
    mlir.lower_fun(_weighted_batched_impl, multiple_results=False),
    platform="cuda",
)


def _unweighted_batcher(batched_args, batch_dimensions, **params):
    if any(dimension is not None for dimension in batch_dimensions[:4]):
        raise NotImplementedError(
            "springmv.csr_spmv requires one shared CsrStructure; "
            "batching CSR topology arrays is not supported"
        )
    x = jnp.moveaxis(batched_args[4], batch_dimensions[4], 0)
    batch_size = x.shape[0]
    return _unweighted_batched_spmv_p.bind(
        *batched_args[:4],
        x,
        batch_size=batch_size,
        x_batch_stride=params["num_cols"],
        **params,
    ), 0


def _weighted_batcher(batched_args, batch_dimensions, **params):
    if any(
        batch_dimensions[index] is not None for index in range(1, 6)
    ):
        raise NotImplementedError(
            "springmv.csr_spmv requires one shared CsrStructure; "
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
    values = (
        jnp.moveaxis(batched_args[0], batch_dimensions[0], 0)
        if values_batched
        else batched_args[0]
    )
    x = (
        jnp.moveaxis(batched_args[6], batch_dimensions[6], 0)
        if x_batched
        else batched_args[6]
    )
    return _weighted_batched_spmv_p.bind(
        values,
        *batched_args[1:6],
        x,
        batch_size=batch_size,
        values_batch_stride=values.shape[-1] if values_batched else 0,
        x_batch_stride=params["num_cols"] if x_batched else 0,
        values_batched=values_batched,
        x_batched=x_batched,
        **params,
    ), 0


batching.primitive_batchers[_unweighted_spmv_p] = _unweighted_batcher
batching.primitive_batchers[_weighted_spmv_p] = _weighted_batcher


def _unweighted_bind(structure: CsrStructure, x, subwarp_size: int):
    return _unweighted_spmv_p.bind(
        structure.row_offsets,
        structure.column_indices,
        structure.transpose_row_offsets,
        structure.transpose_column_indices,
        x,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        max_nnz_per_row=structure.max_nnz_per_row,
        transpose_max_nnz_per_row=structure.transpose_max_nnz_per_row,
        subwarp_size=subwarp_size,
    )


def _weighted_bind(
    structure: CsrStructure,
    values,
    x,
    subwarp_size: int,
    input_vjp_strategy: str,
):
    return _weighted_spmv_p.bind(
        values,
        structure.row_offsets,
        structure.column_indices,
        structure.transpose_row_offsets,
        structure.transpose_column_indices,
        structure.transpose_permutation,
        x,
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        max_nnz_per_row=structure.max_nnz_per_row,
        transpose_max_nnz_per_row=structure.transpose_max_nnz_per_row,
        subwarp_size=subwarp_size,
        input_vjp_strategy=input_vjp_strategy,
    )


def _unweighted_jvp_x(
    x_dot,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    x,
    **params,
):
    del x
    return _unweighted_spmv_p.bind(
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        x_dot,
        **params,
    )


ad.defjvp(_unweighted_spmv_p, None, None, None, None, _unweighted_jvp_x)


def _unweighted_batched_jvp_x(
    x_dot,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    x,
    **params,
):
    del x
    return _unweighted_batched_spmv_p.bind(
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        x_dot,
        **params,
    )


ad.defjvp(
    _unweighted_batched_spmv_p,
    None,
    None,
    None,
    None,
    _unweighted_batched_jvp_x,
)


def _unweighted_transpose(
    cotangent,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    x,
    **params,
):
    assert not ad.is_undefined_primal(row_offsets)
    assert not ad.is_undefined_primal(column_indices)
    assert not ad.is_undefined_primal(transpose_row_offsets)
    assert not ad.is_undefined_primal(transpose_column_indices)
    if not ad.is_undefined_primal(x):
        raise AssertionError("unweighted SpMV transpose expected an unknown x")
    input_cotangent = _raw_reduction(
        "spmv",
        transpose_row_offsets,
        transpose_column_indices,
        cotangent,
        None,
        num_rows=params["num_cols"],
        max_nnz_per_row=params["transpose_max_nnz_per_row"],
        subwarp_size=params["subwarp_size"],
    )
    return (
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        input_cotangent,
    )


ad.primitive_transposes[_unweighted_spmv_p] = _unweighted_transpose


def _weighted_jvp_values(
    values_dot,
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
    **params,
):
    del values
    return _weighted_spmv_p.bind(
        values_dot,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        x,
        **params,
    )


def _weighted_jvp_x(
    x_dot,
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
    **params,
):
    del x
    return _weighted_spmv_p.bind(
        values,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        x_dot,
        **params,
    )


ad.defjvp(
    _weighted_spmv_p,
    _weighted_jvp_values,
    None,
    None,
    None,
    None,
    None,
    _weighted_jvp_x,
)


def _weighted_batched_jvp_values(
    values_dot,
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
    **params,
):
    del values
    return _weighted_batched_spmv_p.bind(
        values_dot,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        x,
        **params,
    )


def _weighted_batched_jvp_x(
    x_dot,
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
    **params,
):
    del x
    return _weighted_batched_spmv_p.bind(
        values,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        x_dot,
        **params,
    )


ad.defjvp(
    _weighted_batched_spmv_p,
    _weighted_batched_jvp_values,
    None,
    None,
    None,
    None,
    None,
    _weighted_batched_jvp_x,
)


@functools.cache
def _input_vjp_call(num_cols: int):
    return jffi.ffi_call(
        "springmv_csr_spmv_input_vjp",
        jax.ShapeDtypeStruct((num_cols,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


@functools.cache
def _input_vjp_scatter_call(num_cols: int):
    return jffi.ffi_call(
        "springmv_csr_spmv_input_vjp_scatter",
        jax.ShapeDtypeStruct((num_cols,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


@functools.cache
def _values_vjp_call(num_nonzeros: int):
    return jffi.ffi_call(
        "springmv_csr_spmv_values_vjp",
        jax.ShapeDtypeStruct((num_nonzeros,), np.dtype(np.float32)),
        vmap_method="sequential",
    )


def _weighted_transpose(
    cotangent,
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
    **params,
):
    assert not ad.is_undefined_primal(row_offsets)
    assert not ad.is_undefined_primal(column_indices)
    assert not ad.is_undefined_primal(transpose_row_offsets)
    assert not ad.is_undefined_primal(transpose_column_indices)
    assert not ad.is_undefined_primal(transpose_permutation)

    if ad.is_undefined_primal(x):
        if params["input_vjp_strategy"] == "scatter":
            input_cotangent = _input_vjp_scatter_call(params["num_cols"])(
                row_offsets,
                column_indices,
                values,
                cotangent,
                max_nnz_per_row=np.int32(params["max_nnz_per_row"]),
                subwarp_size=np.int32(params["subwarp_size"]),
            )
        else:
            input_cotangent = _input_vjp_call(params["num_cols"])(
                transpose_row_offsets,
                transpose_column_indices,
                transpose_permutation,
                values,
                cotangent,
                max_nnz_per_row=np.int32(
                    params["transpose_max_nnz_per_row"]
                ),
                subwarp_size=np.int32(params["subwarp_size"]),
            )
        return (
            values,
            row_offsets,
            column_indices,
            transpose_row_offsets,
            transpose_column_indices,
            transpose_permutation,
            input_cotangent,
        )

    if not ad.is_undefined_primal(values):
        raise AssertionError(
            "weighted SpMV transpose expected unknown values or x"
        )
    values_cotangent = _values_vjp_call(column_indices.shape[0])(
        row_offsets,
        column_indices,
        x,
        cotangent,
        max_nnz_per_row=np.int32(params["max_nnz_per_row"]),
        subwarp_size=np.int32(params["subwarp_size"]),
    )
    return (
        values_cotangent,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        x,
    )


ad.primitive_transposes[_weighted_spmv_p] = _weighted_transpose


def _unweighted_batched_transpose(
    cotangent,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    x,
    **params,
):
    if not ad.is_undefined_primal(x):
        raise AssertionError("batched unweighted SpMV expected an unknown x")
    cotangent = ad.instantiate_zeros(cotangent)

    def transpose_one(current_cotangent):
        return _raw_reduction(
            "spmv",
            transpose_row_offsets,
            transpose_column_indices,
            current_cotangent,
            None,
            num_rows=params["num_cols"],
            max_nnz_per_row=params["transpose_max_nnz_per_row"],
            subwarp_size=params["subwarp_size"],
        )

    input_cotangent = jax.lax.map(transpose_one, cotangent)
    return (
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        input_cotangent,
    )


def _weighted_batched_transpose(
    cotangent,
    values,
    row_offsets,
    column_indices,
    transpose_row_offsets,
    transpose_column_indices,
    transpose_permutation,
    x,
    **params,
):
    cotangent = ad.instantiate_zeros(cotangent)

    if ad.is_undefined_primal(x):
        if ad.is_undefined_primal(values):
            raise AssertionError("weighted SpMV requires known values for dx")

        def input_vjp_one(current_values, current_cotangent):
            if params["input_vjp_strategy"] == "scatter":
                return _input_vjp_scatter_call(params["num_cols"])(
                    row_offsets,
                    column_indices,
                    current_values,
                    current_cotangent,
                    max_nnz_per_row=np.int32(params["max_nnz_per_row"]),
                    subwarp_size=np.int32(params["subwarp_size"]),
                )
            return _input_vjp_call(params["num_cols"])(
                transpose_row_offsets,
                transpose_column_indices,
                transpose_permutation,
                current_values,
                current_cotangent,
                max_nnz_per_row=np.int32(
                    params["transpose_max_nnz_per_row"]
                ),
                subwarp_size=np.int32(params["subwarp_size"]),
            )

        if params["values_batched"]:
            input_cotangent = jax.lax.map(
                lambda pair: input_vjp_one(pair[0], pair[1]),
                (values, cotangent),
            )
        else:
            input_cotangent = jax.lax.map(
                lambda current_cotangent: input_vjp_one(
                    values, current_cotangent
                ),
                cotangent,
            )
        if not params["x_batched"]:
            input_cotangent = jnp.sum(input_cotangent, axis=0)
        return (
            values,
            row_offsets,
            column_indices,
            transpose_row_offsets,
            transpose_column_indices,
            transpose_permutation,
            input_cotangent,
        )

    if not ad.is_undefined_primal(values):
        raise AssertionError("weighted SpMV expected unknown values or x")

    def values_vjp_one(current_x, current_cotangent):
        return _values_vjp_call(column_indices.shape[0])(
            row_offsets,
            column_indices,
            current_x,
            current_cotangent,
            max_nnz_per_row=np.int32(params["max_nnz_per_row"]),
            subwarp_size=np.int32(params["subwarp_size"]),
        )

    if params["x_batched"]:
        values_cotangent = jax.lax.map(
            lambda pair: values_vjp_one(pair[0], pair[1]),
            (x, cotangent),
        )
    else:
        values_cotangent = jax.lax.map(
            lambda current_cotangent: values_vjp_one(x, current_cotangent),
            cotangent,
        )
    if not params["values_batched"]:
        values_cotangent = jnp.sum(values_cotangent, axis=0)
    return (
        values_cotangent,
        row_offsets,
        column_indices,
        transpose_row_offsets,
        transpose_column_indices,
        transpose_permutation,
        x,
    )


ad.primitive_transposes[
    _unweighted_batched_spmv_p
] = _unweighted_batched_transpose
ad.primitive_transposes[
    _weighted_batched_spmv_p
] = _weighted_batched_transpose


def csr_spmv(
    structure: CsrStructure,
    x: Any,
    *,
    values: Any | None = None,
    subwarp_size: int = 0,
    input_vjp_strategy: str = "scatter",
) -> jax.Array:
    """Compute a first-order differentiable arithmetic CSR product.

    ``input_vjp_strategy`` selects the weighted input-gradient implementation.
    Unweighted input gradients always use transposed CSR.
    """
    if not isinstance(structure, CsrStructure):
        raise TypeError("structure must be a CsrStructure")
    x_array = as_float_vector(x, "x")
    if x_array.shape[0] != structure.num_cols:
        raise ValueError("x length must equal structure.num_cols")
    subwarp = int(_validate_subwarp_size(subwarp_size))
    if input_vjp_strategy not in ("transpose", "scatter"):
        raise ValueError(
            "input_vjp_strategy must be 'transpose' or 'scatter'"
        )

    if values is None:
        return _unweighted_bind(structure, x_array, subwarp)
    values_array = as_float_vector(values, "values")
    if values_array.shape != structure.column_indices.shape:
        raise ValueError("values and column_indices must have equal shapes")
    return _weighted_bind(
        structure,
        values_array,
        x_array,
        subwarp,
        input_vjp_strategy,
    )
