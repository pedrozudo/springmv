"""Immutable CSR topology and preprocessing metadata."""

from __future__ import annotations

import dataclasses
from dataclasses import field
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._transpose import csr_transpose
from ._validation import as_structure_array, require_eager_cuda


def _host_scalar(value: jax.Array) -> int:
    return int(np.asarray(jax.device_get(value)).item())


def _max_row_length(row_offsets: jax.Array) -> int:
    if row_offsets.shape[0] <= 1:
        return 0
    return _host_scalar(jnp.max(row_offsets[1:] - row_offsets[:-1]))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CsrStructure:
    """CSR topology together with its precomputed transpose.

    The array fields are JAX PyTree leaves. Dimensions and scheduling metadata
    are static fields, so a structure can be passed through ``jax.jit`` without
    introducing native resources or hidden mutable state.
    """

    row_offsets: jax.Array
    column_indices: jax.Array
    transpose_row_offsets: jax.Array
    transpose_column_indices: jax.Array
    transpose_permutation: jax.Array
    num_rows: int = field(metadata={"static": True})
    num_cols: int = field(metadata={"static": True})
    num_nonzeros: int = field(metadata={"static": True})
    max_nnz_per_row: int = field(metadata={"static": True})
    transpose_max_nnz_per_row: int = field(metadata={"static": True})

    @classmethod
    def from_arrays(
        cls,
        row_offsets: Any,
        column_indices: Any,
        *,
        num_cols: int,
    ) -> CsrStructure:
        """Validate CSR topology and eagerly build its transpose."""
        if isinstance(num_cols, bool) or not isinstance(num_cols, Integral):
            raise TypeError("num_cols must be an integer")
        num_cols = int(num_cols)
        if num_cols < 0 or num_cols > np.iinfo(np.int32).max:
            raise ValueError("num_cols must be a nonnegative int32 value")

        row_offsets_array = as_structure_array(row_offsets, "row_offsets")
        column_indices_array = as_structure_array(
            column_indices, "column_indices"
        )
        require_eager_cuda(
            row_offsets_array,
            column_indices_array,
            operation="CsrStructure.from_arrays",
        )
        if row_offsets_array.shape[0] == 0:
            raise ValueError("row_offsets must contain at least one entry")

        num_rows = row_offsets_array.shape[0] - 1
        num_nonzeros = column_indices_array.shape[0]
        if num_rows > np.iinfo(np.int32).max:
            raise ValueError("number of rows is too large for int32 CSR")
        if num_nonzeros > np.iinfo(np.int32).max:
            raise ValueError("number of nonzeros is too large for int32 CSR")
        if _host_scalar(row_offsets_array[0]) != 0:
            raise ValueError("row_offsets must start at zero")
        if _host_scalar(row_offsets_array[-1]) != num_nonzeros:
            raise ValueError(
                "the final row offset must equal the number of nonzeros"
            )
        if num_rows > 0 and _host_scalar(
            jnp.any(row_offsets_array[1:] < row_offsets_array[:-1])
        ):
            raise ValueError("row_offsets must be nondecreasing")
        if num_nonzeros > 0:
            columns_valid = jnp.all(
                (column_indices_array >= 0)
                & (column_indices_array < num_cols)
            )
            if not bool(np.asarray(jax.device_get(columns_valid)).item()):
                raise ValueError("column_indices entries must be in range")

        max_nnz_per_row = _max_row_length(row_offsets_array)
        (
            transpose_row_offsets,
            transpose_column_indices,
            transpose_permutation,
        ) = csr_transpose(
            row_offsets_array,
            column_indices_array,
            num_cols=num_cols,
            return_permutation=True,
        )
        transpose_max_nnz_per_row = _max_row_length(transpose_row_offsets)

        return cls(
            row_offsets=row_offsets_array,
            column_indices=column_indices_array,
            transpose_row_offsets=transpose_row_offsets,
            transpose_column_indices=transpose_column_indices,
            transpose_permutation=transpose_permutation,
            num_rows=num_rows,
            num_cols=num_cols,
            num_nonzeros=num_nonzeros,
            max_nnz_per_row=max_nnz_per_row,
            transpose_max_nnz_per_row=transpose_max_nnz_per_row,
        )

    def transpose(self) -> CsrStructure:
        """Return a structure for the transposed matrix without rebuilding it."""
        require_eager_cuda(
            self.row_offsets,
            self.column_indices,
            self.transpose_row_offsets,
            self.transpose_column_indices,
            self.transpose_permutation,
            operation="CsrStructure.transpose",
        )
        inverse_permutation = jnp.empty_like(self.transpose_permutation)
        inverse_permutation = inverse_permutation.at[
            self.transpose_permutation
        ].set(jnp.arange(self.num_nonzeros, dtype=jnp.int32))
        return CsrStructure(
            row_offsets=self.transpose_row_offsets,
            column_indices=self.transpose_column_indices,
            transpose_row_offsets=self.row_offsets,
            transpose_column_indices=self.column_indices,
            transpose_permutation=inverse_permutation,
            num_rows=self.num_cols,
            num_cols=self.num_rows,
            num_nonzeros=self.num_nonzeros,
            max_nnz_per_row=self.transpose_max_nnz_per_row,
            transpose_max_nnz_per_row=self.max_nnz_per_row,
        )

    @property
    def T(self) -> CsrStructure:
        """Alias for :meth:`transpose`."""
        return self.transpose()
