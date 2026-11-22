# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import springmv


ROW_OFFSETS = jnp.array([0, 2, 3, 5], dtype=jnp.int32)
COLUMN_INDICES = jnp.array([2, 0, 2, 1, 2], dtype=jnp.int32)


@pytest.fixture(scope="module")
def structure() -> springmv.CsrStructure:
    return springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=4
    )


def test_structure_metadata(structure: springmv.CsrStructure) -> None:
    assert structure.num_rows == 3
    assert structure.num_cols == 4
    assert structure.num_nonzeros == 5
    assert structure.max_nnz_per_row == 2
    assert structure.transpose_max_nnz_per_row == 3
    np.testing.assert_array_equal(
        structure.transpose_row_offsets,
        np.array([0, 1, 2, 5, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        structure.transpose_column_indices,
        np.array([0, 2, 0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        structure.transpose_permutation,
        np.array([1, 3, 0, 2, 4], dtype=np.int32),
    )


def test_structure_is_a_jax_pytree(structure: springmv.CsrStructure) -> None:
    leaves, tree = jax.tree.flatten(structure)
    assert len(leaves) == 5
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert rebuilt.num_rows == structure.num_rows
    np.testing.assert_array_equal(rebuilt.row_offsets, structure.row_offsets)


def test_transpose_returns_new_structure(
    structure: springmv.CsrStructure,
) -> None:
    transposed = structure.transpose()
    assert transposed is not structure
    assert transposed.num_rows == structure.num_cols
    assert transposed.num_cols == structure.num_rows
    assert transposed.max_nnz_per_row == structure.transpose_max_nnz_per_row
    assert transposed.transpose_max_nnz_per_row == structure.max_nnz_per_row
    np.testing.assert_array_equal(
        transposed.row_offsets, structure.transpose_row_offsets
    )
    np.testing.assert_array_equal(
        transposed.column_indices, structure.transpose_column_indices
    )
    np.testing.assert_array_equal(
        transposed.transpose_row_offsets, structure.row_offsets
    )
    np.testing.assert_array_equal(
        transposed.transpose_column_indices, structure.column_indices
    )


def test_transpose_permutation_reorders_external_values(
    structure: springmv.CsrStructure,
) -> None:
    values = jnp.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=jnp.float32)
    transposed_values = values[structure.transpose_permutation]
    np.testing.assert_array_equal(
        transposed_values,
        np.array([11.0, 13.0, 10.0, 12.0, 14.0], dtype=np.float32),
    )


def test_transposed_structure_can_drive_weighted_spmv(
    structure: springmv.CsrStructure,
) -> None:
    values = jnp.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=jnp.float32)
    result = springmv.csr_spmv(
        structure.transpose(),
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        values=values[structure.transpose_permutation],
    )
    np.testing.assert_array_equal(
        result, np.array([11.0, 39.0, 76.0, 0.0], dtype=np.float32)
    )


def test_double_transpose_recovers_original_order(
    structure: springmv.CsrStructure,
) -> None:
    recovered = structure.T.T
    np.testing.assert_array_equal(recovered.row_offsets, structure.row_offsets)
    np.testing.assert_array_equal(
        recovered.column_indices, structure.column_indices
    )
    np.testing.assert_array_equal(
        recovered.transpose_permutation, structure.transpose_permutation
    )


@pytest.mark.parametrize(
    ("row_offsets", "column_indices", "message"),
    [
        ([1, 1], [], "start at zero"),
        ([0, 2], [0], "final row offset"),
        ([0, 2, 1], [0], "nondecreasing"),
        ([0, 1], [2], "in range"),
    ],
)
def test_structure_rejects_invalid_csr(
    row_offsets: list[int], column_indices: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        springmv.CsrStructure.from_arrays(
            jnp.asarray(row_offsets, dtype=jnp.int32),
            jnp.asarray(column_indices, dtype=jnp.int32),
            num_cols=2,
        )
