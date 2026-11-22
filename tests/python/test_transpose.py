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
VALUES = jnp.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=jnp.float32)

EXPECTED_ROW_OFFSETS = np.array([0, 1, 2, 5, 5], dtype=np.int32)
EXPECTED_COLUMN_INDICES = np.array([0, 2, 0, 1, 2], dtype=np.int32)
EXPECTED_VALUES = np.array([11.0, 13.0, 10.0, 12.0, 14.0], dtype=np.float32)
EXPECTED_PERMUTATION = np.array([1, 3, 0, 2, 4], dtype=np.int32)


def test_unweighted_transpose() -> None:
    row_offsets, column_indices = springmv.csr_transpose(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=4
    )
    np.testing.assert_array_equal(row_offsets, EXPECTED_ROW_OFFSETS)
    np.testing.assert_array_equal(column_indices, EXPECTED_COLUMN_INDICES)


def test_unweighted_transpose_with_permutation() -> None:
    row_offsets, column_indices, permutation = springmv.csr_transpose(
        ROW_OFFSETS,
        COLUMN_INDICES,
        num_cols=4,
        return_permutation=True,
    )
    np.testing.assert_array_equal(row_offsets, EXPECTED_ROW_OFFSETS)
    np.testing.assert_array_equal(column_indices, EXPECTED_COLUMN_INDICES)
    np.testing.assert_array_equal(permutation, EXPECTED_PERMUTATION)


def test_weighted_transpose() -> None:
    row_offsets, column_indices, values = springmv.csr_transpose(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=4, values=VALUES
    )
    np.testing.assert_array_equal(row_offsets, EXPECTED_ROW_OFFSETS)
    np.testing.assert_array_equal(column_indices, EXPECTED_COLUMN_INDICES)
    np.testing.assert_array_equal(values, EXPECTED_VALUES)


def test_weighted_transpose_with_permutation() -> None:
    row_offsets, column_indices, values, permutation = springmv.csr_transpose(
        ROW_OFFSETS,
        COLUMN_INDICES,
        num_cols=4,
        values=VALUES,
        return_permutation=True,
    )
    np.testing.assert_array_equal(row_offsets, EXPECTED_ROW_OFFSETS)
    np.testing.assert_array_equal(column_indices, EXPECTED_COLUMN_INDICES)
    np.testing.assert_array_equal(values, EXPECTED_VALUES)
    np.testing.assert_array_equal(permutation, EXPECTED_PERMUTATION)
    np.testing.assert_array_equal(values, np.asarray(VALUES)[permutation])


def test_empty_transpose() -> None:
    row_offsets, column_indices, permutation = springmv.csr_transpose(
        jnp.array([0], dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32),
        num_cols=3,
        return_permutation=True,
    )
    np.testing.assert_array_equal(row_offsets, np.zeros(4, dtype=np.int32))
    assert column_indices.shape == (0,)
    assert permutation.shape == (0,)


def test_transpose_rejects_jit() -> None:
    compiled = jax.jit(
        lambda row_offsets, column_indices: springmv.csr_transpose(
            row_offsets, column_indices, num_cols=4
        )
    )
    with pytest.raises(TypeError, match="preprocessing operation"):
        compiled(ROW_OFFSETS, COLUMN_INDICES)
