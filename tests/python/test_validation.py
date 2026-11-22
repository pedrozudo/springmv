# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax.numpy as jnp
import pytest

import springmv


ROW_OFFSETS = jnp.array([0, 1], dtype=jnp.int32)
COLUMN_INDICES = jnp.array([0], dtype=jnp.int32)
X = jnp.array([1.0], dtype=jnp.float32)


@pytest.fixture(scope="module")
def structure() -> springmv.CsrStructure:
    return springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=1
    )


def test_reduction_requires_vectors(
    structure: springmv.CsrStructure,
) -> None:
    with pytest.raises(ValueError, match="x must be one-dimensional"):
        springmv.csr_spmv(
            structure,
            X.reshape(1, 1),
        )


def test_reduction_requires_float32(
    structure: springmv.CsrStructure,
) -> None:
    with pytest.raises(TypeError, match="x must have dtype float32"):
        springmv.csr_spmv(
            structure,
            jnp.array([1], dtype=jnp.int32),
        )


def test_reduction_requires_int32_structure() -> None:
    with pytest.raises(TypeError, match="column_indices must have dtype int32"):
        springmv.CsrStructure.from_arrays(
            ROW_OFFSETS,
            jnp.array([0.0], dtype=jnp.float32),
            num_cols=1,
        )


def test_weight_shape_must_match_indices(
    structure: springmv.CsrStructure,
) -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        springmv.csr_logsumexp(
            structure,
            X,
            log_values=jnp.array([], dtype=jnp.float32),
        )


@pytest.mark.parametrize("subwarp_size", [-1, 3, 64])
def test_invalid_subwarp_size(
    subwarp_size: int, structure: springmv.CsrStructure
) -> None:
    with pytest.raises(ValueError, match="subwarp_size"):
        springmv.csr_spmv(
            structure,
            X,
            subwarp_size=subwarp_size,
        )


def test_vector_length_must_match_num_cols(
    structure: springmv.CsrStructure,
) -> None:
    with pytest.raises(ValueError, match="structure.num_cols"):
        springmv.csr_spmv(structure, jnp.array([], dtype=jnp.float32))


def test_reduction_requires_csr_structure() -> None:
    with pytest.raises(TypeError, match="CsrStructure"):
        springmv.csr_spmv(ROW_OFFSETS, X)


def test_transpose_weight_shape_must_match_indices() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        springmv.csr_transpose(
            ROW_OFFSETS,
            COLUMN_INDICES,
            num_cols=1,
            values=jnp.array([], dtype=jnp.float32),
        )


def test_invalid_num_cols() -> None:
    with pytest.raises(ValueError, match="num_cols"):
        springmv.csr_transpose(
            ROW_OFFSETS, COLUMN_INDICES, num_cols=-1
        )
