# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import springmv


ROW_OFFSETS = np.array([0, 3, 3, 5, 7], dtype=np.int32)
COLUMN_INDICES = np.array([0, 2, 2, 1, 3, 0, 3], dtype=np.int32)
X = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
EDGE_VALUES = np.array(
    [2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.25], dtype=np.float32
)


def _reference(
    operation: str, edge_values: np.ndarray | None
) -> np.ndarray:
    output = []
    for row in range(ROW_OFFSETS.size - 1):
        begin, end = ROW_OFFSETS[row : row + 2]
        inputs = X[COLUMN_INDICES[begin:end]]
        weights = None if edge_values is None else edge_values[begin:end]
        if operation == "spmv":
            terms = inputs if weights is None else weights * inputs
            output.append(np.sum(terms, dtype=np.float32))
        elif operation == "logsumexp":
            terms = inputs if weights is None else weights + inputs
            output.append(
                -np.inf if terms.size == 0 else np.logaddexp.reduce(terms)
            )
        elif operation == "minplus":
            terms = inputs if weights is None else weights + inputs
            output.append(np.inf if terms.size == 0 else np.min(terms))
        elif operation == "maxplus":
            terms = inputs if weights is None else weights + inputs
            output.append(-np.inf if terms.size == 0 else np.max(terms))
        else:
            raise AssertionError(f"unknown operation {operation}")
    return np.asarray(output, dtype=np.float32)


CASES: tuple[
    tuple[str, Callable[..., jax.Array], str], ...
] = (
    ("spmv", springmv.csr_spmv, "values"),
    ("logsumexp", springmv.csr_logsumexp, "log_values"),
    ("minplus", springmv.csr_minplus, "edge_values"),
    ("maxplus", springmv.csr_maxplus, "edge_values"),
)


@pytest.fixture(scope="module")
def structure() -> springmv.CsrStructure:
    return springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=X.size
    )


@pytest.mark.parametrize(("operation", "function", "values_name"), CASES)
@pytest.mark.parametrize("weighted", [False, True])
def test_reduction_matches_reference_eager_and_jit(
    operation: str,
    function: Callable[..., jax.Array],
    values_name: str,
    weighted: bool,
    structure: springmv.CsrStructure,
) -> None:
    x = jnp.asarray(X)
    kwargs = {values_name: jnp.asarray(EDGE_VALUES)} if weighted else {}
    expected = _reference(operation, EDGE_VALUES if weighted else None)

    actual = function(
        structure,
        x,
        **kwargs,
    )
    compiled = jax.jit(
        lambda input_x: function(
            structure,
            input_x,
            **kwargs,
        )
    )(x)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(compiled, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("operation", "function", "values_name"), CASES)
def test_explicit_subwarp_size(
    operation: str,
    function: Callable[..., jax.Array],
    values_name: str,
    structure: springmv.CsrStructure,
) -> None:
    del values_name
    actual = function(
        structure,
        X,
        subwarp_size=8,
    )
    np.testing.assert_allclose(
        actual, _reference(operation, None), rtol=2e-6, atol=2e-6
    )


def test_zero_row_matrix() -> None:
    structure = springmv.CsrStructure.from_arrays(
        jnp.array([0], dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32),
        num_cols=0,
    )
    actual = springmv.csr_spmv(
        structure,
        jnp.array([], dtype=jnp.float32),
    )
    assert actual.shape == (0,)
    assert actual.dtype == jnp.float32


def test_jit_accepts_dynamic_csr_structure(
    structure: springmv.CsrStructure,
) -> None:
    compiled = jax.jit(
        lambda input_structure, x: springmv.csr_spmv(input_structure, x)
    )
    actual = compiled(
        structure,
        jnp.asarray(X),
    )
    np.testing.assert_allclose(
        actual, _reference("spmv", None), rtol=2e-6, atol=2e-6
    )


def test_vmap_is_explicitly_unsupported(
    structure: springmv.CsrStructure,
) -> None:
    batch = jnp.stack((jnp.asarray(X), jnp.asarray(X)))
    with pytest.raises(NotImplementedError, match="vmap"):
        jax.vmap(
            lambda x: springmv.csr_spmv(
                structure,
                x,
            )
        )(batch)
