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


@pytest.mark.parametrize(("operation", "function", "values_name"), CASES)
@pytest.mark.parametrize("weighted", [False, True])
def test_vmap_inputs_matches_stacked_calls(
    operation: str,
    function: Callable[..., jax.Array],
    values_name: str,
    weighted: bool,
    structure: springmv.CsrStructure,
) -> None:
    del operation
    inputs = jnp.stack((jnp.asarray(X), jnp.asarray(X) + 0.5))
    kwargs = {values_name: jnp.asarray(EDGE_VALUES)} if weighted else {}
    mapped = jax.jit(
        jax.vmap(lambda x: function(structure, x, **kwargs))
    )(inputs)
    expected = jnp.stack(
        [function(structure, inputs[index], **kwargs) for index in range(2)]
    )
    np.testing.assert_allclose(mapped, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("operation", "function", "values_name"), CASES)
def test_vmap_batched_values_and_inputs(
    operation: str,
    function: Callable[..., jax.Array],
    values_name: str,
    structure: springmv.CsrStructure,
) -> None:
    del operation
    inputs = jnp.stack((jnp.asarray(X), jnp.asarray(X) - 0.25))
    values = jnp.stack(
        (jnp.asarray(EDGE_VALUES), jnp.asarray(EDGE_VALUES) + 0.75)
    )
    mapped = jax.jit(
        jax.vmap(
            lambda current_values, x: function(
                structure, x, **{values_name: current_values}
            )
        )
    )(values, inputs)
    expected = jnp.stack(
        [
            function(
                structure,
                inputs[index],
                **{values_name: values[index]},
            )
            for index in range(2)
        ]
    )
    np.testing.assert_allclose(mapped, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("operation", "function", "values_name"), CASES)
def test_vmap_batched_values_with_shared_input(
    operation: str,
    function: Callable[..., jax.Array],
    values_name: str,
    structure: springmv.CsrStructure,
) -> None:
    del operation
    values = jnp.stack(
        (jnp.asarray(EDGE_VALUES), jnp.asarray(EDGE_VALUES) + 0.75)
    )
    actual = jax.jit(
        jax.vmap(
            lambda current_values: function(
                structure,
                jnp.asarray(X),
                **{values_name: current_values},
            )
        )
    )(values)
    expected = jnp.stack(
        [
            function(
                structure,
                jnp.asarray(X),
                **{values_name: values[index]},
            )
            for index in range(2)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("operation", "function", "values_name"), CASES)
def test_vmap_empty_batch(
    operation: str,
    function: Callable[..., jax.Array],
    values_name: str,
    structure: springmv.CsrStructure,
) -> None:
    del operation, values_name
    inputs = jnp.empty((0, X.size), dtype=jnp.float32)
    actual = jax.jit(jax.vmap(lambda x: function(structure, x)))(inputs)
    assert actual.shape == (0, structure.num_rows)
    assert actual.dtype == jnp.float32


def test_vmap_nonleading_input_axis(
    structure: springmv.CsrStructure,
) -> None:
    inputs = jnp.stack((jnp.asarray(X), jnp.asarray(X) + 1.0), axis=1)
    actual = jax.vmap(
        lambda x: springmv.csr_spmv(structure, x),
        in_axes=1,
        out_axes=1,
    )(inputs)
    expected = jnp.stack(
        [springmv.csr_spmv(structure, inputs[:, index]) for index in range(2)],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_vmap_of_jitted_operation(
    structure: springmv.CsrStructure,
) -> None:
    inputs = jnp.stack((jnp.asarray(X), jnp.asarray(X) + 1.0))
    single = jax.jit(lambda x: springmv.csr_spmv(structure, x))
    actual = jax.vmap(single)(inputs)
    expected = jnp.stack([single(inputs[index]) for index in range(2)])
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_vmap_rejects_batched_structure(
    structure: springmv.CsrStructure,
) -> None:
    batched_structure = jax.tree.map(
        lambda array: jnp.stack((array, array)), structure
    )
    inputs = jnp.stack((jnp.asarray(X), jnp.asarray(X)))
    with pytest.raises(NotImplementedError, match="shared CsrStructure"):
        jax.vmap(springmv.csr_spmv)(batched_structure, inputs)
