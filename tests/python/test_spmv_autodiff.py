# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import springmv


ROW_OFFSETS = jnp.array([0, 3, 3, 5, 7], dtype=jnp.int32)
COLUMN_INDICES = jnp.array([0, 2, 2, 1, 4, 0, 3], dtype=jnp.int32)
VALUES = jnp.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.25], dtype=jnp.float32)
X = jnp.array([1.0, -2.0, 3.0, 4.0, -1.0], dtype=jnp.float32)
X_DOT = jnp.array([-0.5, 2.0, 1.0, -3.0, 0.25], dtype=jnp.float32)
VALUES_DOT = jnp.array(
    [0.1, -0.2, 0.3, 0.5, -0.7, 0.9, 1.1], dtype=jnp.float32
)
OUTPUT_COTANGENT = jnp.array([0.5, -2.0, 3.0, -1.0], dtype=jnp.float32)


@pytest.fixture(scope="module")
def structure() -> springmv.CsrStructure:
    return springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=X.shape[0]
    )


def _edge_rows() -> jax.Array:
    return jnp.repeat(
        jnp.arange(ROW_OFFSETS.shape[0] - 1, dtype=jnp.int32),
        jnp.diff(ROW_OFFSETS),
        total_repeat_length=COLUMN_INDICES.shape[0],
    )


def _reference_spmv(x: jax.Array, values: jax.Array | None) -> jax.Array:
    terms = x[COLUMN_INDICES]
    if values is not None:
        terms = values * terms
    return jnp.zeros(ROW_OFFSETS.shape[0] - 1, dtype=jnp.float32).at[
        _edge_rows()
    ].add(terms)


def _reference_input_vjp(
    cotangent: jax.Array, values: jax.Array | None
) -> jax.Array:
    terms = cotangent[_edge_rows()]
    if values is not None:
        terms = values * terms
    return jnp.zeros(X.shape[0], dtype=jnp.float32).at[COLUMN_INDICES].add(
        terms
    )


def test_unweighted_jvp(structure: springmv.CsrStructure) -> None:
    primal, tangent = jax.jvp(
        lambda x: springmv.csr_spmv(structure, x),
        (X,),
        (X_DOT,),
    )
    np.testing.assert_allclose(primal, _reference_spmv(X, None), rtol=2e-6)
    np.testing.assert_allclose(
        tangent, _reference_spmv(X_DOT, None), rtol=2e-6
    )


def test_weighted_jvp_x(structure: springmv.CsrStructure) -> None:
    primal, tangent = jax.jvp(
        lambda x: springmv.csr_spmv(structure, x, values=VALUES),
        (X,),
        (X_DOT,),
    )
    np.testing.assert_allclose(primal, _reference_spmv(X, VALUES), rtol=2e-6)
    np.testing.assert_allclose(
        tangent, _reference_spmv(X_DOT, VALUES), rtol=2e-6
    )


def test_weighted_jvp_values(structure: springmv.CsrStructure) -> None:
    primal, tangent = jax.jvp(
        lambda values: springmv.csr_spmv(structure, X, values=values),
        (VALUES,),
        (VALUES_DOT,),
    )
    np.testing.assert_allclose(primal, _reference_spmv(X, VALUES), rtol=2e-6)
    np.testing.assert_allclose(
        tangent, _reference_spmv(X, VALUES_DOT), rtol=2e-6
    )


def test_weighted_jvp_values_and_x(
    structure: springmv.CsrStructure,
) -> None:
    primal, tangent = jax.jvp(
        lambda values, x: springmv.csr_spmv(structure, x, values=values),
        (VALUES, X),
        (VALUES_DOT, X_DOT),
    )
    expected_tangent = _reference_spmv(X_DOT, VALUES) + _reference_spmv(
        X, VALUES_DOT
    )
    np.testing.assert_allclose(primal, _reference_spmv(X, VALUES), rtol=2e-6)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-6)


def test_weighted_jvp_zero_tangents(
    structure: springmv.CsrStructure,
) -> None:
    _, tangent = jax.jvp(
        lambda values, x: springmv.csr_spmv(structure, x, values=values),
        (VALUES, X),
        (jnp.zeros_like(VALUES), jnp.zeros_like(X)),
    )
    np.testing.assert_array_equal(
        tangent, jnp.zeros(structure.num_rows, dtype=jnp.float32)
    )


def test_unweighted_vjp_with_arbitrary_cotangent(
    structure: springmv.CsrStructure,
) -> None:
    primal, pullback = jax.vjp(lambda x: springmv.csr_spmv(structure, x), X)
    (input_cotangent,) = pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(primal, _reference_spmv(X, None), rtol=2e-6)
    np.testing.assert_allclose(
        input_cotangent,
        _reference_input_vjp(OUTPUT_COTANGENT, None),
        rtol=2e-6,
    )


def test_weighted_vjp_with_arbitrary_cotangent(
    structure: springmv.CsrStructure,
) -> None:
    primal, pullback = jax.vjp(
        lambda values, x: springmv.csr_spmv(
            structure, x, values=values
        ),
        VALUES,
        X,
    )
    values_cotangent, input_cotangent = pullback(OUTPUT_COTANGENT)
    expected_values = OUTPUT_COTANGENT[_edge_rows()] * X[COLUMN_INDICES]
    np.testing.assert_allclose(primal, _reference_spmv(X, VALUES), rtol=2e-6)
    np.testing.assert_allclose(values_cotangent, expected_values, rtol=2e-6)
    np.testing.assert_allclose(
        input_cotangent,
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES),
        rtol=2e-6,
    )


def test_value_and_grad_both_arguments(
    structure: springmv.CsrStructure,
) -> None:
    def loss(values, x):
        return jnp.vdot(
            springmv.csr_spmv(structure, x, values=values),
            OUTPUT_COTANGENT,
        )

    value, (values_cotangent, input_cotangent) = jax.value_and_grad(
        loss, argnums=(0, 1)
    )(VALUES, X)
    expected_output = _reference_spmv(X, VALUES)
    np.testing.assert_allclose(
        value, jnp.vdot(expected_output, OUTPUT_COTANGENT), rtol=2e-6
    )
    np.testing.assert_allclose(
        values_cotangent,
        OUTPUT_COTANGENT[_edge_rows()] * X[COLUMN_INDICES],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        input_cotangent,
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES),
        rtol=2e-6,
    )


def test_jitted_jvp_and_grad(structure: springmv.CsrStructure) -> None:
    jitted_jvp = jax.jit(
        lambda values, x, values_dot, x_dot: jax.jvp(
            lambda current_values, current_x: springmv.csr_spmv(
                structure, current_x, values=current_values
            ),
            (values, x),
            (values_dot, x_dot),
        )
    )
    primal, tangent = jitted_jvp(VALUES, X, VALUES_DOT, X_DOT)
    np.testing.assert_allclose(primal, _reference_spmv(X, VALUES), rtol=2e-6)
    np.testing.assert_allclose(
        tangent,
        _reference_spmv(X_DOT, VALUES) + _reference_spmv(X, VALUES_DOT),
        rtol=2e-6,
    )

    jitted_grad = jax.jit(
        jax.grad(
            lambda values, x: jnp.vdot(
                springmv.csr_spmv(structure, x, values=values),
                OUTPUT_COTANGENT,
            ),
            argnums=(0, 1),
        )
    )
    values_cotangent, input_cotangent = jitted_grad(VALUES, X)
    np.testing.assert_allclose(
        values_cotangent,
        OUTPUT_COTANGENT[_edge_rows()] * X[COLUMN_INDICES],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        input_cotangent,
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES),
        rtol=2e-6,
    )


def test_jitted_grad_accepts_dynamic_structure(
    structure: springmv.CsrStructure,
) -> None:
    @jax.jit
    def differentiated(input_structure, values, x):
        return jax.grad(
            lambda current_values, current_x: jnp.vdot(
                springmv.csr_spmv(
                    input_structure, current_x, values=current_values
                ),
                OUTPUT_COTANGENT,
            ),
            argnums=(0, 1),
        )(values, x)

    values_cotangent, input_cotangent = differentiated(
        structure, VALUES, X
    )
    np.testing.assert_allclose(
        values_cotangent,
        OUTPUT_COTANGENT[_edge_rows()] * X[COLUMN_INDICES],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        input_cotangent,
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES),
        rtol=2e-6,
    )


@pytest.mark.parametrize("subwarp_size", [0, 1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_vjp_all_subwarp_widths(
    structure: springmv.CsrStructure,
    subwarp_size: int,
    input_vjp_strategy: str,
) -> None:
    _, pullback = jax.vjp(
        lambda values, x: springmv.csr_spmv(
            structure,
            x,
            values=values,
            subwarp_size=subwarp_size,
            input_vjp_strategy=input_vjp_strategy,
        ),
        VALUES,
        X,
    )
    values_cotangent, input_cotangent = pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(
        values_cotangent,
        OUTPUT_COTANGENT[_edge_rows()] * X[COLUMN_INDICES],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        input_cotangent,
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES),
        rtol=2e-6,
    )


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_empty_structure_autodiff(input_vjp_strategy: str) -> None:
    structure = springmv.CsrStructure.from_arrays(
        jnp.array([0, 0, 0], dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32),
        num_cols=3,
    )
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    values = jnp.array([], dtype=jnp.float32)
    output_cotangent = jnp.array([2.0, -1.0], dtype=jnp.float32)

    _, pullback = jax.vjp(
        lambda current_values, current_x: springmv.csr_spmv(
            structure,
            current_x,
            values=current_values,
            input_vjp_strategy=input_vjp_strategy,
        ),
        values,
        x,
    )
    values_cotangent, input_cotangent = pullback(output_cotangent)
    assert values_cotangent.shape == (0,)
    np.testing.assert_array_equal(input_cotangent, jnp.zeros_like(x))


def test_scatter_input_vjp_under_jit(
    structure: springmv.CsrStructure,
) -> None:
    differentiated = jax.jit(
        jax.grad(
            lambda x: jnp.vdot(
                springmv.csr_spmv(
                    structure,
                    x,
                    values=VALUES,
                    input_vjp_strategy="scatter",
                ),
                OUTPUT_COTANGENT,
            )
        )
    )
    np.testing.assert_allclose(
        differentiated(X),
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES),
        rtol=3e-6,
    )


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_vmap_of_grad(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    inputs = jnp.stack((X, X + 0.25, X - 0.5))
    differentiated = jax.jit(
        jax.vmap(
            jax.grad(
                lambda x: jnp.vdot(
                    springmv.csr_spmv(
                        structure,
                        x,
                        values=VALUES,
                        input_vjp_strategy=input_vjp_strategy,
                    ),
                    OUTPUT_COTANGENT,
                )
            )
        )
    )
    actual = differentiated(inputs)
    expected = jnp.broadcast_to(
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES), inputs.shape
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_grad_through_vmap_with_shared_values(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    inputs = jnp.stack((X, X + 0.25, X - 0.5))

    def loss(values, current_inputs):
        outputs = jax.vmap(
            lambda x: springmv.csr_spmv(
                structure,
                x,
                values=values,
                input_vjp_strategy=input_vjp_strategy,
            )
        )(current_inputs)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    values_cotangent, input_cotangents = jax.jit(
        jax.grad(loss, argnums=(0, 1))
    )(VALUES, inputs)
    expected_values = jnp.sum(
        OUTPUT_COTANGENT[None, _edge_rows()]
        * inputs[:, COLUMN_INDICES],
        axis=0,
    )
    expected_inputs = jnp.broadcast_to(
        _reference_input_vjp(OUTPUT_COTANGENT, VALUES), inputs.shape
    )
    np.testing.assert_allclose(values_cotangent, expected_values, rtol=3e-6)
    np.testing.assert_allclose(input_cotangents, expected_inputs, rtol=3e-6)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_vmap_grad_with_batched_values_and_inputs(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    values = jnp.stack((VALUES, VALUES + 0.5))
    inputs = jnp.stack((X, X - 0.25))

    def loss(current_values, current_x):
        return jnp.vdot(
            springmv.csr_spmv(
                structure,
                current_x,
                values=current_values,
                input_vjp_strategy=input_vjp_strategy,
            ),
            OUTPUT_COTANGENT,
        )

    actual = jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1))))(
        values, inputs
    )
    expected_values = OUTPUT_COTANGENT[None, _edge_rows()] * inputs[
        :, COLUMN_INDICES
    ]
    expected_inputs = jax.vmap(
        lambda current_values: _reference_input_vjp(
            OUTPUT_COTANGENT, current_values
        )
    )(values)
    np.testing.assert_allclose(actual[0], expected_values, rtol=3e-6)
    np.testing.assert_allclose(actual[1], expected_inputs, rtol=3e-6)


def test_invalid_input_vjp_strategy(
    structure: springmv.CsrStructure,
) -> None:
    with pytest.raises(ValueError, match="input_vjp_strategy"):
        springmv.csr_spmv(
            structure,
            X,
            values=VALUES,
            input_vjp_strategy="invalid",
        )


@pytest.mark.parametrize(
    "operation",
    [springmv.csr_minplus, springmv.csr_maxplus],
)
def test_tropical_operations_are_not_differentiable(operation) -> None:
    structure = springmv.CsrStructure.from_arrays(
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        num_cols=1,
    )
    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.grad(lambda x: operation(structure, x).sum())(
            jnp.array([1.0], dtype=jnp.float32)
        )
