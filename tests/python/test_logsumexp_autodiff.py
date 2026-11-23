# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

import springmv


ROW_OFFSETS_HOST = (0, 3, 3, 5, 8)
ROW_OFFSETS = jnp.array(ROW_OFFSETS_HOST, dtype=jnp.int32)
COLUMN_INDICES = jnp.array(
    [0, 2, 2, 1, 4, 0, 3, 4], dtype=jnp.int32
)
LOG_VALUES = jnp.array(
    [0.2, -0.7, 0.4, 1.1, -0.3, 0.5, -1.2, 0.8],
    dtype=jnp.float32,
)
LOG_X = jnp.array([0.3, -0.9, 1.4, 0.2, -0.5], dtype=jnp.float32)
VALUES_TANGENT = jnp.array(
    [0.1, -0.2, 0.3, 0.5, -0.7, 0.9, 1.1, -0.4],
    dtype=jnp.float32,
)
INPUT_TANGENT = jnp.array(
    [-0.5, 2.0, 1.0, -3.0, 0.25], dtype=jnp.float32
)
OUTPUT_COTANGENT = jnp.array(
    [0.5, -2.0, 3.0, -1.0], dtype=jnp.float32
)
NONEMPTY_ROWS = jnp.array([0, 2, 3], dtype=jnp.int32)


@pytest.fixture(scope="module")
def structure() -> springmv.CsrStructure:
    return springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=LOG_X.shape[0]
    )


def _reference_logsumexp(
    log_values: jax.Array | None, log_x: jax.Array
) -> jax.Array:
    rows = []
    for row in range(len(ROW_OFFSETS_HOST) - 1):
        begin = ROW_OFFSETS_HOST[row]
        end = ROW_OFFSETS_HOST[row + 1]
        if begin == end:
            rows.append(jnp.asarray(-jnp.inf, dtype=jnp.float32))
            continue
        terms = log_x[COLUMN_INDICES[begin:end]]
        if log_values is not None:
            terms = terms + log_values[begin:end]
        rows.append(jsp.special.logsumexp(terms))
    return jnp.stack(rows)


def test_unweighted_jvp_matches_pure_jax(
    structure: springmv.CsrStructure,
) -> None:
    actual = jax.jvp(
        lambda x: springmv.csr_logsumexp(structure, x),
        (LOG_X,),
        (INPUT_TANGENT,),
    )
    expected = jax.jvp(
        lambda x: _reference_logsumexp(None, x),
        (LOG_X,),
        (INPUT_TANGENT,),
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-6)


def test_weighted_jvp_matches_pure_jax(
    structure: springmv.CsrStructure,
) -> None:
    actual = jax.jvp(
        lambda values, x: springmv.csr_logsumexp(
            structure, x, log_values=values
        ),
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    expected = jax.jvp(
        _reference_logsumexp,
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=4e-6, atol=5e-7)


@pytest.mark.parametrize("differentiate_values", [False, True])
def test_weighted_partial_jvp_matches_pure_jax(
    structure: springmv.CsrStructure, differentiate_values: bool
) -> None:
    if differentiate_values:
        actual = jax.jvp(
            lambda values: springmv.csr_logsumexp(
                structure, LOG_X, log_values=values
            ),
            (LOG_VALUES,),
            (VALUES_TANGENT,),
        )
        expected = jax.jvp(
            lambda values: _reference_logsumexp(values, LOG_X),
            (LOG_VALUES,),
            (VALUES_TANGENT,),
        )
    else:
        actual = jax.jvp(
            lambda x: springmv.csr_logsumexp(
                structure, x, log_values=LOG_VALUES
            ),
            (LOG_X,),
            (INPUT_TANGENT,),
        )
        expected = jax.jvp(
            lambda x: _reference_logsumexp(LOG_VALUES, x),
            (LOG_X,),
            (INPUT_TANGENT,),
        )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=4e-6, atol=5e-7)


def test_weighted_zero_tangents(
    structure: springmv.CsrStructure,
) -> None:
    _, tangent = jax.jvp(
        lambda values, x: springmv.csr_logsumexp(
            structure, x, log_values=values
        ),
        (LOG_VALUES, LOG_X),
        (jnp.zeros_like(LOG_VALUES), jnp.zeros_like(LOG_X)),
    )
    np.testing.assert_array_equal(
        tangent, jnp.zeros(structure.num_rows, dtype=jnp.float32)
    )


def test_unweighted_vjp_matches_pure_jax(
    structure: springmv.CsrStructure,
) -> None:
    actual_output, actual_pullback = jax.vjp(
        lambda x: springmv.csr_logsumexp(structure, x), LOG_X
    )
    expected_output, expected_pullback = jax.vjp(
        lambda x: _reference_logsumexp(None, x), LOG_X
    )
    (actual_input,) = actual_pullback(OUTPUT_COTANGENT)
    (expected_input,) = expected_pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(actual_output, expected_output, rtol=3e-6)
    np.testing.assert_allclose(actual_input, expected_input, rtol=4e-6)


def test_weighted_vjp_matches_pure_jax(
    structure: springmv.CsrStructure,
) -> None:
    actual_output, actual_pullback = jax.vjp(
        lambda values, x: springmv.csr_logsumexp(
            structure, x, log_values=values
        ),
        LOG_VALUES,
        LOG_X,
    )
    expected_output, expected_pullback = jax.vjp(
        _reference_logsumexp, LOG_VALUES, LOG_X
    )
    actual_values, actual_input = actual_pullback(OUTPUT_COTANGENT)
    expected_values, expected_input = expected_pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(actual_output, expected_output, rtol=3e-6)
    np.testing.assert_allclose(actual_values, expected_values, rtol=4e-6)
    np.testing.assert_allclose(actual_input, expected_input, rtol=4e-6)


def test_jvp_vjp_adjoint_identity(
    structure: springmv.CsrStructure,
) -> None:
    operation = lambda values, x: springmv.csr_logsumexp(
        structure, x, log_values=values
    )
    _, output_tangent = jax.jvp(
        operation,
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    _, pullback = jax.vjp(operation, LOG_VALUES, LOG_X)
    values_cotangent, input_cotangent = pullback(OUTPUT_COTANGENT)
    lhs = jnp.vdot(output_tangent, OUTPUT_COTANGENT)
    rhs = jnp.vdot(VALUES_TANGENT, values_cotangent) + jnp.vdot(
        INPUT_TANGENT, input_cotangent
    )
    np.testing.assert_allclose(lhs, rhs, rtol=5e-6, atol=2e-6)


def test_directional_derivative_matches_finite_difference(
    structure: springmv.CsrStructure,
) -> None:
    epsilon = 2.0e-3

    def operation(values, x):
        return springmv.csr_logsumexp(structure, x, log_values=values)

    def finite_loss(values, x):
        output = operation(values, x)
        return jnp.vdot(
            output[NONEMPTY_ROWS], OUTPUT_COTANGENT[NONEMPTY_ROWS]
        )

    _, output_tangent = jax.jvp(
        operation,
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    jvp_contraction = jnp.vdot(
        output_tangent[NONEMPTY_ROWS], OUTPUT_COTANGENT[NONEMPTY_ROWS]
    )
    _, pullback = jax.vjp(operation, LOG_VALUES, LOG_X)
    values_cotangent, input_cotangent = pullback(OUTPUT_COTANGENT)
    vjp_contraction = jnp.vdot(VALUES_TANGENT, values_cotangent) + jnp.vdot(
        INPUT_TANGENT, input_cotangent
    )
    finite_difference = (
        finite_loss(
            LOG_VALUES + epsilon * VALUES_TANGENT,
            LOG_X + epsilon * INPUT_TANGENT,
        )
        - finite_loss(
            LOG_VALUES - epsilon * VALUES_TANGENT,
            LOG_X - epsilon * INPUT_TANGENT,
        )
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        jvp_contraction, finite_difference, rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(
        vjp_contraction, finite_difference, rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_jitted_jvp_and_vjp(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    jitted_jvp = jax.jit(
        lambda values, x, values_dot, x_dot: jax.jvp(
            lambda current_values, current_x: springmv.csr_logsumexp(
                    structure,
                    current_x,
                    log_values=current_values,
                    input_vjp_strategy=input_vjp_strategy,
            ),
            (values, x),
            (values_dot, x_dot),
        )
    )
    actual = jitted_jvp(
        LOG_VALUES, LOG_X, VALUES_TANGENT, INPUT_TANGENT
    )
    expected = jax.jvp(
        _reference_logsumexp,
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=4e-6, atol=5e-7)

    @jax.jit
    def pull_back(values, x, cotangent):
        _, pullback = jax.vjp(
            lambda current_values, current_x: springmv.csr_logsumexp(
                structure,
                current_x,
                log_values=current_values,
                input_vjp_strategy=input_vjp_strategy,
            ),
            values,
            x,
        )
        return pullback(cotangent)

    actual_values, actual_input = pull_back(
        LOG_VALUES, LOG_X, OUTPUT_COTANGENT
    )
    _, reference_pullback = jax.vjp(
        _reference_logsumexp, LOG_VALUES, LOG_X
    )
    expected_values, expected_input = reference_pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(actual_values, expected_values, rtol=4e-6)
    np.testing.assert_allclose(actual_input, expected_input, rtol=4e-6)


def test_jitted_grad_accepts_dynamic_structure(
    structure: springmv.CsrStructure,
) -> None:
    @jax.jit
    def differentiated(input_structure, values, x):
        return jax.grad(
            lambda current_values, current_x: jnp.vdot(
                springmv.csr_logsumexp(
                    input_structure,
                    current_x,
                    log_values=current_values,
                )[NONEMPTY_ROWS],
                OUTPUT_COTANGENT[NONEMPTY_ROWS],
            ),
            argnums=(0, 1),
        )(values, x)

    actual_values, actual_input = differentiated(
        structure, LOG_VALUES, LOG_X
    )
    _, reference_pullback = jax.vjp(
        _reference_logsumexp, LOG_VALUES, LOG_X
    )
    expected_values, expected_input = reference_pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(actual_values, expected_values, rtol=4e-6)
    np.testing.assert_allclose(actual_input, expected_input, rtol=4e-6)


@pytest.mark.parametrize("subwarp_size", [0, 1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_all_subwarp_widths(
    structure: springmv.CsrStructure,
    subwarp_size: int,
    input_vjp_strategy: str,
) -> None:
    operation = lambda values, x: springmv.csr_logsumexp(
        structure,
        x,
        log_values=values,
        subwarp_size=subwarp_size,
        input_vjp_strategy=input_vjp_strategy,
    )
    _, tangent = jax.jvp(
        operation,
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    _, pullback = jax.vjp(operation, LOG_VALUES, LOG_X)
    values_cotangent, input_cotangent = pullback(OUTPUT_COTANGENT)

    _, expected_tangent = jax.jvp(
        _reference_logsumexp,
        (LOG_VALUES, LOG_X),
        (VALUES_TANGENT, INPUT_TANGENT),
    )
    _, expected_pullback = jax.vjp(
        _reference_logsumexp, LOG_VALUES, LOG_X
    )
    expected_values, expected_input = expected_pullback(OUTPUT_COTANGENT)
    np.testing.assert_allclose(
        tangent, expected_tangent, rtol=4e-6, atol=5e-7
    )
    np.testing.assert_allclose(values_cotangent, expected_values, rtol=4e-6)
    np.testing.assert_allclose(input_cotangent, expected_input, rtol=4e-6)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_weighted_infinity_and_empty_row_conventions(
    input_vjp_strategy: str,
) -> None:
    row_offsets = jnp.array([0, 3, 5, 5], dtype=jnp.int32)
    column_indices = jnp.array([0, 1, 2, 2, 3], dtype=jnp.int32)
    structure = springmv.CsrStructure.from_arrays(
        row_offsets, column_indices, num_cols=4
    )
    log_x = jnp.array(
        [jnp.inf, jnp.inf, -jnp.inf, -jnp.inf], dtype=jnp.float32
    )
    log_values = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    values_tangent = jnp.array(
        [0.0, 1.0, 2.0, -3.0, 4.0], dtype=jnp.float32
    )
    input_tangent = jnp.array([1.0, 3.0, 5.0, 7.0], dtype=jnp.float32)
    cotangent = jnp.array([2.0, -3.0, 7.0], dtype=jnp.float32)

    operation = lambda values, x: springmv.csr_logsumexp(
        structure,
        x,
        log_values=values,
        input_vjp_strategy=input_vjp_strategy,
    )
    output, output_tangent = jax.jvp(
        operation,
        (log_values, log_x),
        (values_tangent, input_tangent),
    )
    np.testing.assert_array_equal(output, jnp.array([jnp.inf, -jnp.inf, -jnp.inf]))
    np.testing.assert_allclose(output_tangent, [2.5, 6.5, 0.0])

    _, pullback = jax.vjp(operation, log_values, log_x)
    values_cotangent, input_cotangent = pullback(cotangent)
    np.testing.assert_allclose(
        values_cotangent, [1.0, 1.0, 0.0, -1.5, -1.5]
    )
    np.testing.assert_allclose(input_cotangent, [1.0, 1.0, -1.5, -1.5])


def test_unweighted_infinity_and_empty_row_conventions() -> None:
    structure = springmv.CsrStructure.from_arrays(
        jnp.array([0, 2, 4, 4], dtype=jnp.int32),
        jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        num_cols=4,
    )
    log_x = jnp.array(
        [jnp.inf, jnp.inf, -jnp.inf, -jnp.inf], dtype=jnp.float32
    )
    input_tangent = jnp.array([1.0, 3.0, 5.0, 7.0], dtype=jnp.float32)
    output, output_tangent = jax.jvp(
        lambda x: springmv.csr_logsumexp(structure, x),
        (log_x,),
        (input_tangent,),
    )
    np.testing.assert_array_equal(output, jnp.array([jnp.inf, -jnp.inf, -jnp.inf]))
    np.testing.assert_allclose(output_tangent, [2.0, 6.0, 0.0])

    _, pullback = jax.vjp(
        lambda x: springmv.csr_logsumexp(structure, x), log_x
    )
    (input_cotangent,) = pullback(
        jnp.array([2.0, -3.0, 7.0], dtype=jnp.float32)
    )
    np.testing.assert_allclose(input_cotangent, [1.0, 1.0, -1.5, -1.5])


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_nan_row_propagates_nan_derivatives(
    input_vjp_strategy: str,
) -> None:
    structure = springmv.CsrStructure.from_arrays(
        jnp.array([0, 2], dtype=jnp.int32),
        jnp.array([0, 1], dtype=jnp.int32),
        num_cols=2,
    )
    log_values = jnp.array([jnp.nan, 0.0], dtype=jnp.float32)
    log_x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    operation = lambda values, x: springmv.csr_logsumexp(
        structure,
        x,
        log_values=values,
        input_vjp_strategy=input_vjp_strategy,
    )
    output, output_tangent = jax.jvp(
        operation,
        (log_values, log_x),
        (jnp.ones_like(log_values), jnp.ones_like(log_x)),
    )
    assert bool(jnp.isnan(output[0]))
    assert bool(jnp.isnan(output_tangent[0]))

    _, pullback = jax.vjp(operation, log_values, log_x)
    values_cotangent, input_cotangent = pullback(
        jnp.array([1.0], dtype=jnp.float32)
    )
    assert bool(jnp.all(jnp.isnan(values_cotangent)))
    assert bool(jnp.all(jnp.isnan(input_cotangent)))


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_empty_structure_autodiff(input_vjp_strategy: str) -> None:
    structure = springmv.CsrStructure.from_arrays(
        jnp.array([0, 0, 0], dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32),
        num_cols=3,
    )
    log_x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    log_values = jnp.array([], dtype=jnp.float32)
    output, tangent = jax.jvp(
        lambda values, x: springmv.csr_logsumexp(
            structure,
            x,
            log_values=values,
            input_vjp_strategy=input_vjp_strategy,
        ),
        (log_values, log_x),
        (jnp.zeros_like(log_values), jnp.ones_like(log_x)),
    )
    np.testing.assert_array_equal(output, jnp.array([-jnp.inf, -jnp.inf]))
    np.testing.assert_array_equal(tangent, jnp.zeros(2, dtype=jnp.float32))

    _, pullback = jax.vjp(
        lambda values, x: springmv.csr_logsumexp(
            structure,
            x,
            log_values=values,
            input_vjp_strategy=input_vjp_strategy,
        ),
        log_values,
        log_x,
    )
    values_cotangent, input_cotangent = pullback(
        jnp.array([2.0, -1.0], dtype=jnp.float32)
    )
    assert values_cotangent.shape == (0,)
    np.testing.assert_array_equal(input_cotangent, jnp.zeros_like(log_x))


def test_vmap_jvp_matches_pure_jax(
    structure: springmv.CsrStructure,
) -> None:
    values = jnp.stack((LOG_VALUES, LOG_VALUES + 0.2))
    inputs = jnp.stack((LOG_X, LOG_X - 0.3))
    values_tangent = jnp.stack((VALUES_TANGENT, VALUES_TANGENT * 0.5))
    input_tangent = jnp.stack((INPUT_TANGENT, INPUT_TANGENT * -0.25))

    def spring_jvp(current_values, current_x, values_dot, x_dot):
        return jax.jvp(
            lambda v, x: springmv.csr_logsumexp(
                structure, x, log_values=v
            ),
            (current_values, current_x),
            (values_dot, x_dot),
        )

    actual = jax.jit(jax.vmap(spring_jvp))(
        values, inputs, values_tangent, input_tangent
    )
    expected = jax.vmap(
        lambda current_values, current_x, values_dot, x_dot: jax.jvp(
            _reference_logsumexp,
            (current_values, current_x),
            (values_dot, x_dot),
        )
    )(values, inputs, values_tangent, input_tangent)
    np.testing.assert_allclose(actual[0], expected[0], rtol=4e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=5e-6, atol=5e-7)


def test_unweighted_vmap_jvp_matches_pure_jax(
    structure: springmv.CsrStructure,
) -> None:
    inputs = jnp.stack((LOG_X, LOG_X - 0.3, LOG_X + 0.1))
    input_tangents = jnp.stack(
        (INPUT_TANGENT, INPUT_TANGENT * -0.25, INPUT_TANGENT * 0.5)
    )

    def spring_jvp(current_x, x_dot):
        return jax.jvp(
            lambda x: springmv.csr_logsumexp(structure, x),
            (current_x,),
            (x_dot,),
        )

    actual = jax.jit(jax.vmap(spring_jvp))(inputs, input_tangents)
    expected = jax.vmap(
        lambda current_x, x_dot: jax.jvp(
            lambda x: _reference_logsumexp(None, x),
            (current_x,),
            (x_dot,),
        )
    )(inputs, input_tangents)
    np.testing.assert_allclose(actual[0], expected[0], rtol=4e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_unweighted_grad_through_vmap_matches_pure_jax(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    inputs = jnp.stack((LOG_X, LOG_X + 0.25, LOG_X - 0.5))

    def spring_loss(log_inputs):
        outputs = jax.vmap(
            lambda log_x: springmv.csr_logsumexp(
                structure,
                log_x,
                input_vjp_strategy=input_vjp_strategy,
            )
        )(log_inputs)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    def reference_loss(log_inputs):
        outputs = jax.vmap(
            lambda log_x: _reference_logsumexp(None, log_x)
        )(log_inputs)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    actual = jax.jit(jax.grad(spring_loss))(inputs)
    expected = jax.grad(reference_loss)(inputs)
    np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=1e-6)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_vmap_of_grad_matches_pure_jax(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    inputs = jnp.stack((LOG_X, LOG_X + 0.25, LOG_X - 0.5))

    def spring_loss(log_x):
        return jnp.vdot(
            springmv.csr_logsumexp(
                structure,
                log_x,
                log_values=LOG_VALUES,
                input_vjp_strategy=input_vjp_strategy,
            ),
            OUTPUT_COTANGENT,
        )

    def reference_loss(log_x):
        return jnp.vdot(
            _reference_logsumexp(LOG_VALUES, log_x), OUTPUT_COTANGENT
        )

    actual = jax.jit(jax.vmap(jax.grad(spring_loss)))(inputs)
    expected = jax.vmap(jax.grad(reference_loss))(inputs)
    np.testing.assert_allclose(actual, expected, rtol=6e-6, atol=1e-6)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_grad_through_vmap_with_shared_log_values(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    inputs = jnp.stack((LOG_X, LOG_X + 0.25, LOG_X - 0.5))

    def spring_loss(log_values, log_inputs):
        outputs = jax.vmap(
            lambda log_x: springmv.csr_logsumexp(
                structure,
                log_x,
                log_values=log_values,
                input_vjp_strategy=input_vjp_strategy,
            )
        )(log_inputs)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    def reference_loss(log_values, log_inputs):
        outputs = jax.vmap(
            lambda log_x: _reference_logsumexp(log_values, log_x)
        )(log_inputs)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    actual = jax.jit(jax.grad(spring_loss, argnums=(0, 1)))(
        LOG_VALUES, inputs
    )
    expected = jax.grad(reference_loss, argnums=(0, 1))(LOG_VALUES, inputs)
    np.testing.assert_allclose(actual[0], expected[0], rtol=8e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=8e-6, atol=1e-6)


@pytest.mark.parametrize("input_vjp_strategy", ["transpose", "scatter"])
def test_vmap_grad_with_batched_values_and_inputs(
    structure: springmv.CsrStructure, input_vjp_strategy: str
) -> None:
    values = jnp.stack((LOG_VALUES, LOG_VALUES + 0.2))
    inputs = jnp.stack((LOG_X, LOG_X - 0.3))

    def spring_loss(current_values, current_x):
        return jnp.vdot(
            springmv.csr_logsumexp(
                structure,
                current_x,
                log_values=current_values,
                input_vjp_strategy=input_vjp_strategy,
            ),
            OUTPUT_COTANGENT,
        )

    def reference_loss(current_values, current_x):
        return jnp.vdot(
            _reference_logsumexp(current_values, current_x),
            OUTPUT_COTANGENT,
        )

    actual = jax.jit(
        jax.vmap(jax.grad(spring_loss, argnums=(0, 1)))
    )(values, inputs)
    expected = jax.vmap(jax.grad(reference_loss, argnums=(0, 1)))(
        values, inputs
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=8e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=8e-6, atol=1e-6)


def test_grad_through_vmap_with_batched_values_and_shared_input(
    structure: springmv.CsrStructure,
) -> None:
    values = jnp.stack((LOG_VALUES, LOG_VALUES + 0.2))

    def spring_loss(current_x):
        outputs = jax.vmap(
            lambda current_values: springmv.csr_logsumexp(
                structure,
                current_x,
                log_values=current_values,
                input_vjp_strategy="scatter",
            )
        )(values)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    def reference_loss(current_x):
        outputs = jax.vmap(
            lambda current_values: _reference_logsumexp(
                current_values, current_x
            )
        )(values)
        return jnp.sum(outputs * OUTPUT_COTANGENT)

    actual = jax.jit(jax.grad(spring_loss))(LOG_X)
    expected = jax.grad(reference_loss)(LOG_X)
    np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=1e-6)


def test_invalid_input_vjp_strategy(
    structure: springmv.CsrStructure,
) -> None:
    with pytest.raises(ValueError, match="input_vjp_strategy"):
        springmv.csr_logsumexp(
            structure,
            LOG_X,
            log_values=LOG_VALUES,
            input_vjp_strategy="invalid",
        )
