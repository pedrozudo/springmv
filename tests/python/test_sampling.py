# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import springmv


ROW_OFFSETS = np.array([0, 3, 3, 5, 7, 10], dtype=np.int32)
COLUMN_INDICES = np.array([0, 2, 2, 1, 3, 0, 3, 1, 2, 3], dtype=np.int32)
INF = np.float32(np.inf)
NAN = np.float32(np.nan)
LOG_WEIGHTS = np.array(
    [0.0, 1.0, 1.0, 0.0, 0.0, INF, INF, NAN, -INF, -INF],
    dtype=np.float32,
)
GUMBELS = np.array(
    [0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
ACTIVE_PARENT = np.array([True, True, True, True, False])
EXPECTED = np.array([True, False, True, True])


@pytest.fixture(scope="module")
def structure() -> springmv.CsrStructure:
    return springmv.CsrStructure.from_arrays(
        ROW_OFFSETS, COLUMN_INDICES, num_cols=4
    )


def _reference(
    log_weights: np.ndarray,
    gumbels: np.ndarray,
    active_parent: np.ndarray,
) -> np.ndarray:
    selected = np.full(ROW_OFFSETS.size - 1, -1, dtype=np.int32)
    for row in range(selected.size):
        best_edge = -1
        best_score = -np.inf
        for edge in range(ROW_OFFSETS[row], ROW_OFFSETS[row + 1]):
            score = np.float32(log_weights[edge] + gumbels[edge])
            if np.isnan(score):
                continue
            if (
                best_edge < 0
                or score > best_score
                or (score == best_score and edge < best_edge)
            ):
                best_edge = edge
                best_score = score
        if best_edge >= 0:
            selected[row] = COLUMN_INDICES[best_edge]

    output = np.zeros(4, dtype=np.bool_)
    for row, active in enumerate(active_parent):
        if active and selected[row] >= 0:
            output[selected[row]] = True
    return output


@pytest.mark.parametrize("subwarp_size", [0, 1, 2, 4, 8, 16, 32])
def test_matches_reference_eager_and_jit(
    structure: springmv.CsrStructure, subwarp_size: int
) -> None:
    expected = _reference(LOG_WEIGHTS, GUMBELS, ACTIVE_PARENT)
    actual = springmv.csr_logsumexp_sample(
        structure,
        LOG_WEIGHTS,
        GUMBELS,
        ACTIVE_PARENT,
        subwarp_size=subwarp_size,
    )
    compiled = jax.jit(
        lambda weights, noise, active: springmv.csr_logsumexp_sample(
            structure,
            weights,
            noise,
            active,
            subwarp_size=subwarp_size,
        )
    )(LOG_WEIGHTS, GUMBELS, ACTIVE_PARENT)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(compiled, expected)
    np.testing.assert_array_equal(actual, EXPECTED)


def test_dynamic_structure_under_jit(
    structure: springmv.CsrStructure,
) -> None:
    compiled = jax.jit(
        lambda current_structure: springmv.csr_logsumexp_sample(
            current_structure, LOG_WEIGHTS, GUMBELS, ACTIVE_PARENT
        )
    )
    np.testing.assert_array_equal(compiled(structure), EXPECTED)


@pytest.mark.parametrize(
    ("weights_batched", "gumbels_batched", "active_batched"),
    [case for case in itertools.product((False, True), repeat=3) if any(case)],
)
def test_vmap_shared_and_batched_operands(
    structure: springmv.CsrStructure,
    weights_batched: bool,
    gumbels_batched: bool,
    active_batched: bool,
) -> None:
    second_weights = np.array(
        [5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, NAN, NAN, NAN],
        dtype=np.float32,
    )
    second_gumbels = np.zeros_like(GUMBELS)
    second_active = np.array([True, False, True, True, True])
    weights = (
        jnp.asarray(np.stack((LOG_WEIGHTS, second_weights)))
        if weights_batched
        else jnp.asarray(LOG_WEIGHTS)
    )
    gumbels = (
        jnp.asarray(np.stack((GUMBELS, second_gumbels)))
        if gumbels_batched
        else jnp.asarray(GUMBELS)
    )
    active = (
        jnp.asarray(np.stack((ACTIVE_PARENT, second_active)))
        if active_batched
        else jnp.asarray(ACTIVE_PARENT)
    )
    in_axes = tuple(
        0 if batched else None
        for batched in (weights_batched, gumbels_batched, active_batched)
    )
    actual = jax.jit(
        jax.vmap(
            lambda current_weights, current_gumbels, current_active: (
                springmv.csr_logsumexp_sample(
                    structure,
                    current_weights,
                    current_gumbels,
                    current_active,
                )
            ),
            in_axes=in_axes,
        )
    )(weights, gumbels, active)

    expected = []
    for batch in range(2):
        expected.append(
            _reference(
                np.asarray(weights[batch] if weights_batched else weights),
                np.asarray(gumbels[batch] if gumbels_batched else gumbels),
                np.asarray(active[batch] if active_batched else active),
            )
        )
    np.testing.assert_array_equal(actual, np.stack(expected))


def test_vmap_nonleading_axes(structure: springmv.CsrStructure) -> None:
    weights = jnp.stack((jnp.asarray(LOG_WEIGHTS), jnp.asarray(LOG_WEIGHTS)), 1)
    gumbels = jnp.stack((jnp.asarray(GUMBELS), jnp.asarray(GUMBELS)), 1)
    active = jnp.stack(
        (jnp.asarray(ACTIVE_PARENT), jnp.asarray(ACTIVE_PARENT)), 1
    )
    actual = jax.vmap(
        lambda current_weights, current_gumbels, current_active: (
            springmv.csr_logsumexp_sample(
                structure, current_weights, current_gumbels, current_active
            )
        ),
        in_axes=(1, 1, 1),
        out_axes=1,
    )(weights, gumbels, active)
    np.testing.assert_array_equal(
        actual, np.stack((EXPECTED, EXPECTED), axis=1)
    )


def test_vmap_empty_batch(structure: springmv.CsrStructure) -> None:
    weights = jnp.empty((0, structure.num_nonzeros), dtype=jnp.float32)
    gumbels = jnp.empty((0, structure.num_nonzeros), dtype=jnp.float32)
    active = jnp.empty((0, structure.num_rows), dtype=jnp.bool_)
    actual = jax.jit(
        jax.vmap(
            lambda current_weights, current_gumbels, current_active: (
                springmv.csr_logsumexp_sample(
                    structure,
                    current_weights,
                    current_gumbels,
                    current_active,
                )
            )
        )
    )(weights, gumbels, active)
    assert actual.shape == (0, structure.num_cols)
    assert actual.dtype == jnp.bool_


@pytest.mark.parametrize(("num_rows", "num_cols"), [(0, 3), (2, 0)])
def test_empty_topology(num_rows: int, num_cols: int) -> None:
    structure = springmv.CsrStructure.from_arrays(
        np.zeros(num_rows + 1, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        num_cols=num_cols,
    )
    actual = jax.jit(springmv.csr_logsumexp_sample)(
        structure,
        jnp.empty(0, dtype=jnp.float32),
        jnp.empty(0, dtype=jnp.float32),
        jnp.ones(num_rows, dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(actual, np.zeros(num_cols, dtype=np.bool_))


def test_empirical_distribution_matches_softmax() -> None:
    structure = springmv.CsrStructure.from_arrays(
        np.array([0, 3], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        num_cols=3,
    )
    probabilities = np.array([0.1, 0.3, 0.6], dtype=np.float32)
    log_weights = jnp.log(jnp.asarray(probabilities))
    batch_size = 20_000
    gumbels = jax.random.gumbel(
        jax.random.key(17), (batch_size, 3), dtype=jnp.float32
    )
    samples = jax.jit(
        jax.vmap(
            lambda noise: springmv.csr_logsumexp_sample(
                structure,
                log_weights,
                noise,
                jnp.array([True]),
            )
        )
    )(gumbels)
    frequencies = np.asarray(jnp.mean(samples, axis=0))
    np.testing.assert_allclose(frequencies, probabilities, atol=0.015)
    np.testing.assert_array_equal(np.asarray(jnp.sum(samples, axis=1)), 1)


def test_cannot_be_differentiated(
    structure: springmv.CsrStructure,
) -> None:
    def function(weights):
        return springmv.csr_logsumexp_sample(
            structure, weights, GUMBELS, ACTIVE_PARENT
        ).astype(jnp.float32)

    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.jvp(
            function,
            (jnp.asarray(LOG_WEIGHTS),),
            (jnp.ones_like(jnp.asarray(LOG_WEIGHTS)),),
        )


@pytest.mark.parametrize(
    ("argument", "value", "exception", "message"),
    [
        ("log_weights", np.zeros(9, np.float32), ValueError, "equal shapes"),
        ("gumbels", np.zeros(9, np.float32), ValueError, "equal shapes"),
        (
            "active_parent",
            np.ones(4, np.bool_),
            ValueError,
            "structure.num_rows",
        ),
        ("log_weights", np.zeros(10, np.float64), TypeError, "float32"),
        ("gumbels", np.zeros(10, np.float64), TypeError, "float32"),
        ("active_parent", np.ones(5, np.int32), TypeError, "bool"),
    ],
)
def test_validation(
    structure: springmv.CsrStructure,
    argument: str,
    value: np.ndarray,
    exception: type[Exception],
    message: str,
) -> None:
    arguments = {
        "log_weights": LOG_WEIGHTS,
        "gumbels": GUMBELS,
        "active_parent": ACTIVE_PARENT,
    }
    arguments[argument] = value
    with pytest.raises(exception, match=message):
        springmv.csr_logsumexp_sample(structure, **arguments)
