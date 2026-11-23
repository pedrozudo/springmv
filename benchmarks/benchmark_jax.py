# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

import springmv


def _sample(operation: Callable[[], object]) -> float:
    start = time.perf_counter()
    jax.block_until_ready(operation())
    return (time.perf_counter() - start) * 1.0e3


def _time_pair(
    springmv_operation: Callable[[], object],
    jax_operation: Callable[[], object],
    warmup: int,
    iterations: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    for _ in range(warmup):
        jax.block_until_ready(springmv_operation())
        jax.block_until_ready(jax_operation())

    springmv_samples = []
    jax_samples = []
    for iteration in range(iterations):
        if iteration % 2 == 0:
            springmv_samples.append(_sample(springmv_operation))
            jax_samples.append(_sample(jax_operation))
        else:
            jax_samples.append(_sample(jax_operation))
            springmv_samples.append(_sample(springmv_operation))

    def summarize(samples: list[float]) -> tuple[float, float, float]:
        return statistics.median(samples), min(samples), max(samples)

    return summarize(springmv_samples), summarize(jax_samples)


def _report_comparison(
    name: str,
    springmv_samples: tuple[float, float, float],
    jax_samples: tuple[float, float, float],
    *,
    springmv_label: str = "SpringMV",
    jax_label: str = "JAX",
) -> None:
    springmv_median, springmv_minimum, springmv_maximum = springmv_samples
    jax_median, jax_minimum, jax_maximum = jax_samples
    speedup = jax_median / springmv_median
    print(
        f"{name:65s} "
        f"{springmv_label} {springmv_median:9.4f} "
        f"[{springmv_minimum:9.4f}, {springmv_maximum:9.4f}] ms  "
        f"{jax_label} {jax_median:9.4f} "
        f"[{jax_minimum:9.4f}, {jax_maximum:9.4f}] ms  "
        f"{jax_label}/{springmv_label} {speedup:7.2f}x"
    )


def _benchmark_pair(
    name: str,
    springmv_operation: Callable[[], object],
    jax_operation: Callable[[], object],
    warmup: int,
    iterations: int,
    *,
    springmv_label: str = "SpringMV",
    jax_label: str = "JAX",
) -> None:
    springmv_samples, jax_samples = _time_pair(
        springmv_operation, jax_operation, warmup, iterations
    )
    _report_comparison(
        name,
        springmv_samples,
        jax_samples,
        springmv_label=springmv_label,
        jax_label=jax_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--cols", type=int, default=1 << 20)
    parser.add_argument("--nnz-per-row", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--column-pattern",
        choices=("uniform", "diagonal", "hotspot", "single"),
        default="uniform",
    )
    parser.add_argument(
        "--subwarp-size", type=int, choices=(0, 1, 2, 4, 8, 16, 32), default=0
    )
    parser.add_argument(
        "--diagnose-input-vjp",
        action="store_true",
        help=(
            "compare scatter and transpose input-VJP strategies, including "
            "value-reordering costs"
        ),
    )
    args = parser.parse_args()

    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.cols <= 0:
        parser.error("--cols must be positive")
    if args.nnz_per_row <= 0:
        parser.error("--nnz-per-row must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup <= 0:
        parser.error("--warmup must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    rng = np.random.default_rng(args.seed)
    nnz = args.rows * args.nnz_per_row
    row_offsets = np.arange(
        0, nnz + 1, args.nnz_per_row, dtype=np.int32
    )
    if args.column_pattern == "uniform":
        column_indices = rng.integers(
            0, args.cols, size=nnz, dtype=np.int32
        )
    elif args.column_pattern == "diagonal":
        column_indices = np.repeat(
            np.arange(args.rows, dtype=np.int32) % args.cols,
            args.nnz_per_row,
        )
    elif args.column_pattern == "hotspot":
        hot_columns = max(1, args.cols // 100)
        column_indices = rng.integers(
            0, hot_columns, size=nnz, dtype=np.int32
        )
    else:
        column_indices = np.zeros(nnz, dtype=np.int32)
    values = jnp.asarray(rng.normal(size=nnz).astype(np.float32))
    x = jnp.asarray(rng.normal(size=args.cols).astype(np.float32))
    output_cotangent = jnp.asarray(
        rng.normal(size=args.rows).astype(np.float32)
    )
    batch_scale = jnp.linspace(
        0.75, 1.25, args.batch_size, dtype=jnp.float32
    )
    x_batch = x[None, :] + batch_scale[:, None]
    output_cotangent_batch = output_cotangent[None, :] * batch_scale[:, None]
    structure = springmv.CsrStructure.from_arrays(
        row_offsets, column_indices, num_cols=args.cols
    )
    edge_rows = jnp.asarray(
        np.repeat(
            np.arange(args.rows, dtype=np.int32), args.nnz_per_row
        )
    )
    unit_values = jnp.ones(nnz, dtype=jnp.float32)

    if args.diagnose_input_vjp:
        transposed_structure = structure.transpose()
        transposed_values = jax.block_until_ready(
            values[structure.transpose_permutation]
        )

    def jax_csr_matvec(current_values, current_x):
        matrix = jsparse.CSR(
            (
                current_values,
                structure.column_indices,
                structure.row_offsets,
            ),
            shape=(args.rows, args.cols),
        )
        return jsparse.csr_matvec(matrix, current_x)

    def jax_segment_spmv(current_values, current_x):
        return jax.ops.segment_sum(
            current_values * current_x[structure.column_indices],
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )

    def jax_segment_spmv_unweighted(current_x):
        return jax.ops.segment_sum(
            current_x[structure.column_indices],
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )

    def jax_segment_logsumexp(current_values, current_x):
        terms = current_values + current_x[structure.column_indices]
        row_max = jax.ops.segment_max(
            terms,
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )
        scaled_sum = jax.ops.segment_sum(
            jnp.exp(terms - row_max[edge_rows]),
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )
        return row_max + jnp.log(scaled_sum)

    def jax_segment_logsumexp_unweighted(current_x):
        terms = current_x[structure.column_indices]
        row_max = jax.ops.segment_max(
            terms,
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )
        scaled_sum = jax.ops.segment_sum(
            jnp.exp(terms - row_max[edge_rows]),
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )
        return row_max + jnp.log(scaled_sum)

    def squared_output_loss(operation, *primals):
        output = operation(*primals)
        return 0.5 * jnp.sum(output * output)

    def springmv_arithmetic_unweighted(current_x):
        return springmv.csr_spmv(
            structure,
            current_x,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="scatter",
        )

    def springmv_arithmetic_unweighted_transpose(current_x):
        return springmv.csr_spmv(
            structure,
            current_x,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="transpose",
        )

    def springmv_arithmetic_weighted(current_values, current_x):
        return springmv.csr_spmv(
            structure,
            current_x,
            values=current_values,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="scatter",
        )

    def springmv_arithmetic_weighted_transpose(current_values, current_x):
        return springmv.csr_spmv(
            structure,
            current_x,
            values=current_values,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="transpose",
        )

    def springmv_logsumexp_unweighted(current_x):
        return springmv.csr_logsumexp(
            structure,
            current_x,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="scatter",
        )

    def springmv_logsumexp_unweighted_transpose(current_x):
        return springmv.csr_logsumexp(
            structure,
            current_x,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="transpose",
        )

    def springmv_logsumexp_weighted(current_values, current_x):
        return springmv.csr_logsumexp(
            structure,
            current_x,
            log_values=current_values,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="scatter",
        )

    def springmv_logsumexp_weighted_transpose(current_values, current_x):
        return springmv.csr_logsumexp(
            structure,
            current_x,
            log_values=current_values,
            subwarp_size=args.subwarp_size,
            input_vjp_strategy="transpose",
        )

    jax_arithmetic_unweighted = lambda current_x: jax_csr_matvec(
        unit_values, current_x
    )
    jax_arithmetic_weighted = jax_csr_matvec

    arithmetic_unweighted_forward = jax.jit(springmv_arithmetic_unweighted)
    jax_arithmetic_unweighted_forward = jax.jit(jax_arithmetic_unweighted)
    arithmetic_unweighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_arithmetic_unweighted, current_x
            )
        )
    )
    arithmetic_unweighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_arithmetic_unweighted_transpose, current_x
            )
        )
    )
    jax_arithmetic_unweighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                jax_arithmetic_unweighted, current_x
            )
        )
    )
    logsumexp_unweighted_forward = jax.jit(springmv_logsumexp_unweighted)
    jax_logsumexp_unweighted_forward = jax.jit(
        jax_segment_logsumexp_unweighted
    )
    logsumexp_unweighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_logsumexp_unweighted, current_x
            )
        )
    )
    logsumexp_unweighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_logsumexp_unweighted_transpose, current_x
            )
        )
    )
    jax_logsumexp_unweighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                jax_segment_logsumexp_unweighted, current_x
            )
        )
    )

    arithmetic_weighted_forward = jax.jit(springmv_arithmetic_weighted)
    jax_arithmetic_weighted_forward = jax.jit(jax_arithmetic_weighted)
    arithmetic_weighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_arithmetic_weighted, current_values, current_x
            ),
            argnums=(0, 1),
        )
    )
    arithmetic_weighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_arithmetic_weighted_transpose,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    jax_arithmetic_weighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                jax_arithmetic_weighted, current_values, current_x
            ),
            argnums=(0, 1),
        )
    )
    logsumexp_weighted_forward = jax.jit(springmv_logsumexp_weighted)
    jax_logsumexp_weighted_forward = jax.jit(jax_segment_logsumexp)
    logsumexp_weighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_logsumexp_weighted, current_values, current_x
            ),
            argnums=(0, 1),
        )
    )
    logsumexp_weighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_logsumexp_weighted_transpose,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    jax_logsumexp_weighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                jax_segment_logsumexp, current_values, current_x
            ),
            argnums=(0, 1),
        )
    )

    springmv_batched_arithmetic_unweighted = jax.vmap(
        springmv_arithmetic_unweighted
    )
    springmv_batched_arithmetic_unweighted_transpose = jax.vmap(
        springmv_arithmetic_unweighted_transpose
    )
    jax_batched_arithmetic_unweighted = jax.vmap(
        jax_segment_spmv_unweighted
    )
    springmv_batched_logsumexp_unweighted = jax.vmap(
        springmv_logsumexp_unweighted
    )
    springmv_batched_logsumexp_unweighted_transpose = jax.vmap(
        springmv_logsumexp_unweighted_transpose
    )
    jax_batched_logsumexp_unweighted = jax.vmap(
        jax_segment_logsumexp_unweighted
    )

    batched_arithmetic_unweighted_forward = jax.jit(
        springmv_batched_arithmetic_unweighted
    )
    batched_jax_arithmetic_unweighted_forward = jax.jit(
        jax_batched_arithmetic_unweighted
    )
    batched_arithmetic_unweighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_batched_arithmetic_unweighted, current_x
            )
        )
    )
    batched_arithmetic_unweighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_batched_arithmetic_unweighted_transpose,
                current_x,
            )
        )
    )
    batched_jax_arithmetic_unweighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                jax_batched_arithmetic_unweighted, current_x
            )
        )
    )
    batched_logsumexp_unweighted_forward = jax.jit(
        springmv_batched_logsumexp_unweighted
    )
    batched_jax_logsumexp_unweighted_forward = jax.jit(
        jax_batched_logsumexp_unweighted
    )
    batched_logsumexp_unweighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_batched_logsumexp_unweighted, current_x
            )
        )
    )
    batched_logsumexp_unweighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                springmv_batched_logsumexp_unweighted_transpose,
                current_x,
            )
        )
    )
    batched_jax_logsumexp_unweighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_x: squared_output_loss(
                jax_batched_logsumexp_unweighted, current_x
            )
        )
    )

    def springmv_batched_arithmetic_weighted(current_values, current_x):
        return jax.vmap(
            lambda batch_x: springmv_arithmetic_weighted(
                current_values, batch_x
            )
        )(current_x)

    def springmv_batched_arithmetic_weighted_transpose(
        current_values, current_x
    ):
        return jax.vmap(
            lambda batch_x: springmv_arithmetic_weighted_transpose(
                current_values, batch_x
            )
        )(current_x)

    def jax_batched_arithmetic_weighted(current_values, current_x):
        return jax.vmap(
            lambda batch_x: jax_segment_spmv(current_values, batch_x)
        )(current_x)

    def springmv_batched_logsumexp_weighted(current_values, current_x):
        return jax.vmap(
            lambda batch_x: springmv_logsumexp_weighted(
                current_values, batch_x
            )
        )(current_x)

    def springmv_batched_logsumexp_weighted_transpose(
        current_values, current_x
    ):
        return jax.vmap(
            lambda batch_x: springmv_logsumexp_weighted_transpose(
                current_values, batch_x
            )
        )(current_x)

    def jax_batched_logsumexp_weighted(current_values, current_x):
        return jax.vmap(
            lambda batch_x: jax_segment_logsumexp(
                current_values, batch_x
            )
        )(current_x)

    batched_arithmetic_weighted_forward = jax.jit(
        springmv_batched_arithmetic_weighted
    )
    batched_jax_arithmetic_weighted_forward = jax.jit(
        jax_batched_arithmetic_weighted
    )
    batched_arithmetic_weighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_batched_arithmetic_weighted,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    batched_arithmetic_weighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_batched_arithmetic_weighted_transpose,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    batched_jax_arithmetic_weighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                jax_batched_arithmetic_weighted,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    batched_logsumexp_weighted_forward = jax.jit(
        springmv_batched_logsumexp_weighted
    )
    batched_jax_logsumexp_weighted_forward = jax.jit(
        jax_batched_logsumexp_weighted
    )
    batched_logsumexp_weighted_forward_backward_scatter = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_batched_logsumexp_weighted,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    batched_logsumexp_weighted_forward_backward_transpose = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                springmv_batched_logsumexp_weighted_transpose,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )
    batched_jax_logsumexp_weighted_forward_backward = jax.jit(
        jax.grad(
            lambda current_values, current_x: squared_output_loss(
                jax_batched_logsumexp_weighted,
                current_values,
                current_x,
            ),
            argnums=(0, 1),
        )
    )

    if args.diagnose_input_vjp:
        weighted_input_vjp = jax.jit(
            lambda current_values, cotangent: jax.vjp(
                lambda current_x: springmv.csr_spmv(
                    structure,
                    current_x,
                    values=current_values,
                    subwarp_size=args.subwarp_size,
                    input_vjp_strategy="transpose",
                ),
                x,
            )[1](cotangent)[0]
        )
        weighted_input_vjp_scatter = jax.jit(
            lambda current_values, cotangent: jax.vjp(
                lambda current_x: springmv_arithmetic_weighted(
                    current_values, current_x
                ),
                x,
            )[1](cotangent)[0]
        )
        jax_weighted_input_vjp = jax.jit(
            lambda current_values, cotangent: jax.vjp(
                lambda current_x: jax_csr_matvec(
                    current_values, current_x
                ),
                x,
            )[1](cotangent)[0]
        )
        weighted_logsumexp_input_vjp = jax.jit(
            lambda current_values, cotangent: jax.vjp(
                lambda current_x: springmv.csr_logsumexp(
                    structure,
                    current_x,
                    log_values=current_values,
                    subwarp_size=args.subwarp_size,
                    input_vjp_strategy="transpose",
                ),
                x,
            )[1](cotangent)[0]
        )
        jax_weighted_logsumexp_input_vjp = jax.jit(
            lambda current_values, cotangent: jax.vjp(
                lambda current_x: jax_segment_logsumexp(
                    current_values, current_x
                ),
                x,
            )[1](cotangent)[0]
        )

        def batched_arithmetic_input_vjp_transpose(current_x, cotangent):
            return jax.vmap(
                lambda batch_x, batch_cotangent: jax.vjp(
                    lambda differentiable_x: springmv.csr_spmv(
                        structure,
                        differentiable_x,
                        values=values,
                        subwarp_size=args.subwarp_size,
                        input_vjp_strategy="transpose",
                    ),
                    batch_x,
                )[1](batch_cotangent)[0]
            )(current_x, cotangent)

        batched_arithmetic_input_vjp_transpose = jax.jit(
            batched_arithmetic_input_vjp_transpose
        )
        batched_jax_arithmetic_input_vjp = jax.jit(
            jax.vmap(
                lambda batch_x, batch_cotangent: jax.vjp(
                    lambda differentiable_x: jax_segment_spmv(
                        values, differentiable_x
                    ),
                    batch_x,
                )[1](batch_cotangent)[0]
            )
        )

        def batched_logsumexp_input_vjp_transpose(current_x, cotangent):
            return jax.vmap(
                lambda batch_x, batch_cotangent: jax.vjp(
                    lambda differentiable_x: springmv.csr_logsumexp(
                        structure,
                        differentiable_x,
                        log_values=values,
                        subwarp_size=args.subwarp_size,
                        input_vjp_strategy="transpose",
                    ),
                    batch_x,
                )[1](batch_cotangent)[0]
            )(current_x, cotangent)

        batched_logsumexp_input_vjp_transpose = jax.jit(
            batched_logsumexp_input_vjp_transpose
        )
        batched_jax_logsumexp_input_vjp = jax.jit(
            jax.vmap(
                lambda batch_x, batch_cotangent: jax.vjp(
                    lambda differentiable_x: jax_segment_logsumexp(
                        values, differentiable_x
                    ),
                    batch_x,
                )[1](batch_cotangent)[0]
            )
        )
        reordered_input_vjp = jax.jit(
            lambda current_transposed_values, cotangent: springmv.csr_spmv(
                transposed_structure,
                cotangent,
                values=current_transposed_values,
                subwarp_size=args.subwarp_size,
            )
        )
        reorder_and_input_vjp = jax.jit(
            lambda current_values, cotangent: springmv.csr_spmv(
                transposed_structure,
                cotangent,
                values=current_values[structure.transpose_permutation],
                subwarp_size=args.subwarp_size,
            )
        )
        jax_atomic_input_vjp = jax.jit(
            lambda current_values, cotangent: (
                jnp.zeros(args.cols, dtype=jnp.float32)
                .at[structure.column_indices]
                .add(current_values * cotangent[edge_rows])
            )
        )

    print(
        f"rows={args.rows} cols={args.cols} nnz={nnz} "
        f"nnz/row={args.nnz_per_row} subwarp={args.subwarp_size} "
        f"transpose-max-nnz/row={structure.transpose_max_nnz_per_row} "
        f"columns={args.column_pattern} batch={args.batch_size}"
    )
    print(
        "times are median [minimum, maximum]; final ratio is "
        "right median / left median"
    )
    print("\nunweighted (backward differentiates the input)")
    _benchmark_pair(
        "single arithmetic forward",
        lambda: arithmetic_unweighted_forward(x),
        lambda: jax_arithmetic_unweighted_forward(x),
        args.warmup,
        args.iterations,
        jax_label="JAX CSR",
    )
    _benchmark_pair(
        "single arithmetic forward + backward (scatter)",
        lambda: arithmetic_unweighted_forward_backward_scatter(x),
        lambda: jax_arithmetic_unweighted_forward_backward(x),
        args.warmup,
        args.iterations,
        jax_label="JAX CSR",
    )
    _benchmark_pair(
        "single arithmetic forward + backward (transpose)",
        lambda: arithmetic_unweighted_forward_backward_transpose(x),
        lambda: jax_arithmetic_unweighted_forward_backward(x),
        args.warmup,
        args.iterations,
        jax_label="JAX CSR",
    )
    _benchmark_pair(
        "single log-sum-exp forward",
        lambda: logsumexp_unweighted_forward(x),
        lambda: jax_logsumexp_unweighted_forward(x),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        "single log-sum-exp forward + backward (scatter)",
        lambda: logsumexp_unweighted_forward_backward_scatter(x),
        lambda: jax_logsumexp_unweighted_forward_backward(x),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        "single log-sum-exp forward + backward (transpose)",
        lambda: logsumexp_unweighted_forward_backward_transpose(x),
        lambda: jax_logsumexp_unweighted_forward_backward(x),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched arithmetic forward (batch={args.batch_size})",
        lambda: batched_arithmetic_unweighted_forward(x_batch),
        lambda: batched_jax_arithmetic_unweighted_forward(x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched arithmetic forward + backward (scatter, batch={args.batch_size})",
        lambda: batched_arithmetic_unweighted_forward_backward_scatter(
            x_batch
        ),
        lambda: batched_jax_arithmetic_unweighted_forward_backward(x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched arithmetic forward + backward (transpose, batch={args.batch_size})",
        lambda: batched_arithmetic_unweighted_forward_backward_transpose(
            x_batch
        ),
        lambda: batched_jax_arithmetic_unweighted_forward_backward(x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched log-sum-exp forward (batch={args.batch_size})",
        lambda: batched_logsumexp_unweighted_forward(x_batch),
        lambda: batched_jax_logsumexp_unweighted_forward(x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched log-sum-exp forward + backward (scatter, batch={args.batch_size})",
        lambda: batched_logsumexp_unweighted_forward_backward_scatter(
            x_batch
        ),
        lambda: batched_jax_logsumexp_unweighted_forward_backward(x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched log-sum-exp forward + backward (transpose, batch={args.batch_size})",
        lambda: batched_logsumexp_unweighted_forward_backward_transpose(
            x_batch
        ),
        lambda: batched_jax_logsumexp_unweighted_forward_backward(x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )

    print(
        "\nweighted (backward differentiates edge values and input; "
        "batched rows share edge values)"
    )
    _benchmark_pair(
        "single arithmetic forward",
        lambda: arithmetic_weighted_forward(values, x),
        lambda: jax_arithmetic_weighted_forward(values, x),
        args.warmup,
        args.iterations,
        jax_label="JAX CSR",
    )
    _benchmark_pair(
        "single arithmetic forward + backward (scatter)",
        lambda: arithmetic_weighted_forward_backward_scatter(values, x),
        lambda: jax_arithmetic_weighted_forward_backward(values, x),
        args.warmup,
        args.iterations,
        jax_label="JAX CSR",
    )
    _benchmark_pair(
        "single arithmetic forward + backward (transpose)",
        lambda: arithmetic_weighted_forward_backward_transpose(values, x),
        lambda: jax_arithmetic_weighted_forward_backward(values, x),
        args.warmup,
        args.iterations,
        jax_label="JAX CSR",
    )
    _benchmark_pair(
        "single log-sum-exp forward",
        lambda: logsumexp_weighted_forward(values, x),
        lambda: jax_logsumexp_weighted_forward(values, x),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        "single log-sum-exp forward + backward (scatter)",
        lambda: logsumexp_weighted_forward_backward_scatter(values, x),
        lambda: jax_logsumexp_weighted_forward_backward(values, x),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        "single log-sum-exp forward + backward (transpose)",
        lambda: logsumexp_weighted_forward_backward_transpose(values, x),
        lambda: jax_logsumexp_weighted_forward_backward(values, x),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched arithmetic forward (batch={args.batch_size})",
        lambda: batched_arithmetic_weighted_forward(values, x_batch),
        lambda: batched_jax_arithmetic_weighted_forward(values, x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched arithmetic forward + backward (scatter, batch={args.batch_size})",
        lambda: batched_arithmetic_weighted_forward_backward_scatter(
            values, x_batch
        ),
        lambda: batched_jax_arithmetic_weighted_forward_backward(
            values, x_batch
        ),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched arithmetic forward + backward (transpose, batch={args.batch_size})",
        lambda: batched_arithmetic_weighted_forward_backward_transpose(
            values, x_batch
        ),
        lambda: batched_jax_arithmetic_weighted_forward_backward(
            values, x_batch
        ),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched log-sum-exp forward (batch={args.batch_size})",
        lambda: batched_logsumexp_weighted_forward(values, x_batch),
        lambda: batched_jax_logsumexp_weighted_forward(values, x_batch),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched log-sum-exp forward + backward (scatter, batch={args.batch_size})",
        lambda: batched_logsumexp_weighted_forward_backward_scatter(
            values, x_batch
        ),
        lambda: batched_jax_logsumexp_weighted_forward_backward(
            values, x_batch
        ),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )
    _benchmark_pair(
        f"batched log-sum-exp forward + backward (transpose, batch={args.batch_size})",
        lambda: batched_logsumexp_weighted_forward_backward_transpose(
            values, x_batch
        ),
        lambda: batched_jax_logsumexp_weighted_forward_backward(
            values, x_batch
        ),
        args.warmup,
        args.iterations,
        jax_label="JAX segment",
    )

    if args.diagnose_input_vjp:
        print("\nbackward-only input-VJP strategy diagnostics")
        _benchmark_pair(
            "log-sum-exp weighted VJP (transpose)",
            lambda: weighted_logsumexp_input_vjp(
                values, output_cotangent
            ),
            lambda: jax_weighted_logsumexp_input_vjp(
                values, output_cotangent
            ),
            args.warmup,
            args.iterations,
            jax_label="JAX segment",
        )
        _benchmark_pair(
            "batched arithmetic weighted VJP (transpose)",
            lambda: batched_arithmetic_input_vjp_transpose(
                x_batch, output_cotangent_batch
            ),
            lambda: batched_jax_arithmetic_input_vjp(
                x_batch, output_cotangent_batch
            ),
            args.warmup,
            args.iterations,
            jax_label="JAX segment",
        )
        _benchmark_pair(
            "batched log-sum-exp weighted VJP (transpose)",
            lambda: batched_logsumexp_input_vjp_transpose(
                x_batch, output_cotangent_batch
            ),
            lambda: batched_jax_logsumexp_input_vjp(
                x_batch, output_cotangent_batch
            ),
            args.warmup,
            args.iterations,
            jax_label="JAX segment",
        )

        print("\nweighted arithmetic implementation diagnostics")
        _benchmark_pair(
            "transpose with permutation gather",
            lambda: weighted_input_vjp(values, output_cotangent),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
            jax_label="JAX CSR",
        )
        _benchmark_pair(
            "atomic scatter",
            lambda: weighted_input_vjp_scatter(
                values, output_cotangent
            ),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
            jax_label="JAX CSR",
        )
        _benchmark_pair(
            "transpose with pre-reordered values",
            lambda: reordered_input_vjp(
                transposed_values, output_cotangent
            ),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
            jax_label="JAX CSR",
        )
        _benchmark_pair(
            "transpose with reorder included",
            lambda: reorder_and_input_vjp(values, output_cotangent),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
            jax_label="JAX CSR",
        )
        atomic_samples, csr_samples = _time_pair(
            lambda: jax_atomic_input_vjp(values, output_cotangent),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        )
        _report_comparison(
            "JAX input-VJP implementations",
            atomic_samples,
            csr_samples,
            springmv_label="JAX scatter",
            jax_label="JAX CSR",
        )


if __name__ == "__main__":
    main()
