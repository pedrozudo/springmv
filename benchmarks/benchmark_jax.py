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
        f"{name:33s} "
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
) -> None:
    springmv_samples, jax_samples = _time_pair(
        springmv_operation, jax_operation, warmup, iterations
    )
    _report_comparison(name, springmv_samples, jax_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--cols", type=int, default=1 << 20)
    parser.add_argument("--nnz-per-row", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
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
        help="compare permutation, reordered-value, and cuSPARSE VJP paths",
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

    weighted_forward = jax.jit(
        lambda current_values, current_x: springmv.csr_spmv(
            structure,
            current_x,
            values=current_values,
            subwarp_size=args.subwarp_size,
        )
    )
    unweighted_input_vjp = jax.jit(
        lambda cotangent: jax.vjp(
            lambda current_x: springmv.csr_spmv(
                structure, current_x, subwarp_size=args.subwarp_size
            ),
            x,
        )[1](cotangent)[0]
    )
    weighted_input_vjp = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: springmv.csr_spmv(
                structure,
                current_x,
                values=current_values,
                subwarp_size=args.subwarp_size,
            ),
            x,
        )[1](cotangent)[0]
    )
    weighted_input_vjp_scatter = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: springmv.csr_spmv(
                structure,
                current_x,
                values=current_values,
                subwarp_size=args.subwarp_size,
                input_vjp_strategy="scatter",
            ),
            x,
        )[1](cotangent)[0]
    )
    weighted_values_vjp = jax.jit(
        lambda current_x, cotangent: jax.vjp(
            lambda current_values: springmv.csr_spmv(
                structure,
                current_x,
                values=current_values,
                subwarp_size=args.subwarp_size,
            ),
            values,
        )[1](cotangent)[0]
    )
    weighted_logsumexp_forward = jax.jit(
        lambda current_values, current_x: springmv.csr_logsumexp(
            structure,
            current_x,
            log_values=current_values,
            subwarp_size=args.subwarp_size,
        )
    )
    unweighted_logsumexp_forward = jax.jit(
        lambda current_x: springmv.csr_logsumexp(
            structure,
            current_x,
            subwarp_size=args.subwarp_size,
        )
    )
    weighted_logsumexp_jvp = jax.jit(
        lambda current_values, current_x, values_dot, x_dot: jax.jvp(
            lambda tangent_values, tangent_x: springmv.csr_logsumexp(
                structure,
                tangent_x,
                log_values=tangent_values,
                subwarp_size=args.subwarp_size,
            ),
            (current_values, current_x),
            (values_dot, x_dot),
        )[1]
    )
    weighted_logsumexp_input_vjp = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: springmv.csr_logsumexp(
                structure,
                current_x,
                log_values=current_values,
                subwarp_size=args.subwarp_size,
            ),
            x,
        )[1](cotangent)[0]
    )
    unweighted_logsumexp_input_vjp = jax.jit(
        lambda cotangent: jax.vjp(
            lambda current_x: springmv.csr_logsumexp(
                structure,
                current_x,
                subwarp_size=args.subwarp_size,
            ),
            x,
        )[1](cotangent)[0]
    )
    weighted_logsumexp_input_vjp_scatter = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: springmv.csr_logsumexp(
                structure,
                current_x,
                log_values=current_values,
                subwarp_size=args.subwarp_size,
                input_vjp_strategy="scatter",
            ),
            x,
        )[1](cotangent)[0]
    )
    weighted_logsumexp_values_vjp = jax.jit(
        lambda current_x, cotangent: jax.vjp(
            lambda current_values: springmv.csr_logsumexp(
                structure,
                current_x,
                log_values=current_values,
                subwarp_size=args.subwarp_size,
            ),
            values,
        )[1](cotangent)[0]
    )
    jax_weighted_forward = jax.jit(jax_csr_matvec)
    jax_unweighted_input_vjp = jax.jit(
        lambda cotangent: jax.vjp(
            lambda current_x: jax_csr_matvec(unit_values, current_x), x
        )[1](cotangent)[0]
    )
    jax_weighted_input_vjp = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: jax_csr_matvec(current_values, current_x),
            x,
        )[1](cotangent)[0]
    )
    jax_weighted_values_vjp = jax.jit(
        lambda current_x, cotangent: jax.vjp(
            lambda current_values: jax_csr_matvec(
                current_values, current_x
            ),
            values,
        )[1](cotangent)[0]
    )
    jax_weighted_logsumexp_forward = jax.jit(jax_segment_logsumexp)
    jax_unweighted_logsumexp_forward = jax.jit(
        jax_segment_logsumexp_unweighted
    )
    jax_weighted_logsumexp_jvp = jax.jit(
        lambda current_values, current_x, values_dot, x_dot: jax.jvp(
            jax_segment_logsumexp,
            (current_values, current_x),
            (values_dot, x_dot),
        )[1]
    )
    jax_weighted_logsumexp_input_vjp = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: jax_segment_logsumexp(
                current_values, current_x
            ),
            x,
        )[1](cotangent)[0]
    )
    jax_unweighted_logsumexp_input_vjp = jax.jit(
        lambda cotangent: jax.vjp(
            jax_segment_logsumexp_unweighted,
            x,
        )[1](cotangent)[0]
    )
    jax_weighted_logsumexp_values_vjp = jax.jit(
        lambda current_x, cotangent: jax.vjp(
            lambda current_values: jax_segment_logsumexp(
                current_values, current_x
            ),
            values,
        )[1](cotangent)[0]
    )
    if args.diagnose_input_vjp:
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
        f"columns={args.column_pattern}"
    )
    print(
        "times are median [minimum, maximum]; final ratio is "
        "right median / left median"
    )
    _benchmark_pair(
        "arithmetic weighted forward",
        lambda: weighted_forward(values, x),
        lambda: jax_weighted_forward(values, x),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "arithmetic unweighted input VJP",
        lambda: unweighted_input_vjp(output_cotangent),
        lambda: jax_unweighted_input_vjp(output_cotangent),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "arithmetic input VJP transpose",
        lambda: weighted_input_vjp(values, output_cotangent),
        lambda: jax_weighted_input_vjp(values, output_cotangent),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "arithmetic input VJP scatter",
        lambda: weighted_input_vjp_scatter(values, output_cotangent),
        lambda: jax_weighted_input_vjp(values, output_cotangent),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "arithmetic weighted values VJP",
        lambda: weighted_values_vjp(x, output_cotangent),
        lambda: jax_weighted_values_vjp(x, output_cotangent),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp unweighted forward",
        lambda: unweighted_logsumexp_forward(x),
        lambda: jax_unweighted_logsumexp_forward(x),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp weighted forward",
        lambda: weighted_logsumexp_forward(values, x),
        lambda: jax_weighted_logsumexp_forward(values, x),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp weighted JVP",
        lambda: weighted_logsumexp_jvp(values, x, values, x),
        lambda: jax_weighted_logsumexp_jvp(values, x, values, x),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp unweighted input VJP",
        lambda: unweighted_logsumexp_input_vjp(output_cotangent),
        lambda: jax_unweighted_logsumexp_input_vjp(output_cotangent),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp input VJP transpose",
        lambda: weighted_logsumexp_input_vjp(values, output_cotangent),
        lambda: jax_weighted_logsumexp_input_vjp(
            values, output_cotangent
        ),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp input VJP scatter",
        lambda: weighted_logsumexp_input_vjp_scatter(
            values, output_cotangent
        ),
        lambda: jax_weighted_logsumexp_input_vjp(
            values, output_cotangent
        ),
        args.warmup,
        args.iterations,
    )
    _benchmark_pair(
        "log-sum-exp values VJP",
        lambda: weighted_logsumexp_values_vjp(x, output_cotangent),
        lambda: jax_weighted_logsumexp_values_vjp(x, output_cotangent),
        args.warmup,
        args.iterations,
    )
    if args.diagnose_input_vjp:
        print("\nweighted arithmetic input-VJP diagnostics")
        _benchmark_pair(
            "permutation gather",
            lambda: weighted_input_vjp(values, output_cotangent),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        )
        _benchmark_pair(
            "SpringMV atomic scatter",
            lambda: weighted_input_vjp_scatter(
                values, output_cotangent
            ),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        )
        _benchmark_pair(
            "pre-reordered values",
            lambda: reordered_input_vjp(
                transposed_values, output_cotangent
            ),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        )
        _benchmark_pair(
            "reorder included",
            lambda: reorder_and_input_vjp(values, output_cotangent),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        )
        atomic_samples, cusparse_samples = _time_pair(
            lambda: jax_atomic_input_vjp(values, output_cotangent),
            lambda: jax_weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        )
        _report_comparison(
            "atomic scatter",
            atomic_samples,
            cusparse_samples,
            springmv_label="JAX-scatter",
            jax_label="cuSPARSE",
        )


if __name__ == "__main__":
    main()
