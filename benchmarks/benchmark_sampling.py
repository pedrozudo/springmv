# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark atomic-free sampling against a Klay-style JAX formulation."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import springmv


def _sample(operation: Callable[[], object]) -> float:
    start = time.perf_counter()
    jax.block_until_ready(operation())
    return (time.perf_counter() - start) * 1.0e3


def _summarize(samples: Sequence[float]) -> tuple[float, float, float]:
    return statistics.median(samples), min(samples), max(samples)


def _time_pair(
    springmv_operation: Callable[[], object],
    jax_operation: Callable[[], object],
    warmup: int,
    iterations: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    for _ in range(warmup):
        jax.block_until_ready(springmv_operation())
        jax.block_until_ready(jax_operation())

    springmv_samples: list[float] = []
    jax_samples: list[float] = []
    for iteration in range(iterations):
        if iteration % 2 == 0:
            springmv_samples.append(_sample(springmv_operation))
            jax_samples.append(_sample(jax_operation))
        else:
            jax_samples.append(_sample(jax_operation))
            springmv_samples.append(_sample(springmv_operation))
    return _summarize(springmv_samples), _summarize(jax_samples)


def _time_one(
    operation: Callable[[], object], warmup: int, iterations: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        jax.block_until_ready(operation())
    return _summarize([_sample(operation) for _ in range(iterations)])


def _format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--nnz-per-row", type=int, default=8)
    parser.add_argument(
        "--batch-sizes", type=_parse_batch_sizes, default=[1, 8, 32, 128]
    )
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
        "--batched-weights",
        action="store_true",
        help="use a distinct log-weight vector for every batch element",
    )
    parser.add_argument(
        "--springmv-only",
        action="store_true",
        help="skip construction and execution of the JAX segment/scatter baseline",
    )
    args = parser.parse_args()

    for name in ("rows", "cols", "nnz_per_row", "iterations", "warmup"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    rng = np.random.default_rng(args.seed)
    nnz = args.rows * args.nnz_per_row
    if nnz > np.iinfo(np.int32).max:
        parser.error("the requested matrix has too many nonzeros for int32")
    row_offsets = np.arange(0, nnz + 1, args.nnz_per_row, dtype=np.int32)
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
        column_indices = rng.integers(
            0, max(1, args.cols // 100), size=nnz, dtype=np.int32
        )
    else:
        column_indices = np.zeros(nnz, dtype=np.int32)

    structure = springmv.CsrStructure.from_arrays(
        row_offsets, column_indices, num_cols=args.cols
    )
    edge_rows = jnp.asarray(
        np.repeat(np.arange(args.rows, dtype=np.int32), args.nnz_per_row)
    )
    edge_ids = jnp.arange(nnz, dtype=jnp.int32)
    shared_weights = jnp.asarray(rng.normal(size=nnz).astype(np.float32))

    def jax_sample_one(log_weights, gumbels, active_parent):
        scores = log_weights + gumbels
        row_max = jax.ops.segment_max(
            scores,
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )
        maximum = scores == row_max[edge_rows]
        winning_edge = jax.ops.segment_min(
            jnp.where(maximum, edge_ids, nnz),
            edge_rows,
            num_segments=args.rows,
            indices_are_sorted=True,
        )
        selected = maximum & (edge_ids == winning_edge[edge_rows])
        selected &= active_parent[edge_rows]
        return (
            jax.ops.segment_sum(
                selected.astype(jnp.int32),
                structure.column_indices,
                num_segments=args.cols,
            )
            > 0
        )

    def springmv_one(log_weights, gumbels, active_parent):
        return springmv.csr_logsumexp_sample(
            structure,
            log_weights,
            gumbels,
            active_parent,
            subwarp_size=args.subwarp_size,
        )

    print(
        f"rows={args.rows} cols={args.cols} nnz={nnz} "
        f"nnz/row={args.nnz_per_row} columns={args.column_pattern} "
        f"subwarp={args.subwarp_size} batched-weights={args.batched_weights}"
    )
    print("times are median [minimum, maximum]; ratio is JAX/SpringMV")

    for batch_size in args.batch_sizes:
        key = jax.random.key(args.seed + batch_size)
        gumbels = jax.random.gumbel(
            key, (batch_size, nnz), dtype=jnp.float32
        )
        active_parent = jnp.asarray(
            rng.random((batch_size, args.rows)) < 0.5
        )
        if args.batched_weights:
            scales = jnp.linspace(0.9, 1.1, batch_size, dtype=jnp.float32)
            log_weights = shared_weights[None, :] * scales[:, None]
            springmv_operation = jax.jit(
                jax.vmap(springmv_one, in_axes=(0, 0, 0))
            )
            jax_operation = jax.jit(
                jax.vmap(jax_sample_one, in_axes=(0, 0, 0))
            )
        else:
            log_weights = shared_weights
            springmv_operation = jax.jit(
                jax.vmap(springmv_one, in_axes=(None, 0, 0))
            )
            jax_operation = jax.jit(
                jax.vmap(jax_sample_one, in_axes=(None, 0, 0))
            )

        springmv_bound = lambda: springmv_operation(
            log_weights, gumbels, active_parent
        )
        jax_bound = lambda: jax_operation(log_weights, gumbels, active_parent)
        gumbel_bytes = batch_size * nnz * np.dtype(np.float32).itemsize
        selected_bytes = batch_size * args.rows * np.dtype(np.int32).itemsize
        active_bytes = batch_size * args.rows * np.dtype(np.bool_).itemsize
        output_bytes = batch_size * args.cols * np.dtype(np.bool_).itemsize
        weight_bytes = (
            gumbel_bytes
            if args.batched_weights
            else nnz * np.dtype(np.float32).itemsize
        )
        try:
            springmv_result = jax.block_until_ready(springmv_bound())
        except jax.errors.JaxRuntimeError as error:
            first_line = str(error).splitlines()[0]
            print(f"batch={batch_size:6d}  SpringMV FAILED: {first_line}")
            print(
                " " * 14
                + "explicit buffers: "
                + f"weights={_format_bytes(weight_bytes)}, "
                + f"gumbels={_format_bytes(gumbel_bytes)}, "
                + f"parent activity={_format_bytes(active_bytes)}, "
                + "SpringMV selected-child scratch="
                + f"{_format_bytes(selected_bytes)}, "
                + f"output={_format_bytes(output_bytes)}"
            )
            continue
        if args.springmv_only:
            springmv_median, springmv_minimum, springmv_maximum = _time_one(
                springmv_bound, args.warmup, args.iterations
            )
            print(
                f"batch={batch_size:6d}  "
                f"SpringMV {springmv_median:9.4f} "
                f"[{springmv_minimum:9.4f}, {springmv_maximum:9.4f}] ms"
            )
            print(
                " " * 14
                + "explicit buffers: "
                + f"weights={_format_bytes(weight_bytes)}, "
                + f"gumbels={_format_bytes(gumbel_bytes)}, "
                + f"parent activity={_format_bytes(active_bytes)}, "
                + "SpringMV selected-child scratch="
                + f"{_format_bytes(selected_bytes)}, "
                + f"output={_format_bytes(output_bytes)}"
            )
            continue
        try:
            jax_result = jax.block_until_ready(jax_bound())
            np.testing.assert_array_equal(springmv_result, jax_result)
            springmv_times, jax_times = _time_pair(
                springmv_bound, jax_bound, args.warmup, args.iterations
            )
            springmv_median, springmv_minimum, springmv_maximum = springmv_times
            jax_median, jax_minimum, jax_maximum = jax_times
            print(
                f"batch={batch_size:6d}  "
                f"SpringMV {springmv_median:9.4f} "
                f"[{springmv_minimum:9.4f}, {springmv_maximum:9.4f}] ms  "
                f"JAX segment/scatter {jax_median:9.4f} "
                f"[{jax_minimum:9.4f}, {jax_maximum:9.4f}] ms  "
                f"JAX/SpringMV {jax_median / springmv_median:7.2f}x"
            )
        except jax.errors.JaxRuntimeError as error:
            springmv_median, springmv_minimum, springmv_maximum = _time_one(
                springmv_bound, args.warmup, args.iterations
            )
            first_line = str(error).splitlines()[0]
            print(
                f"batch={batch_size:6d}  "
                f"SpringMV {springmv_median:9.4f} "
                f"[{springmv_minimum:9.4f}, {springmv_maximum:9.4f}] ms  "
                f"JAX segment/scatter FAILED: {first_line}"
            )
        print(
            " " * 14
            + "explicit buffers: "
            + f"weights={_format_bytes(weight_bytes)}, "
            + f"gumbels={_format_bytes(gumbel_bytes)}, "
            + f"parent activity={_format_bytes(active_bytes)}, "
            + f"SpringMV selected-child scratch={_format_bytes(selected_bytes)}, "
            + f"output={_format_bytes(output_bytes)}"
        )


if __name__ == "__main__":
    main()
