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

import springmv


def _time(
    operation: Callable[[], object], warmup: int, iterations: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        jax.block_until_ready(operation())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(operation())
        samples.append((time.perf_counter() - start) * 1.0e3)
    return statistics.median(samples), min(samples), max(samples)


def _report(name: str, samples: tuple[float, float, float]) -> None:
    median, minimum, maximum = samples
    print(
        f"{name:32s} median {median:9.4f} ms  "
        f"min {minimum:9.4f} ms  max {maximum:9.4f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--cols", type=int, default=1 << 20)
    parser.add_argument("--nnz-per-row", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    nnz = args.rows * args.nnz_per_row
    row_offsets = np.arange(
        0, nnz + 1, args.nnz_per_row, dtype=np.int32
    )
    column_indices = rng.integers(
        0, args.cols, size=nnz, dtype=np.int32
    )
    values = jnp.asarray(rng.normal(size=nnz).astype(np.float32))
    x = jnp.asarray(rng.normal(size=args.cols).astype(np.float32))
    output_cotangent = jnp.asarray(
        rng.normal(size=args.rows).astype(np.float32)
    )
    structure = springmv.CsrStructure.from_arrays(
        row_offsets, column_indices, num_cols=args.cols
    )

    weighted_forward = jax.jit(
        lambda current_values, current_x: springmv.csr_spmv(
            structure, current_x, values=current_values
        )
    )
    unweighted_input_vjp = jax.jit(
        lambda cotangent: jax.vjp(
            lambda current_x: springmv.csr_spmv(structure, current_x), x
        )[1](cotangent)[0]
    )
    weighted_input_vjp = jax.jit(
        lambda current_values, cotangent: jax.vjp(
            lambda current_x: springmv.csr_spmv(
                structure, current_x, values=current_values
            ),
            x,
        )[1](cotangent)[0]
    )
    weighted_values_vjp = jax.jit(
        lambda current_x, cotangent: jax.vjp(
            lambda current_values: springmv.csr_spmv(
                structure, current_x, values=current_values
            ),
            values,
        )[1](cotangent)[0]
    )

    print(
        f"rows={args.rows} cols={args.cols} nnz={nnz} "
        f"nnz/row={args.nnz_per_row}"
    )
    _report(
        "weighted forward",
        _time(
            lambda: weighted_forward(values, x), args.warmup, args.iterations
        ),
    )
    _report(
        "unweighted input VJP",
        _time(
            lambda: unweighted_input_vjp(output_cotangent),
            args.warmup,
            args.iterations,
        ),
    )
    _report(
        "weighted input VJP",
        _time(
            lambda: weighted_input_vjp(values, output_cotangent),
            args.warmup,
            args.iterations,
        ),
    )
    _report(
        "weighted values VJP",
        _time(
            lambda: weighted_values_vjp(x, output_cotangent),
            args.warmup,
            args.iterations,
        ),
    )


if __name__ == "__main__":
    main()
