# Benchmarks

SpringMV includes separate JAX, sampling, and native C++ benchmarks. Setup,
allocation, transfers, structure construction, and JIT compilation are outside
the timed regions.

## JAX reductions

Run the default benchmark with:

```bash
python benchmarks/benchmark_jax.py
```

Configure matrix and batch sizes explicitly with:

```bash
python benchmarks/benchmark_jax.py \
  --rows 1048576 --cols 1048576 --nnz-per-row 16 \
  --batch-size 128 --iterations 50 --warmup 10
```

The benchmark reports synchronized median, minimum, and maximum execution
times. It separates unweighted and weighted cases and measures:

- Single-vector forward execution.
- Single-vector combined forward and backward execution.
- `jax.vmap` forward execution.
- `jax.vmap` combined forward and backward execution.

The combined measurement requests only the gradient of
`0.5 * sum(output**2)`. Its output-dependent cotangent keeps the forward pass
in the compiled computation without requiring the complete forward result to
be returned. Unweighted backward differentiates the input. Weighted backward
differentiates both the edge values and input; batched weighted measurements
share one edge-value array and accumulate its gradient across the batch.

Forward-plus-backward measurements report both `scatter` and `transpose`
input-gradient strategies for weighted and unweighted products. Forward-only
execution is identical between the strategies and is reported once.

### JAX baselines

Each operation is paired with a native-JAX baseline. Ratios are printed as
`JAX median / SpringMV median`, so values greater than one mean SpringMV is
faster.

- Single-vector arithmetic uses `jax.experimental.sparse.csr_matvec`. On a
  CUDA-enabled JAX installation this baseline may use JAX's packaged cuSPARSE.
- Batched arithmetic uses a native-JAX segment reduction.
- Log-sum-exp uses stable `jax.ops.segment_max` and
  `jax.ops.segment_sum` compositions because JAX has no sparse log-semiring
  matvec.

The unweighted JAX CSR baseline necessarily stores unit edge values. Segment
baselines store a precomputed row index for every nonzero, while SpringMV
consumes CSR row offsets directly.

### Matrix patterns and scheduling

`--column-pattern` accepts:

- `uniform` for uniformly distributed columns.
- `diagonal` for duplicate-heavy rows.
- `hotspot` for concentrated columns.
- `single` for maximum input-gradient contention.

`--subwarp-size` selects an explicit SpringMV width of `1`, `2`, `4`, `8`,
`16`, or `32`. The default `0` selects automatically from the maximum row
length.

Use `--diagnose-input-vjp` for isolated backward and value-reordering
measurements:

```bash
python benchmarks/benchmark_jax.py \
  --rows 1048576 --cols 1048576 --nnz-per-row 16 \
  --diagnose-input-vjp
```

The diagnostic arithmetic section separates permutation gather,
pre-reordered transpose values, dynamic value reordering, and native-JAX
scatter/CSR paths. The pre-reordered measurement excludes the one-time reorder;
the dynamic measurement includes it.

## Sampling

Compare atomic-free sampling with an exact JAX segment/scatter formulation:

```bash
python benchmarks/benchmark_sampling.py \
  --rows 4096 --cols 4096 --nnz-per-row 8 \
  --batch-sizes 1,8,32,128,1000,10000
```

The benchmark supplies identical pre-generated Gumbels to SpringMV and its JAX
reference and verifies bitwise-equal Boolean outputs before timing the complete
selection-and-propagation operation. The reference implements the same
lowest-edge-index tie rule but materializes generic edge-wise segment/scatter
intermediates.

The output reports explicit sizes for weights, Gumbels, parent activity,
selected-child scratch, and output buffers. Use:

- `--batched-weights` for distinct edge weights per batch element.
- `--column-pattern` for uniform, duplicate-heavy, hotspot, or single-child
  topologies.
- `--springmv-only` for memory-ceiling tests that should not compile or execute
  the generic JAX baseline.

The public sampling API receives a materialized `(batch, nnz)` Gumbel buffer,
which usually dominates large-batch memory usage.

## Native C++ benchmark

When system cuSPARSE development libraries are available, CMake builds a
native comparison executable:

```bash
./build/springmv_benchmark
./build/springmv_benchmark \
  --rows 1048576 --cols 1048576 --nnz-per-row 32 \
  --iterations 50 --warmup 10
```

It uses CUDA events and reports weighted and unweighted arithmetic, a cuSPARSE
arithmetic baseline, SpringMV log-sum-exp, min-plus, and max-plus over the same
CSR data. Results include median, minimum, and maximum times, GNNZ/s, effective
GB/s, and arithmetic GFLOP/s. Unweighted byte counts exclude edge values.

The native target is omitted when cuSPARSE is unavailable or when configured
with:

```bash
cmake -S . -B build -DSPRINGMV_ENABLE_CUSPARSE_VALIDATION=OFF
```

The JAX and sampling benchmarks remain available and do not directly link
system cuSPARSE.
