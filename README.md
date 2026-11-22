# SpringMV

SpringMV is a standalone CUDA sparse matrix-vector multiplication library. It
uses Ginkgo's classical CSR row/subwarp scheduling for several semiring
products:

```text
Weighted arithmetic:   y[row] = sum(values[nz] * x[column_indices[nz]])
Unweighted arithmetic: y[row] = sum(x[column_indices[nz]])
Weighted log:          y[row] = logsumexp(log_values[nz] + log_x[column_indices[nz]])
Unweighted log:        y[row] = logsumexp(log_x[column_indices[nz]])
Weighted min-plus:     y[row] = min(edge_values[nz] + x[column_indices[nz]])
Unweighted min-plus:   y[row] = min(x[column_indices[nz]])
Weighted max-plus:     y[row] = max(edge_values[nz] + x[column_indices[nz]])
Unweighted max-plus:   y[row] = max(x[column_indices[nz]])
```

The current implementation intentionally supports only float32 values, int32
CSR indices, and NVIDIA CUDA. It has no Ginkgo runtime dependency. A typed JAX
FFI frontend exposes the same kernels without host copies and supports
`jax.jit` plus first-order forward- and reverse-mode differentiation for
arithmetic and log-sum-exp SpMV.

Arithmetic, log-sum-exp, min-plus, and max-plus are compile-time policies over
one internal CSR row-reduction engine. Scheduling, validation,
weighted/unweighted dispatch, and subwarp reduction are shared without
exposing templates in the public API.

## Requirements

- CMake 3.24 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit with cuSPARSE
- A CUDA-capable NVIDIA GPU
- Ninja (recommended, but not required)

The Python package additionally requires Python 3.10 or newer and a
CUDA-enabled JAX installation.

## Build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
ctest --test-dir build --output-on-failure
```

If several CUDA toolkits are installed, select one explicitly during the first
configure. For example:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
```

The arithmetic correctness suite compares SpringMV against cuSPARSE. The
log-sum-exp suite uses a stable test-only long-double oracle, transformed
double-precision cuSPARSE for moderate finite inputs, analytic edge cases, and
metamorphic properties. Tests cover empty rows, infinities, NaNs, duplicate
column indices, irregular row lengths, extreme values, every compiled subwarp
width, automatic selection, non-default CUDA streams, and equivalence between
unweighted overloads and unit arithmetic or zero log weights. The transpose
suite compares weighted and structure-only results with cuSPARSE CSR-to-CSC
and validates the
edge permutation, deterministic ordering, rectangular matrices, duplicates,
empty rows and columns, and workspace errors.

The tropical suite covers both semirings and weighting modes, empty-row
identities, infinities, NaNs, duplicates, every subwarp width, zero-weight
equivalence, permutation invariance, and min/max duality.

The arithmetic and log-sum-exp backward suites check weighted and unweighted
JVPs and input VJPs plus edge-value VJPs for every subwarp width on a
non-default CUDA stream. The JAX suite additionally compares derivatives with
a pure-JAX oracle, checks the JVP/VJP adjoint identity and finite differences,
and covers `grad`, `value_and_grad`, JIT composition, dynamic `CsrStructure`
arguments, duplicates, rectangular matrices, empty structures, infinities,
and NaNs.

## Benchmark

```bash
./build/springmv_benchmark
./build/springmv_benchmark --rows 1048576 --cols 1048576 \
  --nnz-per-row 32 --iterations 50 --warmup 10

python benchmarks/benchmark_jax.py
python benchmarks/benchmark_jax.py --rows 1048576 --cols 1048576 \
  --nnz-per-row 16 --batch-size 8 --iterations 50 --warmup 10

# Diagnose weighted arithmetic input-VJP alternatives.
python benchmarks/benchmark_jax.py --rows 1048576 --cols 1048576 \
  --nnz-per-row 16 --diagnose-input-vjp
```

The benchmark uses CUDA events and reports weighted and unweighted arithmetic,
arithmetic cuSPARSE, and weighted and unweighted SpringMV log-sum-exp over the
same CSR data, followed by weighted and unweighted min-plus and max-plus. It
reports median, minimum, and maximum kernel times plus GNNZ/s and effective
GB/s. Unweighted byte counts exclude edge values. Arithmetic operations also
report GFLOP/s. Setup, validation, memory allocation, and transfers are outside
the timed region.

The JAX benchmark reports synchronized median, minimum, and maximum execution
times for arithmetic and log-sum-exp. It first reports the unweighted case and
then the weighted case. Each section contains single-vector and `jax.vmap`
measurements for forward execution and combined forward-plus-backward
execution. The combined measurement requests only the gradient of
`0.5 * sum(output**2)`. Its output-dependent cotangent guarantees that the
forward pass remains part of the compiled computation without forcing the
complete forward result to be returned. Unweighted backward differentiates the
input. Weighted backward differentiates both edge values and the input;
batched weighted measurements share one edge value array and accumulate its
gradient across the batch. Every weighted forward-plus-backward case is
reported with both `scatter` and `transpose` input-gradient strategies;
forward-only execution is identical between those strategies and is reported
once.

Each SpringMV operation is paired with a native-JAX baseline and a speedup
computed as `JAX median / SpringMV median`, so a value greater than one means
SpringMV is faster. Single-vector arithmetic uses
`jax.experimental.sparse.csr_matvec`; its unweighted baseline necessarily
stores unit edge values. Batched arithmetic uses a native-JAX segment
reduction. Because JAX has no sparse log-semiring matvec, its log-sum-exp
baseline uses a stable composition of `jax.ops.segment_max` and
`jax.ops.segment_sum`. Segment baselines store a precomputed row index for every
nonzero, while SpringMV consumes CSR row offsets directly. Structure
construction, row-index construction, and JIT compilation occur outside the
timed region.

SpringMV unweighted backward uses the precomputed transpose. `--batch-size`
selects the mapped input batch size. Output labels distinguish the
single-vector `JAX CSR` baseline from `JAX segment` baselines.
`--diagnose-input-vjp` additionally reports isolated backward-only and
value-reordering diagnostics.

`--subwarp-size` selects an explicit SpringMV width for scheduling experiments.
`--column-pattern` selects `uniform`, duplicate-heavy `diagonal`, concentrated
`hotspot`, or maximally contended `single` column indices.
`--diagnose-input-vjp` additionally compares the scatter and transpose input
VJPs for single and batched inputs. Its detailed arithmetic section separates
permutation-gather, pre-reordered transpose values, dynamic value reordering,
and native-JAX scatter/CSR paths. The pre-reordered measurement excludes the
one-time reorder; the dynamic measurement includes it.

## Python and JAX API

Create or activate a virtual environment with CUDA-enabled JAX, then install
SpringMV from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

If multiple CUDA toolkits are installed, select a coherent compiler and
toolkit explicitly. For example:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS='-DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=native' \
python -m pip install -e '.[test]'
```

The four sparse reductions share one immutable `CsrStructure`. Constructing it
is an eager preprocessing step that validates the topology, builds its
transpose, and computes static scheduling metadata:

```python
import jax
import jax.numpy as jnp

from springmv import CsrStructure, csr_logsumexp

row_offsets = jnp.array([0, 2, 3], dtype=jnp.int32)
column_indices = jnp.array([0, 2, 1], dtype=jnp.int32)
log_values = jnp.array([0.2, -0.4, 0.7], dtype=jnp.float32)
log_x = jnp.array([-1.0, -2.0, -3.0], dtype=jnp.float32)

structure = CsrStructure.from_arrays(
    row_offsets,
    column_indices,
    num_cols=log_x.shape[0],
)

weighted = csr_logsumexp(
    structure,
    log_x,
    log_values=log_values,
)

# No dummy edge values are allocated or read by the unweighted target.
unweighted = csr_logsumexp(
    structure,
    log_x,
)

compiled = jax.jit(
    lambda x: csr_logsumexp(
        structure,
        x,
    )
)
compiled_result = compiled(log_x)
```

The CSR topology remains shared under `jax.vmap`. Inputs and edge values may
be mapped independently, so shared weights, per-example weights, or both
per-example weights and inputs are supported:

```python
log_x_batch = jnp.stack([log_x, log_x + 0.5])
log_values_batch = jnp.stack([log_values, log_values - 0.25])

# One sparse matrix and one set of weights applied to a batch of inputs.
shared_weights = jax.jit(
    jax.vmap(lambda current_x: csr_logsumexp(
        structure, current_x, log_values=log_values
    ))
)(log_x_batch)

# Per-example edge weights and inputs, still sharing the CSR topology.
batched_weights = jax.jit(
    jax.vmap(lambda current_values, current_x: csr_logsumexp(
        structure, current_x, log_values=current_values
    ))
)(log_values_batch, log_x_batch)
```

Vmapped forward reductions use one fused batch-aware CUDA launch. Arithmetic
and log-sum-exp `vmap` also compose with first-order `jvp`, VJP, and `grad`;
their mapped backward rules reuse the existing per-example native kernels.
Batched or ragged `CsrStructure` objects are not supported.

`csr_spmv` accepts optional `values`, `csr_logsumexp` accepts optional
`log_values`, and `csr_minplus`/`csr_maxplus` accept optional `edge_values`.
Omitting the corresponding array dispatches to a distinct unweighted FFI
target.

Arithmetic and log-sum-exp SpMV support first-order forward- and reverse-mode
differentiation. Their weighted forms differentiate both the input and edge
values:

```python
from springmv import csr_spmv

values = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32)
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

primal, tangent = jax.jvp(
    lambda current_values, current_x: csr_spmv(
        structure, current_x, values=current_values
    ),
    (values, x),
    (jnp.ones_like(values), jnp.ones_like(x)),
)

loss, (values_grad, x_grad) = jax.value_and_grad(
    lambda current_values, current_x: csr_spmv(
        structure, current_x, values=current_values
    ).sum(),
    argnums=(0, 1),
)(values, x)

log_primal, log_tangent = jax.jvp(
    lambda current_log_values, current_log_x: csr_logsumexp(
        structure,
        current_log_x,
        log_values=current_log_values,
    ),
    (log_values, log_x),
    (jnp.ones_like(log_values), jnp.ones_like(log_x)),
)

log_values_grad, log_x_grad = jax.grad(
    lambda current_log_values, current_log_x: csr_logsumexp(
        structure,
        current_log_x,
        log_values=current_log_values,
    ).sum(),
    argnums=(0, 1),
)(log_values, log_x)
```

Unweighted input VJPs always use the transpose stored in `CsrStructure`.
Weighted arithmetic and log-sum-exp operations retain transpose and scatter
implementations for benchmarking. `input_vjp_strategy="transpose"` reads
original values through `transpose_permutation`, while
`input_vjp_strategy="scatter"` traverses original CSR and atomically
accumulates into the input cotangent. Scatter is the default for weighted
operations because it avoids indirect value reads. Transpose remains available
for matrices where atomic contention makes it faster:

```python
x_grad_scatter = jax.grad(
    lambda current_x: csr_logsumexp(
        structure,
        current_x,
        log_values=log_values,
        input_vjp_strategy="scatter",
    ).sum()
)(log_x)
```

The transpose strategy is deterministic. Atomic scatter avoids permutation
reads but its floating-point accumulation order is not deterministic.
Higher-order differentiation is not currently supported.

At extended-real boundary points, log-sum-exp derivatives use a deterministic
symmetric convention. Multiple `+inf` terms split the derivative equally;
when every term in a nonempty row is `-inf`, all terms split it equally. Empty
rows have zero derivatives. A NaN term makes the row output and all derivatives
associated with that row NaN.

Values are deliberately not part of `CsrStructure`, so the same topology can
be reused with changing or trainable weights. Calling `transpose()` returns a
new structure using the already-precomputed transpose:

```python
transposed_structure = structure.transpose()
transposed_log_values = log_values[structure.transpose_permutation]
```

The transpose swaps dimensions and scheduling metadata and inverts the edge
permutation. Constructing or transposing a structure is intentionally rejected
inside JAX transformations. The lower-level `csr_transpose` array API remains
available when only raw transposed arrays are required:

```python
from springmv import csr_transpose

t_row_offsets, t_column_indices = csr_transpose(
    row_offsets, column_indices, num_cols=3
)
```

Supplying `values=` also returns transposed values. Temporary CUDA workspace is
managed internally. When `return_permutation=False`, no permutation output is
allocated.

Current transformation support is:

| Operation | Eager | `jax.jit` | `jax.jvp` | VJP/`jax.grad` | `jax.vmap` |
|---|---:|---:|---:|---:|---:|
| Arithmetic CSR reduction | Yes | Yes | Yes | Yes | Yes |
| Log-sum-exp CSR reduction | Yes | Yes | Yes | Yes | Yes |
| Min-plus and max-plus reductions | Yes | Yes | No | No | Yes |
| CSR structure preprocessing | Yes | No | No | No | No |

## C++ API

```cpp
#include <springmv/csr_spmv.hpp>
#include <springmv/csr_spmv_backward.hpp>
#include <springmv/csr_logsumexp.hpp>
#include <springmv/csr_logsumexp_backward.hpp>
#include <springmv/csr_batched.hpp>
#include <springmv/csr_tropical.hpp>
#include <springmv/csr_transpose.hpp>

springmv::CsrKernelOptions options{
    .max_nnz_per_row = max_row_length,
    .subwarp_size = 0,  // automatic
};

springmv::csr_spmv(num_rows, d_row_offsets, d_column_indices, d_values,
                   d_x, d_y, options, stream);

// Unweighted arithmetic: no edge-value array is stored or read.
springmv::csr_spmv(num_rows, d_row_offsets, d_column_indices,
                   d_x, d_y, options, stream);

springmv::csr_logsumexp(num_rows, d_row_offsets, d_column_indices,
                        d_log_values, d_log_x, d_log_y, options, stream);

// Unweighted log semiring: no edge-value array is stored or read.
springmv::csr_logsumexp(num_rows, d_row_offsets, d_column_indices,
                        d_log_x, d_log_y, options, stream);

springmv::csr_minplus(num_rows, d_row_offsets, d_column_indices,
                      d_edge_values, d_x, d_min_y, options, stream);
springmv::csr_minplus(num_rows, d_row_offsets, d_column_indices,
                      d_x, d_min_y, options, stream);

springmv::csr_maxplus(num_rows, d_row_offsets, d_column_indices,
                      d_edge_values, d_x, d_max_y, options, stream);
springmv::csr_maxplus(num_rows, d_row_offsets, d_column_indices,
                      d_x, d_max_y, options, stream);

// Weighted input VJP. The transpose structure and permutation are reusable.
springmv::csr_spmv_input_vjp(
    num_cols, d_transposed_row_offsets, d_transposed_column_indices,
    d_edge_permutation, d_values, d_output_cotangent,
    d_input_cotangent, transpose_options, stream);

// Alternative weighted input VJP over original CSR using atomic scatter.
springmv::csr_spmv_input_vjp_scatter(
    num_rows, num_cols, d_row_offsets, d_column_indices, d_values,
    d_output_cotangent, d_input_cotangent, options, stream);

// Per-edge weighted value VJP.
springmv::csr_spmv_values_vjp(
    num_rows, d_row_offsets, d_column_indices, d_x,
    d_output_cotangent, d_values_cotangent, options, stream);

// Weighted log-sum-exp JVP. d_log_y is the previously computed primal.
springmv::csr_logsumexp_jvp(
    num_rows, d_row_offsets, d_column_indices, d_log_values, d_log_x,
    d_log_y, d_log_values_tangent, d_log_x_tangent,
    d_log_y_tangent, options, stream);

// Weighted log-sum-exp input VJP using the precomputed transpose.
springmv::csr_logsumexp_input_vjp(
    num_cols, d_row_offsets, d_column_indices,
    d_transposed_row_offsets, d_transposed_column_indices,
    d_edge_permutation, d_log_values, d_log_x, d_log_y,
    d_output_cotangent, d_log_x_cotangent,
    transpose_options, stream);

// Alternative weighted log-sum-exp input VJP using atomic scatter.
springmv::csr_logsumexp_input_vjp_scatter(
    num_rows, num_cols, d_row_offsets, d_column_indices,
    d_log_values, d_log_x, d_log_y, d_output_cotangent,
    d_log_x_cotangent, options, stream);

springmv::csr_logsumexp_values_vjp(
    num_rows, d_row_offsets, d_column_indices, d_log_values, d_log_x,
    d_log_y, d_output_cotangent, d_log_values_cotangent,
    options, stream);

// Batch-major dense inputs and outputs. A zero values stride shares weights.
springmv::csr_logsumexp_batched(
    batch_size, num_rows, d_row_offsets, d_column_indices,
    d_log_values, 0, d_log_x_batch, num_cols, d_log_y_batch,
    options, stream);
```

All pointers are device pointers. Calls enqueue work on the supplied CUDA
stream without allocating or synchronizing. Explicit subwarp widths of 1, 2,
4, 8, 16, and 32 are available for experiments. Log-semiring empty rows and
rows containing only `-inf` produce `-inf`; any `+inf` term produces `+inf`,
and NaN terms propagate.

Min-plus empty rows produce `+inf`; max-plus empty rows produce `-inf`. Both
tropical operations propagate NaNs. Their weighted forms use additive edge
weights, while the unweighted overloads store and read no edge-value array.

## CSR transpose

`csr_transpose` constructs CSR for `A` transposed entirely on the GPU. It has
structure-only and weighted overloads, uses caller-owned temporary storage,
and executes asynchronously on a CUDA stream. Query the storage requirement
before allocating it:

```cpp
std::size_t workspace_size{};
springmv::csr_transpose_workspace_size(
    num_rows, num_cols, num_nonzeros, &workspace_size);

springmv::csr_transpose(
    num_rows, num_cols, num_nonzeros,
    d_row_offsets, d_column_indices, d_values,
    d_transposed_row_offsets, d_transposed_column_indices,
    d_transposed_values, d_edge_permutation,
    d_workspace, workspace_size, stream);

// Unweighted/structure-only transpose: no edge values are stored or read.
springmv::csr_transpose(
    num_rows, num_cols, num_nonzeros,
    d_row_offsets, d_column_indices,
    d_transposed_row_offsets, d_transposed_column_indices,
    d_edge_permutation,
    d_workspace, workspace_size, stream);
```

The output has `num_cols` rows and `num_rows` columns. Entries within each
output row are deterministically ordered by original row. The optional edge
permutation maps each transposed entry back to its input nonzero position,
allowing changing values to be gathered without rebuilding the structure.

## Provenance and licensing

SpringMV is licensed under BSD-3-Clause. Its classical CSR row/subwarp
scheduling is adapted from Ginkgo v1.11.0, commit
`4c77a5d826bd1597fd093f39c47688f8cc735690`, specifically
`common/cuda_hip/matrix/csr_kernels.template.cpp`. The adapted source files
retain the Ginkgo copyright and provenance information.

