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
CSR indices, and NVIDIA CUDA. It has no Ginkgo runtime dependency. Python and
JAX are outside the current milestone.

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

## Benchmark

```bash
./build/springmv_benchmark
./build/springmv_benchmark --rows 1048576 --cols 1048576 \
  --nnz-per-row 32 --iterations 50 --warmup 10
```

The benchmark uses CUDA events and reports weighted and unweighted arithmetic,
arithmetic cuSPARSE, and weighted and unweighted SpringMV log-sum-exp over the
same CSR data, followed by weighted and unweighted min-plus and max-plus. It
reports median, minimum, and maximum kernel times plus GNNZ/s and effective
GB/s. Unweighted byte counts exclude edge values. Arithmetic operations also
report GFLOP/s. Setup, validation, memory allocation, and transfers are outside
the timed region.

## C++ API

```cpp
#include <springmv/csr_spmv.hpp>
#include <springmv/csr_logsumexp.hpp>
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

