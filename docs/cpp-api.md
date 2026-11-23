# Low-level C++/CUDA API

The JAX interface is SpringMV's primary public API. The native layer is
available for applications that need direct CUDA stream and device-pointer
control.

All pointers passed to SpringMV are device pointers. Calls enqueue work on the
provided CUDA stream without synchronizing. Numerical values are `float` and
CSR indices are `std::int32_t`.

## Building the native library

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DSPRINGMV_BUILD_JAX=OFF
cmake --build build
```

The installed CMake target is `SpringMV::springmv`. The library links CUDA
Runtime and uses CUB headers for transpose preprocessing. It does not link
cuSPARSE.

## Kernel options

All row-reduction APIs accept `CsrKernelOptions`:

```cpp
springmv::CsrKernelOptions options{
    .max_nnz_per_row = max_row_length,
    .subwarp_size = 0,
};
```

`max_nnz_per_row` is scheduling metadata supplied by the caller. A
`subwarp_size` of zero selects automatically. Explicit widths of `1`, `2`,
`4`, `8`, `16`, and `32` are supported.

## Forward reductions

```cpp
#include <springmv/csr_spmv.hpp>
#include <springmv/csr_logsumexp.hpp>
#include <springmv/csr_tropical.hpp>
```

Weighted and unweighted arithmetic:

```cpp
springmv::csr_spmv(
    num_rows, d_row_offsets, d_column_indices,
    d_values, d_x, d_y, options, stream);

springmv::csr_spmv(
    num_rows, d_row_offsets, d_column_indices,
    d_x, d_y, options, stream);
```

Weighted and unweighted log-sum-exp:

```cpp
springmv::csr_logsumexp(
    num_rows, d_row_offsets, d_column_indices,
    d_log_values, d_log_x, d_log_y, options, stream);

springmv::csr_logsumexp(
    num_rows, d_row_offsets, d_column_indices,
    d_log_x, d_log_y, options, stream);
```

Weighted and unweighted tropical reductions:

```cpp
springmv::csr_minplus(
    num_rows, d_row_offsets, d_column_indices,
    d_edge_values, d_x, d_min_y, options, stream);
springmv::csr_minplus(
    num_rows, d_row_offsets, d_column_indices,
    d_x, d_min_y, options, stream);

springmv::csr_maxplus(
    num_rows, d_row_offsets, d_column_indices,
    d_edge_values, d_x, d_max_y, options, stream);
springmv::csr_maxplus(
    num_rows, d_row_offsets, d_column_indices,
    d_x, d_max_y, options, stream);
```

Unweighted overloads do not allocate or read an edge-value array.

## Batched reductions

```cpp
#include <springmv/csr_batched.hpp>
```

Batched operands and outputs are batch-major. A zero operand stride shares
that operand across the batch; nonzero strides are measured in elements:

```cpp
springmv::csr_logsumexp_batched(
    batch_size,
    num_rows,
    d_row_offsets,
    d_column_indices,
    d_log_values,
    0,                 // shared edge values
    d_log_x_batch,
    num_cols,
    d_log_y_batch,
    options,
    stream);
```

Arithmetic, min-plus, and max-plus provide matching weighted and unweighted
batched entry points in `csr_batched.hpp`.

## First-order derivatives

```cpp
#include <springmv/csr_spmv_backward.hpp>
#include <springmv/csr_logsumexp_backward.hpp>
```

Weighted arithmetic input VJP using transposed CSR:

```cpp
springmv::csr_spmv_input_vjp(
    num_cols,
    d_transposed_row_offsets,
    d_transposed_column_indices,
    d_edge_permutation,
    d_values,
    d_output_cotangent,
    d_input_cotangent,
    transpose_options,
    stream);
```

The atomic-scatter alternative traverses original CSR:

```cpp
springmv::csr_spmv_input_vjp_scatter(
    num_rows, num_cols,
    d_row_offsets, d_column_indices,
    d_values, d_output_cotangent, d_input_cotangent,
    options, stream);
```

Per-edge value VJP:

```cpp
springmv::csr_spmv_values_vjp(
    num_rows, d_row_offsets, d_column_indices,
    d_x, d_output_cotangent, d_values_cotangent,
    options, stream);
```

Weighted log-sum-exp JVP consumes the previously computed primal output:

```cpp
springmv::csr_logsumexp_jvp(
    num_rows,
    d_row_offsets,
    d_column_indices,
    d_log_values,
    d_log_x,
    d_log_y,
    d_log_values_tangent,
    d_log_x_tangent,
    d_log_y_tangent,
    options,
    stream);
```

Log-sum-exp provides transpose and scatter input VJPs plus a per-edge value
VJP in `csr_logsumexp_backward.hpp`.

## Sampling

```cpp
#include <springmv/csr_sampling.hpp>
```

The combined sampling API uses original CSR for Gumbel-max selection and
transposed CSR for atomic-free Boolean propagation. The caller supplies
Gumbels and selected-child scratch storage:

```cpp
springmv::csr_logsumexp_sample(
    num_rows,
    num_cols,
    d_row_offsets,
    d_column_indices,
    d_transposed_row_offsets,
    d_transposed_column_indices,
    d_log_values,
    d_gumbels,
    d_active_parent,
    d_selected_child_scratch,  // num_rows int32 entries
    d_active_child,
    options,
    transpose_options,
    stream);
```

`csr_logsumexp_sample_batched` supports independently shared or batched edge
weights, Gumbels, and parent activity through element strides. The selection
and propagation stages are also exposed separately for callers that manage the
pipeline themselves.

## CSR transpose

```cpp
#include <springmv/csr_transpose.hpp>
```

Transpose is an asynchronous preprocessing operation using caller-owned
workspace. Query the workspace requirement before launching it:

```cpp
std::size_t workspace_size{};
springmv::csr_transpose_workspace_size(
    num_rows,
    num_cols,
    num_nonzeros,
    &workspace_size);

springmv::csr_transpose(
    num_rows,
    num_cols,
    num_nonzeros,
    d_row_offsets,
    d_column_indices,
    d_values,
    d_transposed_row_offsets,
    d_transposed_column_indices,
    d_transposed_values,
    d_edge_permutation,
    d_workspace,
    workspace_size,
    stream);
```

The structure-only overload omits input/output values:

```cpp
springmv::csr_transpose(
    num_rows,
    num_cols,
    num_nonzeros,
    d_row_offsets,
    d_column_indices,
    d_transposed_row_offsets,
    d_transposed_column_indices,
    d_edge_permutation,
    d_workspace,
    workspace_size,
    stream);
```

The result has `num_cols` rows and `num_rows` columns. Entries within each
output row are deterministically ordered by original row. The optional edge
permutation maps each transposed entry back to its original nonzero position.

## Numerical identities

- Log-sum-exp empty rows and nonempty all-`-inf` rows produce `-inf`.
- Any `+inf` log-sum-exp term produces `+inf`; NaNs propagate.
- Min-plus empty rows produce `+inf`.
- Max-plus empty rows produce `-inf`.
- Both tropical operations propagate NaNs.
