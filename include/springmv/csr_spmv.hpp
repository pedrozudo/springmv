// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "springmv/csr_common.hpp"

namespace springmv {

/**
 * Compute an unweighted arithmetic CSR product.
 *
 * For each row, this computes
 *
 *     y[row] = sum(x[column_indices[nz]]).
 *
 * This overload does not read or store an edge-value array. Empty rows
 * produce zero.
 */
cudaError_t csr_spmv(std::int32_t num_rows,
                     const std::int32_t* row_offsets,
                     const std::int32_t* column_indices, const float* x,
                     float* y, CsrKernelOptions options,
                     cudaStream_t stream = nullptr) noexcept;

/**
 * Compute y = A * x for a float32/int32 matrix in CSR format.
 *
 * All pointers refer to device memory. The operation is enqueued on stream
 * and does not allocate memory or synchronize. The CSR uses zero-based column
 * indices and row_offsets contains num_rows + 1 entries. Duplicate column
 * indices are permitted.
 *
 * When num_rows is zero, pointers may be null. If max_nnz_per_row is zero,
 * column_indices, values, and x may be null, and the output is filled with
 * zeros.
 */
cudaError_t csr_spmv(std::int32_t num_rows,
                     const std::int32_t* row_offsets,
                     const std::int32_t* column_indices, const float* values,
                     const float* x, float* y, CsrKernelOptions options,
                     cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv

