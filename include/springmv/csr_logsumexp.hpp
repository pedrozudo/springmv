// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "springmv/csr_common.hpp"

namespace springmv {

/**
 * Compute an unweighted CSR matrix-vector product over the log semiring.
 *
 * For each row, this computes
 *
 *     y[row] = logsumexp(x[column_indices[nz]]).
 *
 * This overload does not read or store an edge-value array. Empty rows
 * produce -infinity, +infinity terms produce +infinity, and NaNs propagate.
 */
cudaError_t csr_logsumexp(std::int32_t num_rows,
                          const std::int32_t* row_offsets,
                          const std::int32_t* column_indices, const float* x,
                          float* y, CsrKernelOptions options,
                          cudaStream_t stream = nullptr) noexcept;

/**
 * Compute a weighted CSR matrix-vector product over the log semiring.
 *
 * For each row, this computes
 *
 *     y[row] = logsumexp(log_values[nz] + x[column_indices[nz]]).
 *
 * All pointers refer to device memory. The operation is enqueued on stream
 * and does not allocate memory or synchronize. Empty rows and rows containing
 * only -infinity produce -infinity. A +infinity term produces +infinity.
 * NaN terms propagate to the row output. Duplicate column indices represent
 * distinct terms and are permitted.
 *
 * When num_rows is zero, pointers may be null. If max_nnz_per_row is zero,
 * column_indices, log_values, and x may be null.
 */
cudaError_t csr_logsumexp(std::int32_t num_rows,
                          const std::int32_t* row_offsets,
                          const std::int32_t* column_indices,
                          const float* log_values, const float* x, float* y,
                          CsrKernelOptions options,
                          cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv

