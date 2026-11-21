// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "springmv/csr_common.hpp"

namespace springmv {

/**
 * Compute an unweighted CSR product over the min-plus semiring.
 *
 *     y[row] = min(x[column_indices[nz]])
 *
 * Empty rows produce +infinity and NaNs propagate.
 */
cudaError_t csr_minplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices, const float* x,
                        float* y, CsrKernelOptions options,
                        cudaStream_t stream = nullptr) noexcept;

/**
 * Compute a weighted CSR product over the min-plus semiring.
 *
 *     y[row] = min(edge_values[nz] + x[column_indices[nz]])
 *
 * Empty rows produce +infinity and NaNs propagate.
 */
cudaError_t csr_minplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices,
                        const float* edge_values, const float* x, float* y,
                        CsrKernelOptions options,
                        cudaStream_t stream = nullptr) noexcept;

/**
 * Compute an unweighted CSR product over the max-plus semiring.
 *
 *     y[row] = max(x[column_indices[nz]])
 *
 * Empty rows produce -infinity and NaNs propagate.
 */
cudaError_t csr_maxplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices, const float* x,
                        float* y, CsrKernelOptions options,
                        cudaStream_t stream = nullptr) noexcept;

/**
 * Compute a weighted CSR product over the max-plus semiring.
 *
 *     y[row] = max(edge_values[nz] + x[column_indices[nz]])
 *
 * Empty rows produce -infinity and NaNs propagate.
 */
cudaError_t csr_maxplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices,
                        const float* edge_values, const float* x, float* y,
                        CsrKernelOptions options,
                        cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv

