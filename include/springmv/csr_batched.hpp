// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "springmv/csr_common.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace springmv {

/** Batched CSR reductions with batch-major dense vectors and outputs.
 *
 * A zero input or edge-value batch stride shares that operand across every
 * batch member. Otherwise the stride is measured in float elements.
 */
cudaError_t csr_spmv_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* x, std::int32_t x_batch_stride, float* y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_spmv_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* values, std::int32_t values_batch_stride, const float* x,
    std::int32_t x_batch_stride, float* y, CsrKernelOptions options = {},
    cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_logsumexp_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_x, std::int32_t x_batch_stride, float* log_y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_logsumexp_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_values, std::int32_t values_batch_stride,
    const float* log_x, std::int32_t x_batch_stride, float* log_y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_minplus_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* x, std::int32_t x_batch_stride, float* y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_minplus_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* edge_values, std::int32_t values_batch_stride,
    const float* x, std::int32_t x_batch_stride, float* y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_maxplus_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* x, std::int32_t x_batch_stride, float* y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

cudaError_t csr_maxplus_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* edge_values, std::int32_t values_batch_stride,
    const float* x, std::int32_t x_batch_stride, float* y,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv
