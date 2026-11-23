// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "springmv/csr_common.hpp"

namespace springmv {

/**
 * Derivative convention for extended-real rows:
 *
 * - k positive-infinity terms each receive softmax weight 1 / k;
 * - an all-negative-infinity nonempty row assigns 1 / row_length per term;
 * - an empty row has zero JVP and contributes nothing to a VJP;
 * - a NaN term makes all derivatives associated with that row NaN.
 */

/** Compute the JVP of an unweighted log-sum-exp CSR product. */
cudaError_t csr_logsumexp_jvp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_x,
    const float* output, const float* input_tangent, float* output_tangent,
    CsrKernelOptions options, cudaStream_t stream = nullptr) noexcept;

/** Compute batched JVPs of unweighted log-sum-exp CSR products. */
cudaError_t csr_logsumexp_jvp_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_x,
    const float* output, const float* input_tangent,
    float* output_tangent, CsrKernelOptions options,
    cudaStream_t stream = nullptr) noexcept;

/** Compute the JVP of a weighted log-sum-exp CSR product. */
cudaError_t csr_logsumexp_jvp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output, const float* values_tangent,
    const float* input_tangent, float* output_tangent,
    CsrKernelOptions options, cudaStream_t stream = nullptr) noexcept;

/** Compute the input VJP of an unweighted log-sum-exp CSR product. */
cudaError_t csr_logsumexp_input_vjp(
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices, const float* log_x,
    const float* output, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions transpose_options,
    cudaStream_t stream = nullptr) noexcept;

/** Compute batched unweighted input VJPs using transposed CSR. */
cudaError_t csr_logsumexp_input_vjp_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices, const float* log_x,
    const float* output, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions transpose_options,
    cudaStream_t stream = nullptr) noexcept;

/** Compute the input VJP of a weighted log-sum-exp CSR product. */
cudaError_t csr_logsumexp_input_vjp(
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* transpose_permutation, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions transpose_options,
    cudaStream_t stream = nullptr) noexcept;

/**
 * Compute the weighted input VJP by scattering original CSR edges.
 *
 * The output is initialized to zero before contributions are accumulated with
 * atomic additions. Floating-point accumulation order is not deterministic.
 */
cudaError_t csr_logsumexp_input_vjp_scatter(
    std::int32_t num_rows, std::int32_t num_columns,
    const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions options, cudaStream_t stream = nullptr) noexcept;

/** Compute an unweighted input VJP by scattering original CSR edges. */
cudaError_t csr_logsumexp_input_vjp_scatter(
    std::int32_t num_rows, std::int32_t num_columns,
    const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_x,
    const float* output, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions options,
    cudaStream_t stream = nullptr) noexcept;

/** Compute batched unweighted input VJPs by atomic scatter. */
cudaError_t csr_logsumexp_input_vjp_scatter_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_x,
    const float* output, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions options,
    cudaStream_t stream = nullptr) noexcept;

/** Compute the edge-log-value VJP of a weighted log-sum-exp product. */
cudaError_t csr_logsumexp_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* values_cotangent,
    CsrKernelOptions options, cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv
