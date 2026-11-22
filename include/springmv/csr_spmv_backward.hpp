// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "springmv/csr_common.hpp"

namespace springmv {

/**
 * Compute the input VJP for weighted arithmetic SpMV using transposed CSR.
 *
 * For y = A * x and output cotangent g, this computes
 *
 *     input_cotangent = A^T * g.
 *
 * transpose_permutation[t_nz] maps each transposed nonzero to its position in
 * the original values array. No reordered-value temporary is required.
 */
cudaError_t csr_spmv_input_vjp(
    std::int32_t num_columns,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* transpose_permutation, const float* values,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions transpose_options,
    cudaStream_t stream = nullptr) noexcept;

/**
 * Compute the edge-value VJP for weighted arithmetic SpMV.
 *
 * For an edge nz in row i and column j, this computes
 *
 *     values_cotangent[nz] = output_cotangent[i] * x[j].
 */
cudaError_t csr_spmv_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* x,
    const float* output_cotangent, float* values_cotangent,
    CsrKernelOptions options, cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv
