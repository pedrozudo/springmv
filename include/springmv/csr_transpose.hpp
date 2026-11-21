// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace springmv {

/**
 * Return the temporary-storage requirement for csr_transpose.
 *
 * The returned storage can be reused between calls with equal or smaller
 * dimensions. No workspace is required when num_nonzeros is zero.
 */
cudaError_t csr_transpose_workspace_size(std::int32_t num_rows,
                                         std::int32_t num_cols,
                                         std::int32_t num_nonzeros,
                                         std::size_t* workspace_size) noexcept;

/**
 * Transpose an unweighted int32 CSR structure on a CUDA stream.
 *
 * The input is num_rows by num_cols. The output is CSR for the num_cols by
 * num_rows transpose. All arrays and workspace refer to device memory. Output
 * entries are ordered by their original row, so the result is deterministic.
 *
 * edge_permutation is optional. When provided, edge_permutation[t_nz] is the
 * input nonzero position corresponding to transposed entry t_nz.
 */
cudaError_t csr_transpose(
    std::int32_t num_rows, std::int32_t num_cols,
    std::int32_t num_nonzeros, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    std::int32_t* transposed_row_offsets,
    std::int32_t* transposed_column_indices,
    std::int32_t* edge_permutation, void* workspace,
    std::size_t workspace_size, cudaStream_t stream = nullptr) noexcept;

/**
 * Transpose a weighted float32/int32 CSR matrix on a CUDA stream.
 *
 * This overload also gathers values into transposed_values. Its structure and
 * optional edge_permutation have the same semantics as the unweighted form.
 */
cudaError_t csr_transpose(
    std::int32_t num_rows, std::int32_t num_cols,
    std::int32_t num_nonzeros, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* values,
    std::int32_t* transposed_row_offsets,
    std::int32_t* transposed_column_indices, float* transposed_values,
    std::int32_t* edge_permutation, void* workspace,
    std::size_t workspace_size, cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv

