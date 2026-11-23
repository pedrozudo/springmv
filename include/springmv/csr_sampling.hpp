// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "springmv/csr_common.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace springmv {

/** Select one child per CSR row using log_weights + supplied Gumbels.
 *
 * NaN scores are ignored. Empty rows and rows whose scores are all NaN write
 * -1. Equal non-NaN scores are resolved in favor of the lowest original CSR
 * edge index. The selected_child output has num_rows entries.
 */
cudaError_t csr_logsumexp_sample_select(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_weights,
    const float* gumbels, std::int32_t* selected_child,
    CsrKernelOptions options = {}, cudaStream_t stream = nullptr) noexcept;

/** Batched Gumbel-max selection with independently shared numeric operands.
 *
 * selected_child is batch-major with shape (batch_size, num_rows). A zero
 * batch stride shares that input across the batch; otherwise strides are
 * measured in elements.
 */
cudaError_t csr_logsumexp_sample_select_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_weights, std::int32_t weights_batch_stride,
    const float* gumbels, std::int32_t gumbels_batch_stride,
    std::int32_t* selected_child, CsrKernelOptions options = {},
    cudaStream_t stream = nullptr) noexcept;

/** Propagate sampled active parents to children using transposed CSR.
 *
 * Each child output is a row-owned Boolean OR. No atomics, edge values, or
 * transpose permutation are used.
 */
cudaError_t csr_sample_propagate(
    std::int32_t num_parents, std::int32_t num_children,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* selected_child, const bool* active_parent,
    bool* active_child, CsrKernelOptions transpose_options = {},
    cudaStream_t stream = nullptr) noexcept;

/** Batched transposed-CSR Boolean propagation.
 *
 * selected_child is always batch-major. A zero active_parent_batch_stride
 * shares parent activity across the batch. active_child is batch-major with
 * shape (batch_size, num_children).
 */
cudaError_t csr_sample_propagate_batched(
    std::int32_t batch_size, std::int32_t num_parents,
    std::int32_t num_children,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* selected_child, const bool* active_parent,
    std::int32_t active_parent_batch_stride, bool* active_child,
    CsrKernelOptions transpose_options = {},
    cudaStream_t stream = nullptr) noexcept;

/** Select and propagate one sample using caller-provided scratch storage. */
cudaError_t csr_logsumexp_sample(
    std::int32_t num_rows, std::int32_t num_children,
    const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const float* log_weights, const float* gumbels,
    const bool* active_parent, std::int32_t* selected_child_workspace,
    bool* active_child, CsrKernelOptions options = {},
    CsrKernelOptions transpose_options = {},
    cudaStream_t stream = nullptr) noexcept;

/** Batched selection and propagation using caller-provided scratch storage. */
cudaError_t csr_logsumexp_sample_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    std::int32_t num_children, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const float* log_weights, std::int32_t weights_batch_stride,
    const float* gumbels, std::int32_t gumbels_batch_stride,
    const bool* active_parent, std::int32_t active_parent_batch_stride,
    std::int32_t* selected_child_workspace, bool* active_child,
    CsrKernelOptions options = {}, CsrKernelOptions transpose_options = {},
    cudaStream_t stream = nullptr) noexcept;

}  // namespace springmv
