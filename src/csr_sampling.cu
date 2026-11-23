// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_sampling.hpp"

#include "csr_kernel_common.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace springmv {
namespace {

namespace cg = cooperative_groups;

struct Candidate {
    float score;
    std::int32_t edge;
};

__device__ Candidate better_candidate(Candidate left,
                                      Candidate right) noexcept
{
    if (right.edge < 0) {
        return left;
    }
    if (left.edge < 0 || right.score > left.score ||
        (right.score == left.score && right.edge < left.edge)) {
        return right;
    }
    return left;
}

template <int subwarp_size>
__global__ __launch_bounds__(detail::block_size) void select_kernel(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const float* __restrict__ log_weights,
    std::int32_t weights_batch_stride,
    const float* __restrict__ gumbels,
    std::int32_t gumbels_batch_stride,
    std::int32_t* __restrict__ selected_child)
{
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto work = detail::initial_row<subwarp_size>();
    const auto stride = detail::row_stride<subwarp_size>();
    const auto total_rows = batch_size * num_rows;

    for (; work < total_rows; work += stride) {
        const auto batch = work / num_rows;
        const auto row = work - batch * num_rows;
        const auto* batch_weights = log_weights;
        const auto* batch_gumbels = gumbels;
        if (weights_batch_stride != 0) {
            batch_weights += static_cast<std::int64_t>(batch) *
                             weights_batch_stride;
        }
        if (gumbels_batch_stride != 0) {
            batch_gumbels += static_cast<std::int64_t>(batch) *
                             gumbels_batch_stride;
        }
        Candidate best{-CUDART_INF_F, -1};
        const auto row_end = row_offsets[row + 1];
        for (auto edge = row_offsets[row] + lane; edge < row_end;
             edge += subwarp_size) {
            const auto score = batch_weights[edge] + batch_gumbels[edge];
            if (!isnan(score)) {
                best = better_candidate(best, Candidate{score, edge});
            }
        }

#pragma unroll
        for (int offset = subwarp_size / 2; offset > 0; offset /= 2) {
            Candidate right{tile.shfl_down(best.score, offset),
                            tile.shfl_down(best.edge, offset)};
            if (lane + offset < subwarp_size) {
                best = better_candidate(best, right);
            }
        }
        if (lane == 0) {
            selected_child[work] =
                best.edge < 0 ? -1 : column_indices[best.edge];
        }
    }
}

template <int subwarp_size>
__global__ __launch_bounds__(detail::block_size) void propagate_kernel(
    std::int32_t batch_size, std::int32_t num_parents,
    std::int32_t num_children,
    const std::int32_t* __restrict__ transpose_row_offsets,
    const std::int32_t* __restrict__ transpose_column_indices,
    const std::int32_t* __restrict__ selected_child,
    const bool* __restrict__ active_parent,
    std::int32_t active_parent_batch_stride,
    bool* __restrict__ active_child)
{
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto work = detail::initial_row<subwarp_size>();
    const auto stride = detail::row_stride<subwarp_size>();
    const auto total_rows = batch_size * num_children;

    for (; work < total_rows; work += stride) {
        const auto batch = work / num_children;
        const auto child = work - batch * num_children;
        const auto* batch_selected = selected_child;
        const auto* batch_active = active_parent;
        if (num_parents != 0) {
            batch_selected +=
                static_cast<std::int64_t>(batch) * num_parents;
        }
        if (active_parent_batch_stride != 0) {
            batch_active += static_cast<std::int64_t>(batch) *
                            active_parent_batch_stride;
        }
        std::int32_t active{};
        const auto row_end = transpose_row_offsets[child + 1];
        for (auto edge = transpose_row_offsets[child] + lane; edge < row_end;
             edge += subwarp_size) {
            const auto parent = transpose_column_indices[edge];
            active |= static_cast<std::int32_t>(
                batch_active[parent] && batch_selected[parent] == child);
        }

#pragma unroll
        for (int offset = subwarp_size / 2; offset > 0; offset /= 2) {
            const auto right = tile.shfl_down(active, offset);
            if (lane + offset < subwarp_size) {
                active |= right;
            }
        }
        if (lane == 0) {
            active_child[work] = active != 0;
        }
    }
}

template <int subwarp_size>
cudaError_t launch_select(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_weights, std::int32_t weights_batch_stride,
    const float* gumbels, std::int32_t gumbels_batch_stride,
    std::int32_t* selected_child, cudaStream_t stream) noexcept
{
    const auto total_rows = batch_size * num_rows;
    const auto grid = detail::grid_size<subwarp_size>(total_rows);
    select_kernel<subwarp_size><<<grid, detail::block_size, 0, stream>>>(
        batch_size, num_rows, row_offsets, column_indices, log_weights,
        weights_batch_stride, gumbels, gumbels_batch_stride, selected_child);
    return cudaPeekAtLastError();
}

template <int subwarp_size>
cudaError_t launch_propagate(
    std::int32_t batch_size, std::int32_t num_parents,
    std::int32_t num_children,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* selected_child, const bool* active_parent,
    std::int32_t active_parent_batch_stride, bool* active_child,
    cudaStream_t stream) noexcept
{
    const auto total_rows = batch_size * num_children;
    const auto grid = detail::grid_size<subwarp_size>(total_rows);
    propagate_kernel<subwarp_size><<<grid, detail::block_size, 0, stream>>>(
        batch_size, num_parents, num_children, transpose_row_offsets,
        transpose_column_indices, selected_child, active_parent,
        active_parent_batch_stride, active_child);
    return cudaPeekAtLastError();
}

cudaError_t dispatch_select(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_weights, std::int32_t weights_batch_stride,
    const float* gumbels, std::int32_t gumbels_batch_stride,
    std::int32_t* selected_child, CsrKernelOptions options,
    cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
#define SPRINGMV_SELECT(size)                                                \
    launch_select<size>(                                                     \
        batch_size, num_rows, row_offsets, column_indices, log_weights,      \
        weights_batch_stride, gumbels, gumbels_batch_stride, selected_child, \
        stream)
    switch (subwarp) {
    case 1:
        return SPRINGMV_SELECT(1);
    case 2:
        return SPRINGMV_SELECT(2);
    case 4:
        return SPRINGMV_SELECT(4);
    case 8:
        return SPRINGMV_SELECT(8);
    case 16:
        return SPRINGMV_SELECT(16);
    case 32:
        return SPRINGMV_SELECT(32);
    default:
        return cudaErrorInvalidValue;
    }
#undef SPRINGMV_SELECT
}

cudaError_t dispatch_propagate(
    std::int32_t batch_size, std::int32_t num_parents,
    std::int32_t num_children,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* selected_child, const bool* active_parent,
    std::int32_t active_parent_batch_stride, bool* active_child,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
#define SPRINGMV_PROPAGATE(size)                                             \
    launch_propagate<size>(                                                  \
        batch_size, num_parents, num_children, transpose_row_offsets,        \
        transpose_column_indices, selected_child, active_parent,             \
        active_parent_batch_stride, active_child, stream)
    switch (subwarp) {
    case 1:
        return SPRINGMV_PROPAGATE(1);
    case 2:
        return SPRINGMV_PROPAGATE(2);
    case 4:
        return SPRINGMV_PROPAGATE(4);
    case 8:
        return SPRINGMV_PROPAGATE(8);
    case 16:
        return SPRINGMV_PROPAGATE(16);
    case 32:
        return SPRINGMV_PROPAGATE(32);
    default:
        return cudaErrorInvalidValue;
    }
#undef SPRINGMV_PROPAGATE
}

bool invalid_work_size(std::int32_t batch_size,
                       std::int32_t rows) noexcept
{
    return batch_size > 0 && rows > 0 &&
           batch_size > std::numeric_limits<std::int32_t>::max() / rows;
}

}  // namespace

cudaError_t csr_logsumexp_sample_select(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_weights,
    const float* gumbels, std::int32_t* selected_child,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return csr_logsumexp_sample_select_batched(
        1, num_rows, row_offsets, column_indices, log_weights, 0, gumbels, 0,
        selected_child, options, stream);
}

cudaError_t csr_logsumexp_sample_select_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_weights, std::int32_t weights_batch_stride,
    const float* gumbels, std::int32_t gumbels_batch_stride,
    std::int32_t* selected_child, CsrKernelOptions options,
    cudaStream_t stream) noexcept
{
    if (batch_size < 0 || num_rows < 0 || weights_batch_stride < 0 ||
        gumbels_batch_stride < 0 || options.max_nnz_per_row < 0 ||
        !detail::valid_subwarp_size(options.subwarp_size) ||
        invalid_work_size(batch_size, num_rows)) {
        return cudaErrorInvalidValue;
    }
    if (batch_size == 0 || num_rows == 0) {
        return cudaSuccess;
    }
    if (row_offsets == nullptr || selected_child == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (options.max_nnz_per_row > 0 &&
        (column_indices == nullptr || log_weights == nullptr ||
         gumbels == nullptr)) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_select(
        batch_size, num_rows, row_offsets, column_indices, log_weights,
        weights_batch_stride, gumbels, gumbels_batch_stride, selected_child,
        options, stream);
}

cudaError_t csr_sample_propagate(
    std::int32_t num_parents, std::int32_t num_children,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* selected_child, const bool* active_parent,
    bool* active_child, CsrKernelOptions transpose_options,
    cudaStream_t stream) noexcept
{
    return csr_sample_propagate_batched(
        1, num_parents, num_children, transpose_row_offsets,
        transpose_column_indices, selected_child, active_parent, 0,
        active_child, transpose_options, stream);
}

cudaError_t csr_sample_propagate_batched(
    std::int32_t batch_size, std::int32_t num_parents,
    std::int32_t num_children,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* selected_child, const bool* active_parent,
    std::int32_t active_parent_batch_stride, bool* active_child,
    CsrKernelOptions transpose_options, cudaStream_t stream) noexcept
{
    if (batch_size < 0 || num_parents < 0 || num_children < 0 ||
        active_parent_batch_stride < 0 ||
        transpose_options.max_nnz_per_row < 0 ||
        !detail::valid_subwarp_size(transpose_options.subwarp_size) ||
        invalid_work_size(batch_size, num_parents) ||
        invalid_work_size(batch_size, num_children)) {
        return cudaErrorInvalidValue;
    }
    if (batch_size == 0 || num_children == 0) {
        return cudaSuccess;
    }
    if (transpose_row_offsets == nullptr || active_child == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (num_parents > 0 &&
        (selected_child == nullptr || active_parent == nullptr)) {
        return cudaErrorInvalidDevicePointer;
    }
    if (transpose_options.max_nnz_per_row > 0 &&
        transpose_column_indices == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_propagate(
        batch_size, num_parents, num_children, transpose_row_offsets,
        transpose_column_indices, selected_child, active_parent,
        active_parent_batch_stride, active_child, transpose_options, stream);
}

cudaError_t csr_logsumexp_sample(
    std::int32_t num_rows, std::int32_t num_children,
    const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const float* log_weights, const float* gumbels,
    const bool* active_parent, std::int32_t* selected_child_workspace,
    bool* active_child, CsrKernelOptions options,
    CsrKernelOptions transpose_options, cudaStream_t stream) noexcept
{
    return csr_logsumexp_sample_batched(
        1, num_rows, num_children, row_offsets, column_indices,
        transpose_row_offsets, transpose_column_indices, log_weights, 0,
        gumbels, 0, active_parent, 0, selected_child_workspace, active_child,
        options, transpose_options, stream);
}

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
    CsrKernelOptions options, CsrKernelOptions transpose_options,
    cudaStream_t stream) noexcept
{
    auto status = csr_logsumexp_sample_select_batched(
        batch_size, num_rows, row_offsets, column_indices, log_weights,
        weights_batch_stride, gumbels, gumbels_batch_stride,
        selected_child_workspace, options, stream);
    if (status != cudaSuccess) {
        return status;
    }
    return csr_sample_propagate_batched(
        batch_size, num_rows, num_children, transpose_row_offsets,
        transpose_column_indices, selected_child_workspace, active_parent,
        active_parent_batch_stride, active_child, transpose_options, stream);
}

}  // namespace springmv
