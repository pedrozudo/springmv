// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// The row/subwarp scheduling is adapted from Ginkgo's classical CUDA/HIP CSR
// SpMV kernel at v1.11.0 (commit 4c77a5d826bd1597fd093f39c47688f8cc735690,
// common/cuda_hip/matrix/csr_kernels.template.cpp). The compile-time
// reduction-policy interface is a SpringMV addition.

#pragma once

#include "springmv/csr_common.hpp"

#include "csr_kernel_common.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

namespace springmv::detail {

namespace cg = cooperative_groups;

/**
 * Policy requirements:
 *
 * - state_type
 * - identity()
 * - accumulate_weighted(state, edge_value, x)
 * - accumulate_unweighted(state, x)
 * - combine(left, right)
 * - shuffle_down(tile, state, offset)
 * - finalize(state)
 */
template <typename Policy, bool weighted, bool permuted, int subwarp_size>
__global__ __launch_bounds__(block_size) void csr_reduce_kernel(
    std::int32_t num_rows, const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const float* __restrict__ edge_values, const float* __restrict__ x,
    float* __restrict__ y,
    const std::int32_t* __restrict__ edge_permutation)
{
    static_assert(!permuted || weighted);
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto row = initial_row<subwarp_size>();
    const auto stride = row_stride<subwarp_size>();

    for (; row < num_rows; row += stride) {
        auto state = Policy::identity();
        const auto row_end = row_offsets[row + 1];
        for (auto nz = row_offsets[row] + lane; nz < row_end;
             nz += subwarp_size) {
            const auto input = x[column_indices[nz]];
            if constexpr (weighted) {
                const auto edge = permuted ? edge_permutation[nz] : nz;
                state = Policy::accumulate_weighted(state, edge_values[edge],
                                                    input);
            } else {
                state = Policy::accumulate_unweighted(state, input);
            }
        }

#pragma unroll
        for (int offset = subwarp_size / 2; offset > 0; offset /= 2) {
            const auto right = Policy::shuffle_down(tile, state, offset);
            if (lane + offset < subwarp_size) {
                state = Policy::combine(state, right);
            }
        }
        if (lane == 0) {
            y[row] = Policy::finalize(state);
        }
    }
}

template <typename Policy, bool weighted, bool permuted, int subwarp_size>
cudaError_t launch_csr_reduce(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* edge_values,
    const float* x, float* y, cudaStream_t stream,
    const std::int32_t* edge_permutation) noexcept
{
    const auto grid = grid_size<subwarp_size>(num_rows);
    csr_reduce_kernel<Policy, weighted, permuted, subwarp_size>
        <<<grid, block_size, 0, stream>>>(num_rows, row_offsets,
                                          column_indices, edge_values, x, y,
                                          edge_permutation);
    return cudaPeekAtLastError();
}

template <typename Policy, bool weighted, bool permuted = false>
cudaError_t dispatch_csr_reduce(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* edge_values,
    const float* x, float* y, CsrKernelOptions options,
    cudaStream_t stream,
    const std::int32_t* edge_permutation = nullptr) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
    switch (subwarp) {
    case 1:
        return launch_csr_reduce<Policy, weighted, permuted, 1>(
            num_rows, row_offsets, column_indices, edge_values, x, y, stream,
            edge_permutation);
    case 2:
        return launch_csr_reduce<Policy, weighted, permuted, 2>(
            num_rows, row_offsets, column_indices, edge_values, x, y, stream,
            edge_permutation);
    case 4:
        return launch_csr_reduce<Policy, weighted, permuted, 4>(
            num_rows, row_offsets, column_indices, edge_values, x, y, stream,
            edge_permutation);
    case 8:
        return launch_csr_reduce<Policy, weighted, permuted, 8>(
            num_rows, row_offsets, column_indices, edge_values, x, y, stream,
            edge_permutation);
    case 16:
        return launch_csr_reduce<Policy, weighted, permuted, 16>(
            num_rows, row_offsets, column_indices, edge_values, x, y, stream,
            edge_permutation);
    case 32:
        return launch_csr_reduce<Policy, weighted, permuted, 32>(
            num_rows, row_offsets, column_indices, edge_values, x, y, stream,
            edge_permutation);
    default:
        return cudaErrorInvalidValue;
    }
}

template <typename Policy, bool weighted, bool permuted = false>
cudaError_t csr_reduce(std::int32_t num_rows,
                       const std::int32_t* row_offsets,
                       const std::int32_t* column_indices,
                       const float* edge_values, const float* x, float* y,
                       CsrKernelOptions options,
                       cudaStream_t stream,
                       const std::int32_t* edge_permutation = nullptr) noexcept
{
    if (num_rows < 0 || options.max_nnz_per_row < 0 ||
        !valid_subwarp_size(options.subwarp_size)) {
        return cudaErrorInvalidValue;
    }
    if (num_rows == 0) {
        return cudaSuccess;
    }
    if (row_offsets == nullptr || y == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (options.max_nnz_per_row > 0 &&
        (column_indices == nullptr || x == nullptr ||
         (weighted && edge_values == nullptr) ||
         (permuted && edge_permutation == nullptr))) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_csr_reduce<Policy, weighted, permuted>(
        num_rows, row_offsets, column_indices, edge_values, x, y, options,
        stream, edge_permutation);
}

}  // namespace springmv::detail

