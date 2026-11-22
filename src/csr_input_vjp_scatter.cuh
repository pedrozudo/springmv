// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "springmv/csr_common.hpp"

#include "csr_kernel_common.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace springmv::detail {

namespace cg = cooperative_groups;

/**
 * Policy requirements:
 *
 * - state_type
 * - prepare_row<subwarp_size>(tile, lane, row, row_begin, row_end,
 *   column_indices, edge_values, x, output)
 * - contribution(state, row, edge, column, row_length, edge_values, x,
 *   output, output_cotangent)
 */
template <typename Policy, int subwarp_size>
__global__ __launch_bounds__(block_size) void input_vjp_scatter_kernel(
    std::int32_t num_rows, const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const float* __restrict__ edge_values, const float* __restrict__ x,
    const float* __restrict__ output,
    const float* __restrict__ output_cotangent,
    float* __restrict__ input_cotangent)
{
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto row = initial_row<subwarp_size>();
    const auto stride = row_stride<subwarp_size>();

    for (; row < num_rows; row += stride) {
        const auto row_begin = row_offsets[row];
        const auto row_end = row_offsets[row + 1];
        const auto state = Policy::template prepare_row<subwarp_size>(
            tile, lane, row, row_begin, row_end, column_indices, edge_values,
            x, output);
        for (auto edge = row_begin + lane; edge < row_end;
             edge += subwarp_size) {
            const auto column = column_indices[edge];
            const auto value = Policy::contribution(
                state, row, edge, column, row_end - row_begin, edge_values, x,
                output, output_cotangent);
            atomicAdd(&input_cotangent[column], value);
        }
    }
}

template <typename Policy, int subwarp_size>
cudaError_t launch_input_vjp_scatter(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* edge_values,
    const float* x, const float* output, const float* output_cotangent,
    float* input_cotangent, cudaStream_t stream) noexcept
{
    const auto grid = grid_size<subwarp_size>(num_rows);
    input_vjp_scatter_kernel<Policy, subwarp_size>
        <<<grid, block_size, 0, stream>>>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent);
    return cudaPeekAtLastError();
}

template <typename Policy>
cudaError_t dispatch_input_vjp_scatter(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* edge_values,
    const float* x, const float* output, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions options,
    cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
    switch (subwarp) {
    case 1:
        return launch_input_vjp_scatter<Policy, 1>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent, stream);
    case 2:
        return launch_input_vjp_scatter<Policy, 2>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent, stream);
    case 4:
        return launch_input_vjp_scatter<Policy, 4>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent, stream);
    case 8:
        return launch_input_vjp_scatter<Policy, 8>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent, stream);
    case 16:
        return launch_input_vjp_scatter<Policy, 16>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent, stream);
    case 32:
        return launch_input_vjp_scatter<Policy, 32>(
            num_rows, row_offsets, column_indices, edge_values, x, output,
            output_cotangent, input_cotangent, stream);
    default:
        return cudaErrorInvalidValue;
    }
}

template <typename Policy>
cudaError_t input_vjp_scatter(
    std::int32_t num_rows, std::int32_t num_columns,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* edge_values, const float* x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    if (num_rows < 0 || num_columns < 0 || options.max_nnz_per_row < 0 ||
        !valid_subwarp_size(options.subwarp_size)) {
        return cudaErrorInvalidValue;
    }
    if (num_columns > 0 && input_cotangent == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (num_rows > 0 && row_offsets == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (options.max_nnz_per_row > 0 &&
        (column_indices == nullptr || edge_values == nullptr ||
         output_cotangent == nullptr ||
         !Policy::valid_optional_inputs(x, output))) {
        return cudaErrorInvalidDevicePointer;
    }

    if (num_columns > 0) {
        const auto status = cudaMemsetAsync(
            input_cotangent, 0,
            static_cast<std::size_t>(num_columns) * sizeof(float), stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (num_rows == 0 || options.max_nnz_per_row == 0) {
        return cudaSuccess;
    }
    return dispatch_input_vjp_scatter<Policy>(
        num_rows, row_offsets, column_indices, edge_values, x, output,
        output_cotangent, input_cotangent, options, stream);
}

}  // namespace springmv::detail
