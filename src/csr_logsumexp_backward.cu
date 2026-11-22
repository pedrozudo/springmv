// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_logsumexp_backward.hpp"

#include "csr_input_vjp_scatter.cuh"
#include "csr_kernel_common.cuh"
#include "csr_logsumexp_policy.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace springmv {
namespace {

namespace cg = cooperative_groups;

template <bool weighted>
__device__ float edge_term(std::int32_t edge,
                           const std::int32_t* column_indices,
                           const float* log_values,
                           const float* log_x) noexcept
{
    const auto input = log_x[column_indices[edge]];
    if constexpr (weighted) {
        return log_values[edge] + input;
    }
    return input;
}

template <bool weighted, int subwarp_size, typename Tile>
__device__ std::int32_t row_positive_infinity_count(
    const Tile& tile, std::int32_t lane, std::int32_t row_begin,
    std::int32_t row_end, const std::int32_t* column_indices,
    const float* log_values, const float* log_x, float output) noexcept
{
    std::int32_t count{};
    if (output == CUDART_INF_F) {
        for (auto edge = row_begin + lane; edge < row_end;
             edge += subwarp_size) {
            count += edge_term<weighted>(edge, column_indices, log_values,
                                         log_x) == CUDART_INF_F;
        }
#pragma unroll
        for (int offset = subwarp_size / 2; offset > 0; offset /= 2) {
            count += tile.shfl_down(count, offset);
        }
        count = tile.shfl(count, 0);
    }
    return count;
}

struct LogSumExpInputVjpPolicy {
    using state_type = std::int32_t;

    template <int subwarp_size, typename Tile>
    __device__ static state_type prepare_row(
        const Tile& tile, std::int32_t lane, std::int32_t row,
        std::int32_t row_begin, std::int32_t row_end,
        const std::int32_t* column_indices, const float* log_values,
        const float* log_x, const float* output) noexcept
    {
        return row_positive_infinity_count<true, subwarp_size>(
            tile, lane, row_begin, row_end, column_indices, log_values, log_x,
            output[row]);
    }

    __device__ static float contribution(
        state_type infinity_count, std::int32_t row, std::int32_t edge,
        std::int32_t column, std::int32_t row_length,
        const float* log_values, const float* log_x, const float* output,
        const float* output_cotangent) noexcept
    {
        const auto term = log_values[edge] + log_x[column];
        const auto factor = detail::logsumexp_softmax_factor(
            term, output[row], row_length, infinity_count);
        return output_cotangent[row] * factor;
    }

    static bool valid_optional_inputs(const float* log_x,
                                      const float* output) noexcept
    {
        return log_x != nullptr && output != nullptr;
    }
};

template <bool weighted, int subwarp_size>
__global__ __launch_bounds__(detail::block_size) void jvp_kernel(
    std::int32_t num_rows, const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const float* __restrict__ log_values,
    const float* __restrict__ log_x, const float* __restrict__ output,
    const float* __restrict__ values_tangent,
    const float* __restrict__ input_tangent,
    float* __restrict__ output_tangent)
{
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto row = detail::initial_row<subwarp_size>();
    const auto stride = detail::row_stride<subwarp_size>();

    for (; row < num_rows; row += stride) {
        const auto row_begin = row_offsets[row];
        const auto row_end = row_offsets[row + 1];
        const auto row_length = row_end - row_begin;
        const auto row_output = output[row];
        const auto infinity_count =
            row_positive_infinity_count<weighted, subwarp_size>(
                tile, lane, row_begin, row_end, column_indices, log_values,
                log_x, row_output);

        float sum{};
        for (auto edge = row_begin + lane; edge < row_end;
             edge += subwarp_size) {
            const auto term = edge_term<weighted>(
                edge, column_indices, log_values, log_x);
            const auto factor = detail::logsumexp_softmax_factor(
                term, row_output, row_length, infinity_count);
            auto tangent = input_tangent[column_indices[edge]];
            if constexpr (weighted) {
                tangent += values_tangent[edge];
            }
            sum += factor * tangent;
        }
#pragma unroll
        for (int offset = subwarp_size / 2; offset > 0; offset /= 2) {
            sum += tile.shfl_down(sum, offset);
        }
        if (lane == 0) {
            output_tangent[row] = sum;
        }
    }
}

template <int subwarp_size>
__global__ __launch_bounds__(detail::block_size) void values_vjp_kernel(
    std::int32_t num_rows, const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const float* __restrict__ log_values,
    const float* __restrict__ log_x, const float* __restrict__ output,
    const float* __restrict__ output_cotangent,
    float* __restrict__ values_cotangent)
{
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto row = detail::initial_row<subwarp_size>();
    const auto stride = detail::row_stride<subwarp_size>();

    for (; row < num_rows; row += stride) {
        const auto row_begin = row_offsets[row];
        const auto row_end = row_offsets[row + 1];
        const auto row_length = row_end - row_begin;
        const auto row_output = output[row];
        const auto infinity_count =
            row_positive_infinity_count<true, subwarp_size>(
                tile, lane, row_begin, row_end, column_indices, log_values,
                log_x, row_output);
        for (auto edge = row_begin + lane; edge < row_end;
             edge += subwarp_size) {
            const auto term =
                edge_term<true>(edge, column_indices, log_values, log_x);
            const auto factor = detail::logsumexp_softmax_factor(
                term, row_output, row_length, infinity_count);
            values_cotangent[edge] = output_cotangent[row] * factor;
        }
    }
}

template <bool weighted>
__device__ std::int32_t scalar_positive_infinity_count(
    std::int32_t row_begin, std::int32_t row_end,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x) noexcept
{
    std::int32_t count{};
    for (auto edge = row_begin; edge < row_end; ++edge) {
        count += edge_term<weighted>(edge, column_indices, log_values, log_x) ==
                 CUDART_INF_F;
    }
    return count;
}

template <bool weighted, int subwarp_size>
__global__ __launch_bounds__(detail::block_size) void input_vjp_kernel(
    std::int32_t num_columns,
    const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const std::int32_t* __restrict__ transpose_row_offsets,
    const std::int32_t* __restrict__ transpose_column_indices,
    const std::int32_t* __restrict__ transpose_permutation,
    const float* __restrict__ log_values,
    const float* __restrict__ log_x, const float* __restrict__ output,
    const float* __restrict__ output_cotangent,
    float* __restrict__ input_cotangent)
{
    auto tile = cg::tiled_partition<subwarp_size>(cg::this_thread_block());
    const auto lane = static_cast<std::int32_t>(tile.thread_rank());
    auto column = detail::initial_row<subwarp_size>();
    const auto stride = detail::row_stride<subwarp_size>();

    for (; column < num_columns; column += stride) {
        float sum{};
        const auto transpose_end = transpose_row_offsets[column + 1];
        for (auto transpose_edge = transpose_row_offsets[column] + lane;
             transpose_edge < transpose_end;
             transpose_edge += subwarp_size) {
            const auto row = transpose_column_indices[transpose_edge];
            const auto row_begin = row_offsets[row];
            const auto row_end = row_offsets[row + 1];
            const auto row_output = output[row];
            const auto edge = weighted
                                  ? transpose_permutation[transpose_edge]
                                  : std::int32_t{};
            const auto term = weighted ? log_values[edge] + log_x[column]
                                       : log_x[column];
            auto infinity_count = std::int32_t{};
            if (row_output == CUDART_INF_F) {
                infinity_count = scalar_positive_infinity_count<weighted>(
                    row_begin, row_end, column_indices, log_values, log_x);
            }
            const auto factor = detail::logsumexp_softmax_factor(
                term, row_output, row_end - row_begin, infinity_count);
            sum += output_cotangent[row] * factor;
        }
#pragma unroll
        for (int offset = subwarp_size / 2; offset > 0; offset /= 2) {
            sum += tile.shfl_down(sum, offset);
        }
        if (lane == 0) {
            input_cotangent[column] = sum;
        }
    }
}

template <bool weighted, int subwarp_size>
cudaError_t launch_jvp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output, const float* values_tangent,
    const float* input_tangent, float* output_tangent,
    cudaStream_t stream) noexcept
{
    const auto grid = detail::grid_size<subwarp_size>(num_rows);
    jvp_kernel<weighted, subwarp_size>
        <<<grid, detail::block_size, 0, stream>>>(
            num_rows, row_offsets, column_indices, log_values, log_x, output,
            values_tangent, input_tangent, output_tangent);
    return cudaPeekAtLastError();
}

template <bool weighted, int subwarp_size>
cudaError_t launch_input_vjp(
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* transpose_permutation, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    cudaStream_t stream) noexcept
{
    const auto grid = detail::grid_size<subwarp_size>(num_columns);
    input_vjp_kernel<weighted, subwarp_size>
        <<<grid, detail::block_size, 0, stream>>>(
            num_columns, row_offsets, column_indices, transpose_row_offsets,
            transpose_column_indices, transpose_permutation, log_values,
            log_x, output, output_cotangent, input_cotangent);
    return cudaPeekAtLastError();
}

template <int subwarp_size>
cudaError_t launch_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* values_cotangent,
    cudaStream_t stream) noexcept
{
    const auto grid = detail::grid_size<subwarp_size>(num_rows);
    values_vjp_kernel<subwarp_size><<<grid, detail::block_size, 0, stream>>>(
        num_rows, row_offsets, column_indices, log_values, log_x, output,
        output_cotangent, values_cotangent);
    return cudaPeekAtLastError();
}

#define SPRINGMV_DISPATCH_SUBWARP(LAUNCH)                                    \
    switch (subwarp) {                                                       \
    case 1:                                                                 \
        return LAUNCH(1);                                                    \
    case 2:                                                                 \
        return LAUNCH(2);                                                    \
    case 4:                                                                 \
        return LAUNCH(4);                                                    \
    case 8:                                                                 \
        return LAUNCH(8);                                                    \
    case 16:                                                                \
        return LAUNCH(16);                                                   \
    case 32:                                                                \
        return LAUNCH(32);                                                   \
    default:                                                                \
        return cudaErrorInvalidValue;                                       \
    }

template <bool weighted>
cudaError_t dispatch_jvp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output, const float* values_tangent,
    const float* input_tangent, float* output_tangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
#define SPRINGMV_LAUNCH_JVP(size)                                            \
    launch_jvp<weighted, size>(num_rows, row_offsets, column_indices,         \
                               log_values, log_x, output, values_tangent,     \
                               input_tangent, output_tangent, stream)
    SPRINGMV_DISPATCH_SUBWARP(SPRINGMV_LAUNCH_JVP)
#undef SPRINGMV_LAUNCH_JVP
}

template <bool weighted>
cudaError_t dispatch_input_vjp(
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* transpose_permutation, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
#define SPRINGMV_LAUNCH_INPUT_VJP(size)                                      \
    launch_input_vjp<weighted, size>(                                        \
        num_columns, row_offsets, column_indices, transpose_row_offsets,      \
        transpose_column_indices, transpose_permutation, log_values, log_x,   \
        output, output_cotangent, input_cotangent, stream)
    SPRINGMV_DISPATCH_SUBWARP(SPRINGMV_LAUNCH_INPUT_VJP)
#undef SPRINGMV_LAUNCH_INPUT_VJP
}

cudaError_t dispatch_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* values_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
#define SPRINGMV_LAUNCH_VALUES_VJP(size)                                     \
    launch_values_vjp<size>(num_rows, row_offsets, column_indices,            \
                            log_values, log_x, output, output_cotangent,      \
                            values_cotangent, stream)
    SPRINGMV_DISPATCH_SUBWARP(SPRINGMV_LAUNCH_VALUES_VJP)
#undef SPRINGMV_LAUNCH_VALUES_VJP
}

#undef SPRINGMV_DISPATCH_SUBWARP

bool invalid_options(std::int32_t size, CsrKernelOptions options) noexcept
{
    return size < 0 || options.max_nnz_per_row < 0 ||
           !detail::valid_subwarp_size(options.subwarp_size);
}

}  // namespace

cudaError_t csr_logsumexp_jvp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_x,
    const float* output, const float* input_tangent, float* output_tangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    if (invalid_options(num_rows, options)) {
        return cudaErrorInvalidValue;
    }
    if (num_rows == 0) {
        return cudaSuccess;
    }
    if (row_offsets == nullptr || output == nullptr ||
        output_tangent == nullptr ||
        (options.max_nnz_per_row > 0 &&
         (column_indices == nullptr || log_x == nullptr ||
          input_tangent == nullptr))) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_jvp<false>(
        num_rows, row_offsets, column_indices, nullptr, log_x, output, nullptr,
        input_tangent, output_tangent, options, stream);
}

cudaError_t csr_logsumexp_jvp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output, const float* values_tangent,
    const float* input_tangent, float* output_tangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    if (invalid_options(num_rows, options)) {
        return cudaErrorInvalidValue;
    }
    if (num_rows == 0) {
        return cudaSuccess;
    }
    if (row_offsets == nullptr || output == nullptr ||
        output_tangent == nullptr ||
        (options.max_nnz_per_row > 0 &&
         (column_indices == nullptr || log_values == nullptr ||
          log_x == nullptr || values_tangent == nullptr ||
          input_tangent == nullptr))) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_jvp<true>(num_rows, row_offsets, column_indices,
                              log_values, log_x, output, values_tangent,
                              input_tangent, output_tangent, options, stream);
}

cudaError_t csr_logsumexp_input_vjp(
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices, const float* log_x,
    const float* output, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions transpose_options,
    cudaStream_t stream) noexcept
{
    if (invalid_options(num_columns, transpose_options)) {
        return cudaErrorInvalidValue;
    }
    if (num_columns == 0) {
        return cudaSuccess;
    }
    if (transpose_row_offsets == nullptr || input_cotangent == nullptr ||
        (transpose_options.max_nnz_per_row > 0 &&
         (row_offsets == nullptr || column_indices == nullptr ||
          transpose_column_indices == nullptr || log_x == nullptr ||
          output == nullptr || output_cotangent == nullptr))) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_input_vjp<false>(
        num_columns, row_offsets, column_indices, transpose_row_offsets,
        transpose_column_indices, nullptr, nullptr, log_x, output,
        output_cotangent, input_cotangent, transpose_options, stream);
}

cudaError_t csr_logsumexp_input_vjp(
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* transpose_permutation, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions transpose_options, cudaStream_t stream) noexcept
{
    if (invalid_options(num_columns, transpose_options)) {
        return cudaErrorInvalidValue;
    }
    if (num_columns == 0) {
        return cudaSuccess;
    }
    if (transpose_row_offsets == nullptr || input_cotangent == nullptr ||
        (transpose_options.max_nnz_per_row > 0 &&
         (row_offsets == nullptr || column_indices == nullptr ||
          transpose_column_indices == nullptr ||
          transpose_permutation == nullptr || log_values == nullptr ||
          log_x == nullptr || output == nullptr ||
          output_cotangent == nullptr))) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_input_vjp<true>(
        num_columns, row_offsets, column_indices, transpose_row_offsets,
        transpose_column_indices, transpose_permutation, log_values, log_x,
        output, output_cotangent, input_cotangent, transpose_options, stream);
}

cudaError_t csr_logsumexp_input_vjp_scatter(
    std::int32_t num_rows, std::int32_t num_columns,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_values, const float* log_x, const float* output,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return detail::input_vjp_scatter<LogSumExpInputVjpPolicy>(
        num_rows, num_columns, row_offsets, column_indices, log_values, log_x,
        output, output_cotangent, input_cotangent, options, stream);
}

cudaError_t csr_logsumexp_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* log_values,
    const float* log_x, const float* output,
    const float* output_cotangent, float* values_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    if (invalid_options(num_rows, options)) {
        return cudaErrorInvalidValue;
    }
    if (num_rows == 0 || options.max_nnz_per_row == 0) {
        return cudaSuccess;
    }
    if (row_offsets == nullptr || column_indices == nullptr ||
        log_values == nullptr || log_x == nullptr || output == nullptr ||
        output_cotangent == nullptr || values_cotangent == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_values_vjp(num_rows, row_offsets, column_indices,
                               log_values, log_x, output, output_cotangent,
                               values_cotangent, options, stream);
}

}  // namespace springmv
