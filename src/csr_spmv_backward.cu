// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_spmv_backward.hpp"

#include "csr_arithmetic_policy.cuh"
#include "csr_input_vjp_scatter.cuh"
#include "csr_kernel_common.cuh"
#include "csr_reduce.cuh"

#include <cstdint>

namespace springmv {
namespace {

template <bool weighted>
struct ArithmeticInputVjpPolicy {
    using state_type = std::int32_t;

    template <int subwarp_size, typename Tile>
    __device__ static state_type prepare_row(
        const Tile&, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
        const std::int32_t*, const float*, const float*,
        const float*) noexcept
    {
        return {};
    }

    __device__ static float contribution(
        state_type, std::int32_t row, std::int32_t edge, std::int32_t,
        std::int32_t, const float* values, const float*, const float*,
        const float* output_cotangent) noexcept
    {
        if constexpr (weighted) {
            return values[edge] * output_cotangent[row];
        }
        return output_cotangent[row];
    }

    static bool valid_edge_values(const float* values) noexcept
    {
        return !weighted || values != nullptr;
    }

    static bool valid_optional_inputs(const float*, const float*) noexcept
    {
        return true;
    }
};

template <int subwarp_size>
__global__ __launch_bounds__(detail::block_size) void values_vjp_kernel(
    std::int32_t num_rows, const std::int32_t* __restrict__ row_offsets,
    const std::int32_t* __restrict__ column_indices,
    const float* __restrict__ x,
    const float* __restrict__ output_cotangent,
    float* __restrict__ values_cotangent)
{
    const auto lane = static_cast<std::int32_t>(threadIdx.x) % subwarp_size;
    auto row = detail::initial_row<subwarp_size>();
    const auto stride = detail::row_stride<subwarp_size>();

    for (; row < num_rows; row += stride) {
        const auto row_end = row_offsets[row + 1];
        for (auto nz = row_offsets[row] + lane; nz < row_end;
             nz += subwarp_size) {
            values_cotangent[nz] =
                output_cotangent[row] * x[column_indices[nz]];
        }
    }
}

template <int subwarp_size>
cudaError_t launch_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* x,
    const float* output_cotangent, float* values_cotangent,
    cudaStream_t stream) noexcept
{
    const auto grid = detail::grid_size<subwarp_size>(num_rows);
    values_vjp_kernel<subwarp_size><<<grid, detail::block_size, 0, stream>>>(
        num_rows, row_offsets, column_indices, x, output_cotangent,
        values_cotangent);
    return cudaPeekAtLastError();
}

cudaError_t dispatch_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* x,
    const float* output_cotangent, float* values_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    const auto subwarp = options.subwarp_size == 0
                             ? select_subwarp_size(options.max_nnz_per_row)
                             : options.subwarp_size;
    switch (subwarp) {
    case 1:
        return launch_values_vjp<1>(num_rows, row_offsets, column_indices, x,
                                    output_cotangent, values_cotangent, stream);
    case 2:
        return launch_values_vjp<2>(num_rows, row_offsets, column_indices, x,
                                    output_cotangent, values_cotangent, stream);
    case 4:
        return launch_values_vjp<4>(num_rows, row_offsets, column_indices, x,
                                    output_cotangent, values_cotangent, stream);
    case 8:
        return launch_values_vjp<8>(num_rows, row_offsets, column_indices, x,
                                    output_cotangent, values_cotangent, stream);
    case 16:
        return launch_values_vjp<16>(
            num_rows, row_offsets, column_indices, x, output_cotangent,
            values_cotangent, stream);
    case 32:
        return launch_values_vjp<32>(
            num_rows, row_offsets, column_indices, x, output_cotangent,
            values_cotangent, stream);
    default:
        return cudaErrorInvalidValue;
    }
}

}  // namespace

cudaError_t csr_spmv_input_vjp(
    std::int32_t num_columns,
    const std::int32_t* transpose_row_offsets,
    const std::int32_t* transpose_column_indices,
    const std::int32_t* transpose_permutation, const float* values,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions transpose_options, cudaStream_t stream) noexcept
{
    return detail::csr_reduce<detail::ArithmeticPolicy, true, true>(
        num_columns, transpose_row_offsets, transpose_column_indices, values,
        output_cotangent, input_cotangent, transpose_options, stream,
        transpose_permutation);
}

cudaError_t csr_spmv_input_vjp_scatter(
    std::int32_t num_rows, std::int32_t num_columns,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* values,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return detail::input_vjp_scatter<ArithmeticInputVjpPolicy<true>>(
        num_rows, num_columns, row_offsets, column_indices, values, nullptr,
        nullptr, output_cotangent, input_cotangent, options, stream);
}

cudaError_t csr_spmv_input_vjp_scatter(
    std::int32_t num_rows, std::int32_t num_columns,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* output_cotangent, float* input_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return detail::input_vjp_scatter<ArithmeticInputVjpPolicy<false>>(
        num_rows, num_columns, row_offsets, column_indices, nullptr, nullptr,
        nullptr, output_cotangent, input_cotangent, options, stream);
}

cudaError_t csr_spmv_input_vjp_scatter_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    std::int32_t num_columns, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* output_cotangent,
    float* input_cotangent, CsrKernelOptions options,
    cudaStream_t stream) noexcept
{
    return detail::input_vjp_scatter_batched<
        ArithmeticInputVjpPolicy<false>>(
        batch_size, num_rows, num_columns, row_offsets, column_indices,
        nullptr, nullptr, nullptr, output_cotangent, input_cotangent, options,
        stream);
}

cudaError_t csr_spmv_values_vjp(
    std::int32_t num_rows, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* x,
    const float* output_cotangent, float* values_cotangent,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    if (num_rows < 0 || options.max_nnz_per_row < 0 ||
        !detail::valid_subwarp_size(options.subwarp_size)) {
        return cudaErrorInvalidValue;
    }
    if (num_rows == 0 || options.max_nnz_per_row == 0) {
        return cudaSuccess;
    }
    if (row_offsets == nullptr || column_indices == nullptr || x == nullptr ||
        output_cotangent == nullptr || values_cotangent == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    return dispatch_values_vjp(num_rows, row_offsets, column_indices, x,
                               output_cotangent, values_cotangent, options,
                               stream);
}

}  // namespace springmv
