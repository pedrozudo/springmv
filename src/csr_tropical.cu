// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// These operations use the classical CSR scheduling adapted from Ginkgo
// v1.11.0 through SpringMV's shared reduction engine. The tropical policies
// are SpringMV additions. Full provenance is recorded in csr_reduce.cuh.

#include "springmv/csr_tropical.hpp"

#include "csr_reduce.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace springmv {
namespace {

template <bool take_minimum>
struct TropicalPolicy {
    using state_type = float;

    __device__ static state_type identity() noexcept
    {
        if constexpr (take_minimum) {
            return CUDART_INF_F;
        } else {
            return -CUDART_INF_F;
        }
    }

    __device__ static state_type combine(state_type left,
                                         state_type right) noexcept
    {
        if (isnan(left) || isnan(right)) {
            return CUDART_NAN_F;
        }
        if constexpr (take_minimum) {
            return fminf(left, right);
        } else {
            return fmaxf(left, right);
        }
    }

    __device__ static state_type accumulate_weighted(
        state_type state, float edge_value, float input) noexcept
    {
        return combine(state, edge_value + input);
    }

    __device__ static state_type accumulate_unweighted(
        state_type state, float input) noexcept
    {
        return combine(state, input);
    }

    template <typename Tile>
    __device__ static state_type shuffle_down(const Tile& tile,
                                              state_type state,
                                              int offset) noexcept
    {
        return tile.shfl_down(state, offset);
    }

    __device__ static float finalize(state_type state) noexcept
    {
        return state;
    }
};

using MinPlusPolicy = TropicalPolicy<true>;
using MaxPlusPolicy = TropicalPolicy<false>;

}  // namespace

cudaError_t csr_minplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices, const float* x,
                        float* y, CsrKernelOptions options,
                        cudaStream_t stream) noexcept
{
    return detail::csr_reduce<MinPlusPolicy, false>(
        num_rows, row_offsets, column_indices, nullptr, x, y, options, stream);
}

cudaError_t csr_minplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices,
                        const float* edge_values, const float* x, float* y,
                        CsrKernelOptions options,
                        cudaStream_t stream) noexcept
{
    return detail::csr_reduce<MinPlusPolicy, true>(
        num_rows, row_offsets, column_indices, edge_values, x, y, options,
        stream);
}

cudaError_t csr_maxplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices, const float* x,
                        float* y, CsrKernelOptions options,
                        cudaStream_t stream) noexcept
{
    return detail::csr_reduce<MaxPlusPolicy, false>(
        num_rows, row_offsets, column_indices, nullptr, x, y, options, stream);
}

cudaError_t csr_maxplus(std::int32_t num_rows,
                        const std::int32_t* row_offsets,
                        const std::int32_t* column_indices,
                        const float* edge_values, const float* x, float* y,
                        CsrKernelOptions options,
                        cudaStream_t stream) noexcept
{
    return detail::csr_reduce<MaxPlusPolicy, true>(
        num_rows, row_offsets, column_indices, edge_values, x, y, options,
        stream);
}

}  // namespace springmv

