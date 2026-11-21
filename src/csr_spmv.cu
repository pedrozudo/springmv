// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// This operation uses the classical CSR scheduling adapted from Ginkgo
// v1.11.0 through SpringMV's shared reduction engine. Full provenance is
// recorded in csr_reduce.cuh.

#include "springmv/csr_spmv.hpp"

#include "csr_reduce.cuh"

namespace springmv {
namespace {

struct ArithmeticPolicy {
    using state_type = float;

    __device__ static state_type identity() noexcept { return 0.0F; }

    __device__ static state_type accumulate_weighted(
        state_type state, float edge_value, float input) noexcept
    {
        return state + edge_value * input;
    }

    __device__ static state_type accumulate_unweighted(
        state_type state, float input) noexcept
    {
        return state + input;
    }

    __device__ static state_type combine(state_type left,
                                         state_type right) noexcept
    {
        return left + right;
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

}  // namespace

cudaError_t csr_spmv(std::int32_t num_rows,
                     const std::int32_t* row_offsets,
                     const std::int32_t* column_indices, const float* x,
                     float* y, CsrKernelOptions options,
                     cudaStream_t stream) noexcept
{
    return detail::csr_reduce<ArithmeticPolicy, false>(
        num_rows, row_offsets, column_indices, nullptr, x, y, options, stream);
}

cudaError_t csr_spmv(std::int32_t num_rows,
                     const std::int32_t* row_offsets,
                     const std::int32_t* column_indices, const float* values,
                     const float* x, float* y, CsrKernelOptions options,
                     cudaStream_t stream) noexcept
{
    return detail::csr_reduce<ArithmeticPolicy, true>(
        num_rows, row_offsets, column_indices, values, x, y, options, stream);
}

}  // namespace springmv

