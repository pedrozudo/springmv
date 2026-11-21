// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// The row/subwarp scheduling is adapted from Ginkgo's classical CUDA/HIP CSR
// SpMV kernel at v1.11.0. The stable log-semiring accumulator and reduction
// are SpringMV additions. Full provenance is recorded in csr_reduce.cuh.

#include "springmv/csr_logsumexp.hpp"

#include "csr_reduce.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace springmv {
namespace {

struct LogSumExpState {
    float maximum;
    float scaled_sum;
};

struct LogSumExpPolicy {
    using state_type = LogSumExpState;

    __device__ static state_type identity() noexcept
    {
        return {-CUDART_INF_F, 0.0F};
    }

    __device__ static state_type nan_state() noexcept
    {
        return {CUDART_NAN_F, CUDART_NAN_F};
    }

    __device__ static state_type accumulate_term(state_type state,
                                                 float term) noexcept
    {
        if (isnan(term) || isnan(state.maximum)) {
            return nan_state();
        }
        if (term == CUDART_INF_F) {
            return {CUDART_INF_F, 1.0F};
        }
        if (term == -CUDART_INF_F) {
            return state;
        }
        if (state.scaled_sum == 0.0F) {
            return {term, 1.0F};
        }
        if (term > state.maximum) {
            state.scaled_sum =
                state.scaled_sum * expf(state.maximum - term) + 1.0F;
            state.maximum = term;
        } else {
            state.scaled_sum += expf(term - state.maximum);
        }
        return state;
    }

    __device__ static state_type accumulate_weighted(
        state_type state, float log_weight, float log_input) noexcept
    {
        return accumulate_term(state, log_weight + log_input);
    }

    __device__ static state_type accumulate_unweighted(
        state_type state, float log_input) noexcept
    {
        return accumulate_term(state, log_input);
    }

    __device__ static state_type combine(state_type left,
                                         state_type right) noexcept
    {
        if (isnan(left.maximum) || isnan(right.maximum)) {
            return nan_state();
        }
        if (left.maximum == CUDART_INF_F ||
            right.maximum == CUDART_INF_F) {
            return {CUDART_INF_F, 1.0F};
        }
        if (left.scaled_sum == 0.0F) {
            return right;
        }
        if (right.scaled_sum == 0.0F) {
            return left;
        }

        const auto maximum = fmaxf(left.maximum, right.maximum);
        const auto scaled_sum =
            left.scaled_sum * expf(left.maximum - maximum) +
            right.scaled_sum * expf(right.maximum - maximum);
        return {maximum, scaled_sum};
    }

    template <typename Tile>
    __device__ static state_type shuffle_down(const Tile& tile,
                                              state_type state,
                                              int offset) noexcept
    {
        return {tile.shfl_down(state.maximum, offset),
                tile.shfl_down(state.scaled_sum, offset)};
    }

    __device__ static float finalize(state_type state) noexcept
    {
        if (isnan(state.maximum) || state.maximum == CUDART_INF_F ||
            state.scaled_sum == 0.0F) {
            return state.maximum;
        }
        return state.maximum + logf(state.scaled_sum);
    }
};

}  // namespace

cudaError_t csr_logsumexp(std::int32_t num_rows,
                          const std::int32_t* row_offsets,
                          const std::int32_t* column_indices, const float* x,
                          float* y, CsrKernelOptions options,
                          cudaStream_t stream) noexcept
{
    return detail::csr_reduce<LogSumExpPolicy, false>(
        num_rows, row_offsets, column_indices, nullptr, x, y, options, stream);
}

cudaError_t csr_logsumexp(std::int32_t num_rows,
                          const std::int32_t* row_offsets,
                          const std::int32_t* column_indices,
                          const float* log_values, const float* x, float* y,
                          CsrKernelOptions options,
                          cudaStream_t stream) noexcept
{
    return detail::csr_reduce<LogSumExpPolicy, true>(
        num_rows, row_offsets, column_indices, log_values, x, y, options,
        stream);
}

}  // namespace springmv

