// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace springmv::detail {

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

__device__ inline float logsumexp_softmax_factor(
    float term, float output, std::int32_t row_length,
    std::int32_t positive_infinity_count) noexcept
{
    if (isnan(term) || isnan(output)) {
        return CUDART_NAN_F;
    }
    if (output == CUDART_INF_F) {
        return term == CUDART_INF_F
                   ? 1.0F / static_cast<float>(positive_infinity_count)
                   : 0.0F;
    }
    if (output == -CUDART_INF_F) {
        return 1.0F / static_cast<float>(row_length);
    }
    return expf(term - output);
}

}  // namespace springmv::detail
