// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace springmv::detail {

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

}  // namespace springmv::detail
