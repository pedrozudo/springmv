// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

namespace springmv::detail {

constexpr int block_size = 128;

inline bool valid_subwarp_size(std::int32_t value) noexcept
{
    return value == 0 || value == 1 || value == 2 || value == 4 ||
           value == 8 || value == 16 || value == 32;
}

template <int subwarp_size>
__device__ std::int32_t initial_row() noexcept
{
    constexpr auto subwarps_per_block = block_size / subwarp_size;
    return static_cast<std::int32_t>(blockIdx.x) * subwarps_per_block +
           static_cast<std::int32_t>(threadIdx.x) / subwarp_size;
}

template <int subwarp_size>
__device__ std::int32_t row_stride() noexcept
{
    constexpr auto subwarps_per_block = block_size / subwarp_size;
    return static_cast<std::int32_t>(gridDim.x) * subwarps_per_block;
}

template <int subwarp_size>
unsigned int grid_size(std::int32_t num_rows) noexcept
{
    constexpr auto rows_per_block = block_size / subwarp_size;
    return static_cast<unsigned int>(
        (static_cast<std::int64_t>(num_rows) + rows_per_block - 1) /
        rows_per_block);
}

}  // namespace springmv::detail

