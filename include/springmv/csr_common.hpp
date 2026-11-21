// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

namespace springmv {

/** Configuration shared by the classical CSR kernels. */
struct CsrKernelOptions {
    /**
     * Maximum number of nonzeros in any row.
     *
     * This is used to choose a subwarp width when subwarp_size is zero and
     * must be nonnegative.
     */
    std::int32_t max_nnz_per_row{};

    /**
     * Threads assigned to each row. Zero selects a width automatically.
     * Explicit values must be one of 1, 2, 4, 8, 16, or 32.
     */
    std::int32_t subwarp_size{};
};

/** Backward-compatible name for arithmetic SpMV configuration. */
using CsrSpmvOptions = CsrKernelOptions;

/**
 * Select the classical-kernel subwarp width used for a row length.
 *
 * The result is the largest power of two no greater than max_nnz_per_row,
 * capped at 32. Empty rows select one thread.
 */
std::int32_t select_subwarp_size(std::int32_t max_nnz_per_row) noexcept;

}  // namespace springmv

