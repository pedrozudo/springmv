// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_common.hpp"

namespace springmv {

std::int32_t select_subwarp_size(std::int32_t max_nnz_per_row) noexcept
{
    if (max_nnz_per_row >= 32) {
        return 32;
    }
    if (max_nnz_per_row >= 16) {
        return 16;
    }
    if (max_nnz_per_row >= 8) {
        return 8;
    }
    if (max_nnz_per_row >= 4) {
        return 4;
    }
    if (max_nnz_per_row >= 2) {
        return 2;
    }
    return 1;
}

}  // namespace springmv

