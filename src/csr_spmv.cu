// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// This operation uses the classical CSR scheduling adapted from Ginkgo
// v1.11.0 through SpringMV's shared reduction engine. Full provenance is
// recorded in csr_reduce.cuh.

#include "springmv/csr_spmv.hpp"

#include "csr_arithmetic_policy.cuh"
#include "csr_reduce.cuh"

namespace springmv {

cudaError_t csr_spmv(std::int32_t num_rows,
                     const std::int32_t* row_offsets,
                     const std::int32_t* column_indices, const float* x,
                     float* y, CsrKernelOptions options,
                     cudaStream_t stream) noexcept
{
    return detail::csr_reduce<detail::ArithmeticPolicy, false>(
        num_rows, row_offsets, column_indices, nullptr, x, y, options, stream);
}

cudaError_t csr_spmv(std::int32_t num_rows,
                     const std::int32_t* row_offsets,
                     const std::int32_t* column_indices, const float* values,
                     const float* x, float* y, CsrKernelOptions options,
                     cudaStream_t stream) noexcept
{
    return detail::csr_reduce<detail::ArithmeticPolicy, true>(
        num_rows, row_offsets, column_indices, values, x, y, options, stream);
}

}  // namespace springmv

