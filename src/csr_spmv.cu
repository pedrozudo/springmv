// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// This operation uses the classical CSR scheduling adapted from Ginkgo
// v1.11.0 through SpringMV's shared reduction engine. Full provenance is
// recorded in csr_reduce.cuh.

#include "springmv/csr_spmv.hpp"
#include "springmv/csr_batched.hpp"

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

cudaError_t csr_spmv_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* x, std::int32_t x_batch_stride, float* y,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return detail::csr_reduce_batched<detail::ArithmeticPolicy, false>(
        batch_size, num_rows, row_offsets, column_indices, nullptr, 0, x,
        x_batch_stride, y, options, stream);
}

cudaError_t csr_spmv_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* values, std::int32_t values_batch_stride, const float* x,
    std::int32_t x_batch_stride, float* y, CsrKernelOptions options,
    cudaStream_t stream) noexcept
{
    return detail::csr_reduce_batched<detail::ArithmeticPolicy, true>(
        batch_size, num_rows, row_offsets, column_indices, values,
        values_batch_stride, x, x_batch_stride, y, options, stream);
}

}  // namespace springmv

