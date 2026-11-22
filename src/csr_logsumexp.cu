// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// The row/subwarp scheduling is adapted from Ginkgo's classical CUDA/HIP CSR
// SpMV kernel at v1.11.0. The stable log-semiring accumulator and reduction
// are SpringMV additions. Full provenance is recorded in csr_reduce.cuh.

#include "springmv/csr_logsumexp.hpp"
#include "springmv/csr_batched.hpp"

#include "csr_logsumexp_policy.cuh"
#include "csr_reduce.cuh"

namespace springmv {
cudaError_t csr_logsumexp(std::int32_t num_rows,
                          const std::int32_t* row_offsets,
                          const std::int32_t* column_indices, const float* x,
                          float* y, CsrKernelOptions options,
                          cudaStream_t stream) noexcept
{
    return detail::csr_reduce<detail::LogSumExpPolicy, false>(
        num_rows, row_offsets, column_indices, nullptr, x, y, options, stream);
}

cudaError_t csr_logsumexp(std::int32_t num_rows,
                          const std::int32_t* row_offsets,
                          const std::int32_t* column_indices,
                          const float* log_values, const float* x, float* y,
                          CsrKernelOptions options,
                          cudaStream_t stream) noexcept
{
    return detail::csr_reduce<detail::LogSumExpPolicy, true>(
        num_rows, row_offsets, column_indices, log_values, x, y, options,
        stream);
}

cudaError_t csr_logsumexp_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_x, std::int32_t x_batch_stride, float* log_y,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return detail::csr_reduce_batched<detail::LogSumExpPolicy, false>(
        batch_size, num_rows, row_offsets, column_indices, nullptr, 0, log_x,
        x_batch_stride, log_y, options, stream);
}

cudaError_t csr_logsumexp_batched(
    std::int32_t batch_size, std::int32_t num_rows,
    const std::int32_t* row_offsets, const std::int32_t* column_indices,
    const float* log_values, std::int32_t values_batch_stride,
    const float* log_x, std::int32_t x_batch_stride, float* log_y,
    CsrKernelOptions options, cudaStream_t stream) noexcept
{
    return detail::csr_reduce_batched<detail::LogSumExpPolicy, true>(
        batch_size, num_rows, row_offsets, column_indices, log_values,
        values_batch_stride, log_x, x_batch_stride, log_y, options, stream);
}

}  // namespace springmv

