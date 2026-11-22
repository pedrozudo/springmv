// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
// SPDX-FileCopyrightText: 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause
//
// The row/subwarp scheduling is adapted from Ginkgo's classical CUDA/HIP CSR
// SpMV kernel at v1.11.0. The stable log-semiring accumulator and reduction
// are SpringMV additions. Full provenance is recorded in csr_reduce.cuh.

#include "springmv/csr_logsumexp.hpp"

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

}  // namespace springmv

