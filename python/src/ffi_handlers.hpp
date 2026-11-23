// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "xla/ffi/api/c_api.h"

namespace springmv::jax_ffi {

XLA_FFI_Handler* csr_spmv_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_spmv_weighted_handler() noexcept;
XLA_FFI_Handler* csr_spmv_batched_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_spmv_batched_weighted_handler() noexcept;
XLA_FFI_Handler* csr_spmv_input_vjp_handler() noexcept;
XLA_FFI_Handler* csr_spmv_input_vjp_scatter_handler() noexcept;
XLA_FFI_Handler* csr_spmv_values_vjp_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_weighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_batched_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_batched_weighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_jvp_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_jvp_weighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_input_vjp_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_input_vjp_weighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_input_vjp_scatter_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_values_vjp_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_sample_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_sample_batched_handler() noexcept;
XLA_FFI_Handler* csr_minplus_batched_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_minplus_batched_weighted_handler() noexcept;
XLA_FFI_Handler* csr_maxplus_batched_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_maxplus_batched_weighted_handler() noexcept;
XLA_FFI_Handler* csr_minplus_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_minplus_weighted_handler() noexcept;
XLA_FFI_Handler* csr_maxplus_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_maxplus_weighted_handler() noexcept;
XLA_FFI_Handler* csr_transpose_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_transpose_weighted_handler() noexcept;
XLA_FFI_Handler* csr_transpose_unweighted_permutation_handler() noexcept;
XLA_FFI_Handler* csr_transpose_weighted_permutation_handler() noexcept;

}  // namespace springmv::jax_ffi
