// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "xla/ffi/api/c_api.h"

namespace springmv::jax_ffi {

XLA_FFI_Handler* csr_spmv_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_spmv_weighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_logsumexp_weighted_handler() noexcept;
XLA_FFI_Handler* csr_minplus_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_minplus_weighted_handler() noexcept;
XLA_FFI_Handler* csr_maxplus_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_maxplus_weighted_handler() noexcept;
XLA_FFI_Handler* csr_transpose_unweighted_handler() noexcept;
XLA_FFI_Handler* csr_transpose_weighted_handler() noexcept;
XLA_FFI_Handler* csr_transpose_unweighted_permutation_handler() noexcept;
XLA_FFI_Handler* csr_transpose_weighted_permutation_handler() noexcept;

}  // namespace springmv::jax_ffi
