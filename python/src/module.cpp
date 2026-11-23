// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "ffi_handlers.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace springmv::jax_ffi {
namespace {

py::capsule encapsulate(XLA_FFI_Handler* handler)
{
    return py::capsule(reinterpret_cast<void*>(handler));
}

}  // namespace
}  // namespace springmv::jax_ffi

PYBIND11_MODULE(_springmv, module)
{
    using namespace springmv::jax_ffi;

    module.doc() = "Typed JAX FFI registrations for SpringMV";
    module.def("registrations", [] {
        py::dict registrations;
        registrations["springmv_csr_spmv_unweighted"] =
            encapsulate(csr_spmv_unweighted_handler());
        registrations["springmv_csr_spmv_weighted"] =
            encapsulate(csr_spmv_weighted_handler());
        registrations["springmv_csr_spmv_batched_unweighted"] =
            encapsulate(csr_spmv_batched_unweighted_handler());
        registrations["springmv_csr_spmv_batched_weighted"] =
            encapsulate(csr_spmv_batched_weighted_handler());
        registrations["springmv_csr_spmv_input_vjp"] =
            encapsulate(csr_spmv_input_vjp_handler());
        registrations["springmv_csr_spmv_input_vjp_scatter"] =
            encapsulate(csr_spmv_input_vjp_scatter_handler());
        registrations["springmv_csr_spmv_values_vjp"] =
            encapsulate(csr_spmv_values_vjp_handler());
        registrations["springmv_csr_logsumexp_unweighted"] =
            encapsulate(csr_logsumexp_unweighted_handler());
        registrations["springmv_csr_logsumexp_weighted"] =
            encapsulate(csr_logsumexp_weighted_handler());
        registrations["springmv_csr_logsumexp_batched_unweighted"] =
            encapsulate(csr_logsumexp_batched_unweighted_handler());
        registrations["springmv_csr_logsumexp_batched_weighted"] =
            encapsulate(csr_logsumexp_batched_weighted_handler());
        registrations["springmv_csr_logsumexp_jvp_unweighted"] =
            encapsulate(csr_logsumexp_jvp_unweighted_handler());
        registrations["springmv_csr_logsumexp_jvp_weighted"] =
            encapsulate(csr_logsumexp_jvp_weighted_handler());
        registrations["springmv_csr_logsumexp_input_vjp_unweighted"] =
            encapsulate(csr_logsumexp_input_vjp_unweighted_handler());
        registrations["springmv_csr_logsumexp_input_vjp_weighted"] =
            encapsulate(csr_logsumexp_input_vjp_weighted_handler());
        registrations["springmv_csr_logsumexp_input_vjp_scatter"] =
            encapsulate(csr_logsumexp_input_vjp_scatter_handler());
        registrations["springmv_csr_logsumexp_values_vjp"] =
            encapsulate(csr_logsumexp_values_vjp_handler());
        registrations["springmv_csr_logsumexp_sample"] =
            encapsulate(csr_logsumexp_sample_handler());
        registrations["springmv_csr_logsumexp_sample_batched"] =
            encapsulate(csr_logsumexp_sample_batched_handler());
        registrations["springmv_csr_minplus_batched_unweighted"] =
            encapsulate(csr_minplus_batched_unweighted_handler());
        registrations["springmv_csr_minplus_batched_weighted"] =
            encapsulate(csr_minplus_batched_weighted_handler());
        registrations["springmv_csr_maxplus_batched_unweighted"] =
            encapsulate(csr_maxplus_batched_unweighted_handler());
        registrations["springmv_csr_maxplus_batched_weighted"] =
            encapsulate(csr_maxplus_batched_weighted_handler());
        registrations["springmv_csr_minplus_unweighted"] =
            encapsulate(csr_minplus_unweighted_handler());
        registrations["springmv_csr_minplus_weighted"] =
            encapsulate(csr_minplus_weighted_handler());
        registrations["springmv_csr_maxplus_unweighted"] =
            encapsulate(csr_maxplus_unweighted_handler());
        registrations["springmv_csr_maxplus_weighted"] =
            encapsulate(csr_maxplus_weighted_handler());
        registrations["springmv_csr_transpose_unweighted"] =
            encapsulate(csr_transpose_unweighted_handler());
        registrations["springmv_csr_transpose_weighted"] =
            encapsulate(csr_transpose_weighted_handler());
        registrations["springmv_csr_transpose_unweighted_with_permutation"] =
            encapsulate(csr_transpose_unweighted_permutation_handler());
        registrations["springmv_csr_transpose_weighted_with_permutation"] =
            encapsulate(csr_transpose_weighted_permutation_handler());
        return registrations;
    });
}
