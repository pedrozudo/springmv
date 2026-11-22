// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "ffi_common.hpp"
#include "ffi_handlers.hpp"

#include "springmv/csr_spmv_backward.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace springmv::jax_ffi {
namespace {

ffi::Error SpmvInputVjpImpl(
    cudaStream_t stream, S32Buffer transpose_row_offsets,
    S32Buffer transpose_column_indices, S32Buffer transpose_permutation,
    F32Buffer values, F32Buffer output_cotangent, F32Result input_cotangent,
    std::int32_t max_nnz_per_row, std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    if (transpose_row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "transpose_row_offsets must contain at least one entry");
    }
    std::int32_t num_columns{};
    error = checked_int32_size(transpose_row_offsets.element_count() - 1,
                               "number of columns", &num_columns);
    if (error.failure()) {
        return error;
    }
    const auto num_nonzeros = transpose_column_indices.element_count();
    if (transpose_permutation.element_count() != num_nonzeros ||
        values.element_count() != num_nonzeros) {
        return ffi::Error::InvalidArgument(
            "transpose indices, permutation, and values must have equal "
            "lengths");
    }
    if (input_cotangent->element_count() !=
        static_cast<std::size_t>(num_columns)) {
        return ffi::Error::InvalidArgument(
            "input cotangent has the wrong length");
    }
    if (num_nonzeros > 0 && output_cotangent.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "output cotangent cannot be empty for a nonempty matrix");
    }
    return cuda_error(
        csr_spmv_input_vjp(
            num_columns, transpose_row_offsets.typed_data(),
            transpose_column_indices.typed_data(),
            transpose_permutation.typed_data(), values.typed_data(),
            output_cotangent.typed_data(), input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_spmv_input_vjp");
}

ffi::Error SpmvValuesVjpImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32Buffer x, F32Buffer output_cotangent, F32Result values_cotangent,
    std::int32_t max_nnz_per_row, std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    if (row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "row_offsets must contain at least one entry");
    }
    std::int32_t num_rows{};
    error = checked_int32_size(row_offsets.element_count() - 1,
                               "number of rows", &num_rows);
    if (error.failure()) {
        return error;
    }
    if (output_cotangent.element_count() !=
        static_cast<std::size_t>(num_rows)) {
        return ffi::Error::InvalidArgument(
            "output cotangent length must equal the number of rows");
    }
    if (values_cotangent->element_count() !=
        column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "values cotangent length must equal the number of nonzeros");
    }
    if (column_indices.element_count() > 0 && x.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "x cannot be empty for a nonempty matrix");
    }
    return cuda_error(
        csr_spmv_values_vjp(
            num_rows, row_offsets.typed_data(), column_indices.typed_data(),
            x.typed_data(), output_cotangent.typed_data(),
            values_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_spmv_values_vjp");
}

XLA_FFI_DEFINE_HANDLER(
    SpringmvSpmvInputVjp, SpmvInputVjpImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvSpmvValuesVjp, SpmvValuesVjpImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

}  // namespace

XLA_FFI_Handler* csr_spmv_input_vjp_handler() noexcept
{
    return SpringmvSpmvInputVjp;
}

XLA_FFI_Handler* csr_spmv_values_vjp_handler() noexcept
{
    return SpringmvSpmvValuesVjp;
}

}  // namespace springmv::jax_ffi
