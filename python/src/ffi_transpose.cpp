// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "ffi_common.hpp"
#include "ffi_handlers.hpp"

#include "springmv/csr_transpose.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace springmv::jax_ffi {
namespace {

ffi::Error validate_transpose_shapes(
    S32Buffer row_offsets, S32Buffer column_indices,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices,
    std::int32_t num_cols, std::int32_t* num_rows,
    std::int32_t* num_nonzeros)
{
    if (num_cols < 0) {
        return ffi::Error::InvalidArgument("num_cols must be nonnegative");
    }
    if (row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "row_offsets must contain at least one entry");
    }
    auto error = checked_int32_size(row_offsets.element_count() - 1,
                                    "number of rows", num_rows);
    if (error.failure()) {
        return error;
    }
    error = checked_int32_size(column_indices.element_count(),
                               "number of nonzeros", num_nonzeros);
    if (error.failure()) {
        return error;
    }
    if (transposed_row_offsets->element_count() !=
        static_cast<std::size_t>(num_cols) + 1) {
        return ffi::Error::InvalidArgument(
            "transposed row_offsets has the wrong length");
    }
    if (transposed_column_indices->element_count() !=
        column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "transpose edge outputs have the wrong length");
    }
    return ffi::Error::Success();
}

ffi::Error allocate_transpose_workspace(std::int32_t num_rows,
                                        std::int32_t num_cols,
                                        std::int32_t num_nonzeros,
                                        ffi::ScratchAllocator& scratch,
                                        void** workspace,
                                        std::size_t* workspace_size)
{
    auto error = cuda_error(
        csr_transpose_workspace_size(num_rows, num_cols, num_nonzeros,
                                     workspace_size),
        "csr_transpose_workspace_size");
    if (error.failure() || *workspace_size == 0) {
        *workspace = nullptr;
        return error;
    }
    auto allocation = scratch.Allocate(*workspace_size);
    if (!allocation.has_value()) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "could not allocate CSR transpose workspace");
    }
    *workspace = *allocation;
    return ffi::Error::Success();
}

ffi::Error transpose_unweighted(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    S32Buffer row_offsets, S32Buffer column_indices,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices, std::int32_t* edge_permutation,
    std::int32_t num_cols)
{
    std::int32_t num_rows{};
    std::int32_t num_nonzeros{};
    auto error = validate_transpose_shapes(
        row_offsets, column_indices, transposed_row_offsets,
        transposed_column_indices, num_cols, &num_rows, &num_nonzeros);
    if (error.failure()) {
        return error;
    }
    void* workspace{};
    std::size_t workspace_size{};
    error = allocate_transpose_workspace(num_rows, num_cols, num_nonzeros,
                                         scratch, &workspace,
                                         &workspace_size);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        csr_transpose(num_rows, num_cols, num_nonzeros,
                      row_offsets.typed_data(), column_indices.typed_data(),
                      transposed_row_offsets->typed_data(),
                      transposed_column_indices->typed_data(),
                      edge_permutation, workspace, workspace_size, stream),
        "csr_transpose");
}

ffi::Error transpose_weighted(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    S32Buffer row_offsets, S32Buffer column_indices, F32Buffer values,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices, F32Result transposed_values,
    std::int32_t* edge_permutation, std::int32_t num_cols)
{
    if (values.element_count() != column_indices.element_count() ||
        transposed_values->element_count() !=
            column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "values and column_indices must have equal lengths");
    }
    std::int32_t num_rows{};
    std::int32_t num_nonzeros{};
    auto error = validate_transpose_shapes(
        row_offsets, column_indices, transposed_row_offsets,
        transposed_column_indices, num_cols, &num_rows, &num_nonzeros);
    if (error.failure()) {
        return error;
    }
    void* workspace{};
    std::size_t workspace_size{};
    error = allocate_transpose_workspace(num_rows, num_cols, num_nonzeros,
                                         scratch, &workspace,
                                         &workspace_size);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        csr_transpose(num_rows, num_cols, num_nonzeros,
                      row_offsets.typed_data(), column_indices.typed_data(),
                      values.typed_data(),
                      transposed_row_offsets->typed_data(),
                      transposed_column_indices->typed_data(),
                      transposed_values->typed_data(),
                      edge_permutation, workspace, workspace_size, stream),
        "csr_transpose");
}

ffi::Error TransposeUnweightedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    S32Buffer row_offsets, S32Buffer column_indices,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices, std::int32_t num_cols)
{
    return transpose_unweighted(stream, std::move(scratch), row_offsets,
                                column_indices, transposed_row_offsets,
                                transposed_column_indices, nullptr, num_cols);
}

ffi::Error TransposeUnweightedPermutationImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    S32Buffer row_offsets, S32Buffer column_indices,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices, S32Result edge_permutation,
    std::int32_t num_cols)
{
    if (edge_permutation->element_count() !=
        column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "transpose edge permutation has the wrong length");
    }
    return transpose_unweighted(
        stream, std::move(scratch), row_offsets, column_indices,
        transposed_row_offsets, transposed_column_indices,
        edge_permutation->typed_data(), num_cols);
}

ffi::Error TransposeWeightedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    S32Buffer row_offsets, S32Buffer column_indices, F32Buffer values,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices, F32Result transposed_values,
    std::int32_t num_cols)
{
    return transpose_weighted(
        stream, std::move(scratch), row_offsets, column_indices, values,
        transposed_row_offsets, transposed_column_indices, transposed_values,
        nullptr, num_cols);
}

ffi::Error TransposeWeightedPermutationImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    S32Buffer row_offsets, S32Buffer column_indices, F32Buffer values,
    S32Result transposed_row_offsets,
    S32Result transposed_column_indices, F32Result transposed_values,
    S32Result edge_permutation, std::int32_t num_cols)
{
    if (edge_permutation->element_count() !=
        column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "transpose edge permutation has the wrong length");
    }
    return transpose_weighted(
        stream, std::move(scratch), row_offsets, column_indices, values,
        transposed_row_offsets, transposed_column_indices, transposed_values,
        edge_permutation->typed_data(), num_cols);
}

XLA_FFI_DEFINE_HANDLER(
    SpringmvTransposeUnweighted, TransposeUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Ret<S32Buffer>()
        .Ret<S32Buffer>()
        .Attr<std::int32_t>("num_cols"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvTransposeWeighted, TransposeWeightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Ret<S32Buffer>()
        .Ret<S32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("num_cols"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvTransposeUnweightedPermutation,
    TransposeUnweightedPermutationImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Ret<S32Buffer>()
        .Ret<S32Buffer>()
        .Ret<S32Buffer>()
        .Attr<std::int32_t>("num_cols"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvTransposeWeightedPermutation, TransposeWeightedPermutationImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Ret<S32Buffer>()
        .Ret<S32Buffer>()
        .Ret<F32Buffer>()
        .Ret<S32Buffer>()
        .Attr<std::int32_t>("num_cols"));

}  // namespace

XLA_FFI_Handler* csr_transpose_unweighted_handler() noexcept
{
    return SpringmvTransposeUnweighted;
}

XLA_FFI_Handler* csr_transpose_weighted_handler() noexcept
{
    return SpringmvTransposeWeighted;
}

XLA_FFI_Handler* csr_transpose_unweighted_permutation_handler() noexcept
{
    return SpringmvTransposeUnweightedPermutation;
}

XLA_FFI_Handler* csr_transpose_weighted_permutation_handler() noexcept
{
    return SpringmvTransposeWeightedPermutation;
}

}  // namespace springmv::jax_ffi
