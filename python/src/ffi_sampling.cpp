// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "ffi_common.hpp"
#include "ffi_handlers.hpp"

#include "springmv/csr_sampling.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace springmv::jax_ffi {
namespace {

ffi::Error validate_topology(
    S32Buffer row_offsets, S32Buffer column_indices,
    S32Buffer transpose_row_offsets, S32Buffer transpose_column_indices,
    std::int32_t* num_rows, std::int32_t* num_children)
{
    if (row_offsets.element_count() == 0 ||
        transpose_row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "CSR row-offset arrays must contain at least one entry");
    }
    auto error = checked_int32_size(row_offsets.element_count() - 1,
                                    "number of parent rows", num_rows);
    if (error.failure()) {
        return error;
    }
    error = checked_int32_size(transpose_row_offsets.element_count() - 1,
                               "number of children", num_children);
    if (error.failure()) {
        return error;
    }
    if (column_indices.element_count() !=
        transpose_column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "CSR and transposed CSR must have equal nonzero counts");
    }
    return ffi::Error::Success();
}

ffi::Error validate_options(std::int32_t max_nnz_per_row,
                            std::int32_t transpose_max_nnz_per_row,
                            std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    return validate_kernel_options(transpose_max_nnz_per_row,
                                   subwarp_size);
}

ffi::Error SamplingImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    F32Buffer log_weights, S32Buffer row_offsets,
    S32Buffer column_indices, S32Buffer transpose_row_offsets,
    S32Buffer transpose_column_indices, F32Buffer gumbels,
    PredBuffer active_parent, PredResult active_child,
    std::int32_t max_nnz_per_row,
    std::int32_t transpose_max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_options(max_nnz_per_row,
                                  transpose_max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    std::int32_t num_children{};
    error = validate_topology(row_offsets, column_indices,
                              transpose_row_offsets,
                              transpose_column_indices, &num_rows,
                              &num_children);
    if (error.failure()) {
        return error;
    }
    const auto nonzeros = column_indices.element_count();
    if (log_weights.element_count() != nonzeros ||
        gumbels.element_count() != nonzeros) {
        return ffi::Error::InvalidArgument(
            "log_weights, gumbels, and column_indices must have equal lengths");
    }
    if (active_parent.element_count() !=
        static_cast<std::size_t>(num_rows)) {
        return ffi::Error::InvalidArgument(
            "active_parent length must equal the number of parent rows");
    }
    if (active_child->element_count() !=
        static_cast<std::size_t>(num_children)) {
        return ffi::Error::InvalidArgument(
            "active_child length must equal the number of children");
    }

    std::int32_t* selected_child{};
    if (num_rows > 0) {
        auto allocation = scratch.Allocate(
            static_cast<std::size_t>(num_rows) * sizeof(std::int32_t),
            alignof(std::int32_t));
        if (!allocation.has_value()) {
            return ffi::Error::Internal(
                "failed to allocate selected-child scratch storage");
        }
        selected_child = static_cast<std::int32_t*>(*allocation);
    }
    return cuda_error(
        csr_logsumexp_sample(
            num_rows, num_children, row_offsets.typed_data(),
            column_indices.typed_data(),
            transpose_row_offsets.typed_data(),
            transpose_column_indices.typed_data(),
            log_weights.typed_data(), gumbels.typed_data(),
            active_parent.typed_data(), selected_child,
            active_child->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size},
            CsrKernelOptions{transpose_max_nnz_per_row, subwarp_size},
            stream),
        "csr_logsumexp_sample");
}

ffi::Error SamplingBatchedImpl(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    F32AnyBuffer log_weights, S32Buffer row_offsets,
    S32Buffer column_indices, S32Buffer transpose_row_offsets,
    S32Buffer transpose_column_indices, F32AnyBuffer gumbels,
    PredAnyBuffer active_parent, PredResultR2 active_child,
    std::int32_t batch_size, std::int32_t weights_batch_stride,
    std::int32_t gumbels_batch_stride,
    std::int32_t active_parent_batch_stride,
    std::int32_t max_nnz_per_row,
    std::int32_t transpose_max_nnz_per_row,
    std::int32_t subwarp_size)
{
    if (batch_size < 0 || weights_batch_stride < 0 ||
        gumbels_batch_stride < 0 || active_parent_batch_stride < 0) {
        return ffi::Error::InvalidArgument(
            "batch size and batch strides must be nonnegative");
    }
    auto error = validate_options(max_nnz_per_row,
                                  transpose_max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    std::int32_t num_children{};
    error = validate_topology(row_offsets, column_indices,
                              transpose_row_offsets,
                              transpose_column_indices, &num_rows,
                              &num_children);
    if (error.failure()) {
        return error;
    }
    if ((num_rows > 0 &&
         batch_size > std::numeric_limits<std::int32_t>::max() / num_rows) ||
        (num_children > 0 &&
         batch_size >
             std::numeric_limits<std::int32_t>::max() / num_children)) {
        return ffi::Error::InvalidArgument(
            "batch dimensions are too large for int32 indexing");
    }

    const auto nonzeros = column_indices.element_count();
    const auto expected_weights = weights_batch_stride == 0
                                      ? nonzeros
                                      : static_cast<std::size_t>(batch_size) *
                                            nonzeros;
    const auto expected_gumbels = gumbels_batch_stride == 0
                                      ? nonzeros
                                      : static_cast<std::size_t>(batch_size) *
                                            nonzeros;
    const auto expected_active = active_parent_batch_stride == 0
                                     ? static_cast<std::size_t>(num_rows)
                                     : static_cast<std::size_t>(batch_size) *
                                           num_rows;
    const auto expected_output = static_cast<std::size_t>(batch_size) *
                                 static_cast<std::size_t>(num_children);
    if ((weights_batch_stride != 0 &&
         weights_batch_stride != static_cast<std::int32_t>(nonzeros)) ||
        (gumbels_batch_stride != 0 &&
         gumbels_batch_stride != static_cast<std::int32_t>(nonzeros)) ||
        (active_parent_batch_stride != 0 &&
         active_parent_batch_stride != num_rows)) {
        return ffi::Error::InvalidArgument(
            "nonzero batch strides must equal the corresponding vector length");
    }
    if (log_weights.element_count() != expected_weights ||
        gumbels.element_count() != expected_gumbels ||
        active_parent.element_count() != expected_active ||
        active_child->element_count() != expected_output) {
        return ffi::Error::InvalidArgument(
            "batched sampling operand or output has an invalid size");
    }

    std::int32_t* selected_child{};
    const auto scratch_count = static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(num_rows);
    if (scratch_count > 0) {
        auto allocation = scratch.Allocate(
            scratch_count * sizeof(std::int32_t), alignof(std::int32_t));
        if (!allocation.has_value()) {
            return ffi::Error::Internal(
                "failed to allocate selected-child scratch storage");
        }
        selected_child = static_cast<std::int32_t*>(*allocation);
    }
    return cuda_error(
        csr_logsumexp_sample_batched(
            batch_size, num_rows, num_children, row_offsets.typed_data(),
            column_indices.typed_data(),
            transpose_row_offsets.typed_data(),
            transpose_column_indices.typed_data(),
            log_weights.typed_data(), weights_batch_stride,
            gumbels.typed_data(), gumbels_batch_stride,
            active_parent.typed_data(), active_parent_batch_stride,
            selected_child, active_child->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size},
            CsrKernelOptions{transpose_max_nnz_per_row, subwarp_size},
            stream),
        "csr_logsumexp_sample_batched");
}

#define SPRINGMV_SAMPLING_BINDING                                           \
    ffi::Ffi::Bind()                                                        \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                           \
        .Ctx<ffi::ScratchAllocator>()                                       \
        .Arg<F32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<F32Buffer>()                                                   \
        .Arg<PredBuffer>()                                                  \
        .Ret<PredBuffer>()                                                  \
        .Attr<std::int32_t>("max_nnz_per_row")                            \
        .Attr<std::int32_t>("transpose_max_nnz_per_row")                  \
        .Attr<std::int32_t>("subwarp_size")

#define SPRINGMV_BATCHED_SAMPLING_BINDING                                   \
    ffi::Ffi::Bind()                                                        \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                           \
        .Ctx<ffi::ScratchAllocator>()                                       \
        .Arg<F32AnyBuffer>()                                                \
        .Arg<S32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<S32Buffer>()                                                   \
        .Arg<F32AnyBuffer>()                                                \
        .Arg<PredAnyBuffer>()                                               \
        .Ret<PredBufferR2>()                                                \
        .Attr<std::int32_t>("batch_size")                                 \
        .Attr<std::int32_t>("weights_batch_stride")                       \
        .Attr<std::int32_t>("gumbels_batch_stride")                       \
        .Attr<std::int32_t>("active_parent_batch_stride")                 \
        .Attr<std::int32_t>("max_nnz_per_row")                            \
        .Attr<std::int32_t>("transpose_max_nnz_per_row")                  \
        .Attr<std::int32_t>("subwarp_size")

XLA_FFI_DEFINE_HANDLER(SpringmvLogsumexpSample, SamplingImpl,
                       SPRINGMV_SAMPLING_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvLogsumexpSampleBatched, SamplingBatchedImpl,
                       SPRINGMV_BATCHED_SAMPLING_BINDING);

#undef SPRINGMV_BATCHED_SAMPLING_BINDING
#undef SPRINGMV_SAMPLING_BINDING

}  // namespace

XLA_FFI_Handler* csr_logsumexp_sample_handler() noexcept
{
    return SpringmvLogsumexpSample;
}

XLA_FFI_Handler* csr_logsumexp_sample_batched_handler() noexcept
{
    return SpringmvLogsumexpSampleBatched;
}

}  // namespace springmv::jax_ffi
