// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "ffi_common.hpp"
#include "ffi_handlers.hpp"

#include "springmv/csr_batched.hpp"
#include "springmv/csr_logsumexp.hpp"
#include "springmv/csr_spmv.hpp"
#include "springmv/csr_tropical.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace springmv::jax_ffi {
namespace {

using UnweightedOperation = cudaError_t (*)(
    std::int32_t, const std::int32_t*, const std::int32_t*, const float*,
    float*, CsrKernelOptions, cudaStream_t) noexcept;
using WeightedOperation = cudaError_t (*)(
    std::int32_t, const std::int32_t*, const std::int32_t*, const float*,
    const float*, float*, CsrKernelOptions, cudaStream_t) noexcept;
using BatchedUnweightedOperation = cudaError_t (*)(
    std::int32_t, std::int32_t, const std::int32_t*, const std::int32_t*,
    const float*, std::int32_t, float*, CsrKernelOptions,
    cudaStream_t) noexcept;
using BatchedWeightedOperation = cudaError_t (*)(
    std::int32_t, std::int32_t, const std::int32_t*, const std::int32_t*,
    const float*, std::int32_t, const float*, std::int32_t, float*,
    CsrKernelOptions, cudaStream_t) noexcept;

ffi::Error validate_reduction_shapes(S32Buffer row_offsets,
                                     S32Buffer column_indices, F32Buffer x,
                                     F32Result output,
                                     std::int32_t* num_rows)
{
    if (row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "row_offsets must contain at least one entry");
    }
    auto error = checked_int32_size(row_offsets.element_count() - 1,
                                    "number of rows", num_rows);
    if (error.failure()) {
        return error;
    }
    if (output->element_count() != static_cast<std::size_t>(*num_rows)) {
        return ffi::Error::InvalidArgument(
            "output length must equal row_offsets length minus one");
    }
    if (column_indices.element_count() > 0 && x.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "x cannot be empty when column_indices is nonempty");
    }
    return ffi::Error::Success();
}

ffi::Error invoke_unweighted(UnweightedOperation operation,
                             const char* operation_name, cudaStream_t stream,
                             S32Buffer row_offsets, S32Buffer column_indices,
                             F32Buffer x, F32Result output,
                             std::int32_t max_nnz_per_row,
                             std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_reduction_shapes(row_offsets, column_indices, x, output,
                                      &num_rows);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        operation(num_rows, row_offsets.typed_data(),
                  column_indices.typed_data(), x.typed_data(),
                  output->typed_data(),
                  CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        operation_name);
}

ffi::Error invoke_weighted(WeightedOperation operation,
                           const char* operation_name, cudaStream_t stream,
                           S32Buffer row_offsets, S32Buffer column_indices,
                           F32Buffer edge_values, F32Buffer x,
                           F32Result output,
                           std::int32_t max_nnz_per_row,
                           std::int32_t subwarp_size)
{
    if (edge_values.element_count() != column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "edge values and column_indices must have equal lengths");
    }
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_reduction_shapes(row_offsets, column_indices, x, output,
                                      &num_rows);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        operation(num_rows, row_offsets.typed_data(),
                  column_indices.typed_data(), edge_values.typed_data(),
                  x.typed_data(), output->typed_data(),
                  CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        operation_name);
}

ffi::Error validate_batched_shapes(
    S32Buffer row_offsets, S32Buffer column_indices,
    F32AnyBuffer edge_values, F32AnyBuffer x, F32ResultR2 output,
    std::int32_t batch_size,
    std::int32_t values_batch_stride, std::int32_t x_batch_stride,
    bool weighted, std::int32_t* num_rows)
{
    if (batch_size < 0 || values_batch_stride < 0 || x_batch_stride < 0) {
        return ffi::Error::InvalidArgument(
            "batch size and batch strides must be nonnegative");
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
    const auto expected_output = static_cast<std::size_t>(batch_size) *
                                 static_cast<std::size_t>(*num_rows);
    if (output->element_count() != expected_output) {
        return ffi::Error::InvalidArgument(
            "batched output size must equal batch_size times num_rows");
    }
    const auto nonzeros = column_indices.element_count();
    if (weighted) {
        const auto expected_values = values_batch_stride == 0
                                         ? nonzeros
                                         : static_cast<std::size_t>(batch_size) *
                                               nonzeros;
        if (edge_values.element_count() != expected_values) {
            return ffi::Error::InvalidArgument(
                "batched edge-value buffer has an invalid size");
        }
    }
    if (x_batch_stride > 0 &&
        x.element_count() != static_cast<std::size_t>(batch_size) *
                                 static_cast<std::size_t>(x_batch_stride)) {
        return ffi::Error::InvalidArgument(
            "batched input buffer has an invalid size");
    }
    if (nonzeros > 0 && x.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "x cannot be empty when column_indices is nonempty");
    }
    return ffi::Error::Success();
}

ffi::Error invoke_batched_unweighted(
    BatchedUnweightedOperation operation, const char* operation_name,
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32AnyBuffer x, F32ResultR2 output, std::int32_t batch_size,
    std::int32_t x_batch_stride, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_batched_shapes(
        row_offsets, column_indices, x, x, output, batch_size, 0,
        x_batch_stride, false, &num_rows);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        operation(batch_size, num_rows, row_offsets.typed_data(),
                  column_indices.typed_data(), x.typed_data(), x_batch_stride,
                  output->typed_data(),
                  CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        operation_name);
}

ffi::Error invoke_batched_weighted(
    BatchedWeightedOperation operation, const char* operation_name,
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32AnyBuffer edge_values, F32AnyBuffer x, F32ResultR2 output,
    std::int32_t batch_size, std::int32_t values_batch_stride,
    std::int32_t x_batch_stride, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_batched_shapes(
        row_offsets, column_indices, edge_values, x, output, batch_size,
        values_batch_stride, x_batch_stride, true, &num_rows);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        operation(batch_size, num_rows, row_offsets.typed_data(),
                  column_indices.typed_data(), edge_values.typed_data(),
                  values_batch_stride, x.typed_data(), x_batch_stride,
                  output->typed_data(),
                  CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        operation_name);
}

#define SPRINGMV_DEFINE_REDUCTION_IMPLS(NAME, FUNCTION)                       \
    ffi::Error NAME##UnweightedImpl(                                         \
        cudaStream_t stream, S32Buffer row_offsets,                          \
        S32Buffer column_indices, F32Buffer x, F32Result output,              \
        std::int32_t max_nnz_per_row, std::int32_t subwarp_size)              \
    {                                                                         \
        return invoke_unweighted(                                             \
            static_cast<UnweightedOperation>(&FUNCTION), #FUNCTION, stream,   \
            row_offsets, column_indices, x, output, max_nnz_per_row,          \
            subwarp_size);                                                    \
    }                                                                         \
    ffi::Error NAME##WeightedImpl(                                           \
        cudaStream_t stream, S32Buffer row_offsets,                          \
        S32Buffer column_indices, F32Buffer edge_values, F32Buffer x,         \
        F32Result output, std::int32_t max_nnz_per_row,                       \
        std::int32_t subwarp_size)                                            \
    {                                                                         \
        return invoke_weighted(                                               \
            static_cast<WeightedOperation>(&FUNCTION), #FUNCTION, stream,     \
            row_offsets, column_indices, edge_values, x, output,              \
            max_nnz_per_row, subwarp_size);                                   \
    }

SPRINGMV_DEFINE_REDUCTION_IMPLS(Spmv, springmv::csr_spmv)
SPRINGMV_DEFINE_REDUCTION_IMPLS(Logsumexp, springmv::csr_logsumexp)
SPRINGMV_DEFINE_REDUCTION_IMPLS(Minplus, springmv::csr_minplus)
SPRINGMV_DEFINE_REDUCTION_IMPLS(Maxplus, springmv::csr_maxplus)

#undef SPRINGMV_DEFINE_REDUCTION_IMPLS

#define SPRINGMV_DEFINE_BATCHED_REDUCTION_IMPLS(NAME, FUNCTION)              \
    ffi::Error NAME##BatchedUnweightedImpl(                                  \
        cudaStream_t stream, S32Buffer row_offsets,                          \
        S32Buffer column_indices, F32AnyBuffer x, F32ResultR2 output,         \
        std::int32_t batch_size, std::int32_t x_batch_stride,                 \
        std::int32_t max_nnz_per_row, std::int32_t subwarp_size)              \
    {                                                                         \
        return invoke_batched_unweighted(                                     \
            static_cast<BatchedUnweightedOperation>(&FUNCTION), #FUNCTION,    \
            stream, row_offsets, column_indices, x, output, batch_size,       \
            x_batch_stride, max_nnz_per_row, subwarp_size);                  \
    }                                                                         \
    ffi::Error NAME##BatchedWeightedImpl(                                    \
        cudaStream_t stream, S32Buffer row_offsets,                          \
        S32Buffer column_indices, F32AnyBuffer edge_values,                   \
        F32AnyBuffer x, F32ResultR2 output, std::int32_t batch_size,          \
        std::int32_t values_batch_stride, std::int32_t x_batch_stride,        \
        std::int32_t max_nnz_per_row, std::int32_t subwarp_size)              \
    {                                                                         \
        return invoke_batched_weighted(                                       \
            static_cast<BatchedWeightedOperation>(&FUNCTION), #FUNCTION,      \
            stream, row_offsets, column_indices, edge_values, x, output,      \
            batch_size, values_batch_stride, x_batch_stride,                 \
            max_nnz_per_row, subwarp_size);                                  \
    }

SPRINGMV_DEFINE_BATCHED_REDUCTION_IMPLS(Spmv, springmv::csr_spmv_batched)
SPRINGMV_DEFINE_BATCHED_REDUCTION_IMPLS(Logsumexp,
                                        springmv::csr_logsumexp_batched)
SPRINGMV_DEFINE_BATCHED_REDUCTION_IMPLS(Minplus,
                                        springmv::csr_minplus_batched)
SPRINGMV_DEFINE_BATCHED_REDUCTION_IMPLS(Maxplus,
                                        springmv::csr_maxplus_batched)

#undef SPRINGMV_DEFINE_BATCHED_REDUCTION_IMPLS

#define SPRINGMV_UNWEIGHTED_BINDING                                           \
    ffi::Ffi::Bind()                                                          \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                             \
        .Arg<S32Buffer>()                                                     \
        .Arg<S32Buffer>()                                                     \
        .Arg<F32Buffer>()                                                     \
        .Ret<F32Buffer>()                                                     \
        .Attr<std::int32_t>("max_nnz_per_row")                               \
        .Attr<std::int32_t>("subwarp_size")

#define SPRINGMV_WEIGHTED_BINDING                                             \
    ffi::Ffi::Bind()                                                          \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                             \
        .Arg<S32Buffer>()                                                     \
        .Arg<S32Buffer>()                                                     \
        .Arg<F32Buffer>()                                                     \
        .Arg<F32Buffer>()                                                     \
        .Ret<F32Buffer>()                                                     \
        .Attr<std::int32_t>("max_nnz_per_row")                               \
        .Attr<std::int32_t>("subwarp_size")

#define SPRINGMV_BATCHED_UNWEIGHTED_BINDING                                  \
    ffi::Ffi::Bind()                                                          \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                             \
        .Arg<S32Buffer>()                                                     \
        .Arg<S32Buffer>()                                                     \
        .Arg<F32AnyBuffer>()                                                  \
        .Ret<F32BufferR2>()                                                   \
        .Attr<std::int32_t>("batch_size")                                   \
        .Attr<std::int32_t>("x_batch_stride")                               \
        .Attr<std::int32_t>("max_nnz_per_row")                              \
        .Attr<std::int32_t>("subwarp_size")

#define SPRINGMV_BATCHED_WEIGHTED_BINDING                                    \
    ffi::Ffi::Bind()                                                          \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                             \
        .Arg<S32Buffer>()                                                     \
        .Arg<S32Buffer>()                                                     \
        .Arg<F32AnyBuffer>()                                                  \
        .Arg<F32AnyBuffer>()                                                  \
        .Ret<F32BufferR2>()                                                   \
        .Attr<std::int32_t>("batch_size")                                   \
        .Attr<std::int32_t>("values_batch_stride")                          \
        .Attr<std::int32_t>("x_batch_stride")                               \
        .Attr<std::int32_t>("max_nnz_per_row")                              \
        .Attr<std::int32_t>("subwarp_size")

XLA_FFI_DEFINE_HANDLER(SpringmvSpmvUnweighted, SpmvUnweightedImpl,
                       SPRINGMV_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvSpmvWeighted, SpmvWeightedImpl,
                       SPRINGMV_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvLogsumexpUnweighted, LogsumexpUnweightedImpl,
                       SPRINGMV_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvLogsumexpWeighted, LogsumexpWeightedImpl,
                       SPRINGMV_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMinplusUnweighted, MinplusUnweightedImpl,
                       SPRINGMV_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMinplusWeighted, MinplusWeightedImpl,
                       SPRINGMV_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMaxplusUnweighted, MaxplusUnweightedImpl,
                       SPRINGMV_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMaxplusWeighted, MaxplusWeightedImpl,
                       SPRINGMV_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvSpmvBatchedUnweighted,
                       SpmvBatchedUnweightedImpl,
                       SPRINGMV_BATCHED_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvSpmvBatchedWeighted,
                       SpmvBatchedWeightedImpl,
                       SPRINGMV_BATCHED_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvLogsumexpBatchedUnweighted,
                       LogsumexpBatchedUnweightedImpl,
                       SPRINGMV_BATCHED_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvLogsumexpBatchedWeighted,
                       LogsumexpBatchedWeightedImpl,
                       SPRINGMV_BATCHED_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMinplusBatchedUnweighted,
                       MinplusBatchedUnweightedImpl,
                       SPRINGMV_BATCHED_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMinplusBatchedWeighted,
                       MinplusBatchedWeightedImpl,
                       SPRINGMV_BATCHED_WEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMaxplusBatchedUnweighted,
                       MaxplusBatchedUnweightedImpl,
                       SPRINGMV_BATCHED_UNWEIGHTED_BINDING);
XLA_FFI_DEFINE_HANDLER(SpringmvMaxplusBatchedWeighted,
                       MaxplusBatchedWeightedImpl,
                       SPRINGMV_BATCHED_WEIGHTED_BINDING);

#undef SPRINGMV_BATCHED_UNWEIGHTED_BINDING
#undef SPRINGMV_BATCHED_WEIGHTED_BINDING
#undef SPRINGMV_UNWEIGHTED_BINDING
#undef SPRINGMV_WEIGHTED_BINDING

}  // namespace

XLA_FFI_Handler* csr_spmv_unweighted_handler() noexcept
{
    return SpringmvSpmvUnweighted;
}

XLA_FFI_Handler* csr_spmv_weighted_handler() noexcept
{
    return SpringmvSpmvWeighted;
}

XLA_FFI_Handler* csr_logsumexp_unweighted_handler() noexcept
{
    return SpringmvLogsumexpUnweighted;
}

XLA_FFI_Handler* csr_logsumexp_weighted_handler() noexcept
{
    return SpringmvLogsumexpWeighted;
}

XLA_FFI_Handler* csr_minplus_unweighted_handler() noexcept
{
    return SpringmvMinplusUnweighted;
}

XLA_FFI_Handler* csr_minplus_weighted_handler() noexcept
{
    return SpringmvMinplusWeighted;
}

XLA_FFI_Handler* csr_maxplus_unweighted_handler() noexcept
{
    return SpringmvMaxplusUnweighted;
}

XLA_FFI_Handler* csr_maxplus_weighted_handler() noexcept
{
    return SpringmvMaxplusWeighted;
}

XLA_FFI_Handler* csr_spmv_batched_unweighted_handler() noexcept
{
    return SpringmvSpmvBatchedUnweighted;
}

XLA_FFI_Handler* csr_spmv_batched_weighted_handler() noexcept
{
    return SpringmvSpmvBatchedWeighted;
}

XLA_FFI_Handler* csr_logsumexp_batched_unweighted_handler() noexcept
{
    return SpringmvLogsumexpBatchedUnweighted;
}

XLA_FFI_Handler* csr_logsumexp_batched_weighted_handler() noexcept
{
    return SpringmvLogsumexpBatchedWeighted;
}

XLA_FFI_Handler* csr_minplus_batched_unweighted_handler() noexcept
{
    return SpringmvMinplusBatchedUnweighted;
}

XLA_FFI_Handler* csr_minplus_batched_weighted_handler() noexcept
{
    return SpringmvMinplusBatchedWeighted;
}

XLA_FFI_Handler* csr_maxplus_batched_unweighted_handler() noexcept
{
    return SpringmvMaxplusBatchedUnweighted;
}

XLA_FFI_Handler* csr_maxplus_batched_weighted_handler() noexcept
{
    return SpringmvMaxplusBatchedWeighted;
}

}  // namespace springmv::jax_ffi
