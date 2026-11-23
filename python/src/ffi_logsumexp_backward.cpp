// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "ffi_common.hpp"
#include "ffi_handlers.hpp"

#include "springmv/csr_logsumexp_backward.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace springmv::jax_ffi {
namespace {

ffi::Error validate_rows(S32Buffer row_offsets, F32Buffer output,
                         F32Buffer output_cotangent,
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
    const auto rows = static_cast<std::size_t>(*num_rows);
    if (output.element_count() != rows ||
        output_cotangent.element_count() != rows) {
        return ffi::Error::InvalidArgument(
            "output arrays must have one entry per row");
    }
    return ffi::Error::Success();
}

ffi::Error LogsumexpJvpUnweightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32Buffer log_x, F32Buffer output, F32Buffer input_tangent,
    F32Result output_tangent, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_rows(row_offsets, output, output, &num_rows);
    if (error.failure()) {
        return error;
    }
    if (output_tangent->element_count() !=
        static_cast<std::size_t>(num_rows)) {
        return ffi::Error::InvalidArgument(
            "output tangent must have one entry per row");
    }
    if (input_tangent.element_count() != log_x.element_count()) {
        return ffi::Error::InvalidArgument(
            "log_x and its tangent must have equal lengths");
    }
    return cuda_error(
        csr_logsumexp_jvp(
            num_rows, row_offsets.typed_data(), column_indices.typed_data(),
            log_x.typed_data(), output.typed_data(),
            input_tangent.typed_data(), output_tangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_jvp");
}

ffi::Error LogsumexpJvpBatchedUnweightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32BufferR2 log_x, F32BufferR2 output, F32BufferR2 input_tangent,
    F32ResultR2 output_tangent, std::int32_t batch_size,
    std::int32_t num_columns, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    if (batch_size < 0 || num_columns < 0 ||
        row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "batch_size and num_columns must be nonnegative and "
            "row_offsets nonempty");
    }
    std::int32_t num_rows{};
    error = checked_int32_size(row_offsets.element_count() - 1,
                               "number of rows", &num_rows);
    if (error.failure()) {
        return error;
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    const auto expected_inputs =
        batch * static_cast<std::size_t>(num_columns);
    const auto expected_outputs =
        batch * static_cast<std::size_t>(num_rows);
    if (log_x.element_count() != expected_inputs ||
        input_tangent.element_count() != expected_inputs ||
        output.element_count() != expected_outputs ||
        output_tangent->element_count() != expected_outputs) {
        return ffi::Error::InvalidArgument(
            "batched JVP buffers have inconsistent sizes");
    }
    return cuda_error(
        csr_logsumexp_jvp_batched(
            batch_size, num_rows, num_columns, row_offsets.typed_data(),
            column_indices.typed_data(), log_x.typed_data(),
            output.typed_data(), input_tangent.typed_data(),
            output_tangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_jvp_batched");
}

ffi::Error LogsumexpJvpWeightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32Buffer log_values, F32Buffer log_x, F32Buffer output,
    F32Buffer values_tangent, F32Buffer input_tangent,
    F32Result output_tangent, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_rows(row_offsets, output, output, &num_rows);
    if (error.failure()) {
        return error;
    }
    const auto nonzeros = column_indices.element_count();
    if (log_values.element_count() != nonzeros ||
        values_tangent.element_count() != nonzeros) {
        return ffi::Error::InvalidArgument(
            "log values, their tangent, and indices must have equal lengths");
    }
    if (input_tangent.element_count() != log_x.element_count()) {
        return ffi::Error::InvalidArgument(
            "log_x and its tangent must have equal lengths");
    }
    if (output_tangent->element_count() !=
        static_cast<std::size_t>(num_rows)) {
        return ffi::Error::InvalidArgument(
            "output tangent must have one entry per row");
    }
    return cuda_error(
        csr_logsumexp_jvp(
            num_rows, row_offsets.typed_data(), column_indices.typed_data(),
            log_values.typed_data(), log_x.typed_data(), output.typed_data(),
            values_tangent.typed_data(), input_tangent.typed_data(),
            output_tangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_jvp");
}

ffi::Error validate_input_vjp(
    S32Buffer row_offsets, S32Buffer column_indices,
    S32Buffer transpose_row_offsets, S32Buffer transpose_column_indices,
    F32Buffer log_x, F32Buffer output, F32Buffer output_cotangent,
    F32Result input_cotangent, std::int32_t* num_columns)
{
    std::int32_t num_rows{};
    auto error = validate_rows(row_offsets, output, output_cotangent,
                               &num_rows);
    if (error.failure()) {
        return error;
    }
    if (transpose_row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "transpose_row_offsets must contain at least one entry");
    }
    error = checked_int32_size(transpose_row_offsets.element_count() - 1,
                               "number of columns", num_columns);
    if (error.failure()) {
        return error;
    }
    const auto nonzeros = column_indices.element_count();
    if (transpose_column_indices.element_count() != nonzeros) {
        return ffi::Error::InvalidArgument(
            "original and transposed indices must have equal lengths");
    }
    const auto columns = static_cast<std::size_t>(*num_columns);
    if (log_x.element_count() != columns ||
        input_cotangent->element_count() != columns) {
        return ffi::Error::InvalidArgument(
            "input arrays must have one entry per column");
    }
    return ffi::Error::Success();
}

ffi::Error LogsumexpInputVjpUnweightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    S32Buffer transpose_row_offsets, S32Buffer transpose_column_indices,
    F32Buffer log_x, F32Buffer output, F32Buffer output_cotangent,
    F32Result input_cotangent, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_columns{};
    error = validate_input_vjp(
        row_offsets, column_indices, transpose_row_offsets,
        transpose_column_indices, log_x, output, output_cotangent,
        input_cotangent, &num_columns);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        csr_logsumexp_input_vjp(
            num_columns, row_offsets.typed_data(),
            column_indices.typed_data(), transpose_row_offsets.typed_data(),
            transpose_column_indices.typed_data(), log_x.typed_data(),
            output.typed_data(), output_cotangent.typed_data(),
            input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_input_vjp");
}

ffi::Error LogsumexpInputVjpBatchedUnweightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    S32Buffer transpose_row_offsets,
    S32Buffer transpose_column_indices, F32BufferR2 log_x,
    F32BufferR2 output, F32BufferR2 output_cotangent,
    F32ResultR2 input_cotangent, std::int32_t batch_size,
    std::int32_t max_nnz_per_row, std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    if (batch_size < 0 || row_offsets.element_count() == 0 ||
        transpose_row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "batch_size must be nonnegative and CSR offsets nonempty");
    }
    std::int32_t num_rows{};
    std::int32_t num_columns{};
    error = checked_int32_size(row_offsets.element_count() - 1,
                               "number of rows", &num_rows);
    if (error.failure()) {
        return error;
    }
    error = checked_int32_size(transpose_row_offsets.element_count() - 1,
                               "number of columns", &num_columns);
    if (error.failure()) {
        return error;
    }
    if (column_indices.element_count() !=
        transpose_column_indices.element_count()) {
        return ffi::Error::InvalidArgument(
            "original and transposed indices must have equal lengths");
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    const auto expected_inputs =
        batch * static_cast<std::size_t>(num_columns);
    const auto expected_outputs =
        batch * static_cast<std::size_t>(num_rows);
    if (log_x.element_count() != expected_inputs ||
        input_cotangent->element_count() != expected_inputs ||
        output.element_count() != expected_outputs ||
        output_cotangent.element_count() != expected_outputs) {
        return ffi::Error::InvalidArgument(
            "batched input-VJP buffers have inconsistent sizes");
    }
    return cuda_error(
        csr_logsumexp_input_vjp_batched(
            batch_size, num_rows, num_columns, row_offsets.typed_data(),
            column_indices.typed_data(),
            transpose_row_offsets.typed_data(),
            transpose_column_indices.typed_data(), log_x.typed_data(),
            output.typed_data(), output_cotangent.typed_data(),
            input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_input_vjp_batched");
}

ffi::Error LogsumexpInputVjpWeightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    S32Buffer transpose_row_offsets, S32Buffer transpose_column_indices,
    S32Buffer transpose_permutation, F32Buffer log_values, F32Buffer log_x,
    F32Buffer output, F32Buffer output_cotangent,
    F32Result input_cotangent, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_columns{};
    error = validate_input_vjp(
        row_offsets, column_indices, transpose_row_offsets,
        transpose_column_indices, log_x, output, output_cotangent,
        input_cotangent, &num_columns);
    if (error.failure()) {
        return error;
    }
    const auto nonzeros = column_indices.element_count();
    if (transpose_permutation.element_count() != nonzeros ||
        log_values.element_count() != nonzeros) {
        return ffi::Error::InvalidArgument(
            "permutation, log values, and indices must have equal lengths");
    }
    return cuda_error(
        csr_logsumexp_input_vjp(
            num_columns, row_offsets.typed_data(),
            column_indices.typed_data(), transpose_row_offsets.typed_data(),
            transpose_column_indices.typed_data(),
            transpose_permutation.typed_data(), log_values.typed_data(),
            log_x.typed_data(), output.typed_data(),
            output_cotangent.typed_data(), input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_input_vjp");
}

ffi::Error LogsumexpValuesVjpImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32Buffer log_values, F32Buffer log_x, F32Buffer output,
    F32Buffer output_cotangent, F32Result values_cotangent,
    std::int32_t max_nnz_per_row, std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_rows(row_offsets, output, output_cotangent, &num_rows);
    if (error.failure()) {
        return error;
    }
    const auto nonzeros = column_indices.element_count();
    if (log_values.element_count() != nonzeros ||
        values_cotangent->element_count() != nonzeros) {
        return ffi::Error::InvalidArgument(
            "log values, their cotangent, and indices must have equal lengths");
    }
    return cuda_error(
        csr_logsumexp_values_vjp(
            num_rows, row_offsets.typed_data(), column_indices.typed_data(),
            log_values.typed_data(), log_x.typed_data(), output.typed_data(),
            output_cotangent.typed_data(), values_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_values_vjp");
}

ffi::Error LogsumexpInputVjpScatterImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32Buffer log_values, F32Buffer log_x, F32Buffer output,
    F32Buffer output_cotangent, F32Result input_cotangent,
    std::int32_t max_nnz_per_row, std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_rows(row_offsets, output, output_cotangent, &num_rows);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_columns{};
    error = checked_int32_size(input_cotangent->element_count(),
                               "number of columns", &num_columns);
    if (error.failure()) {
        return error;
    }
    const auto nonzeros = column_indices.element_count();
    if (log_values.element_count() != nonzeros) {
        return ffi::Error::InvalidArgument(
            "log values and column_indices must have equal lengths");
    }
    if (log_x.element_count() != static_cast<std::size_t>(num_columns)) {
        return ffi::Error::InvalidArgument(
            "log_x must have one entry per column");
    }
    return cuda_error(
        csr_logsumexp_input_vjp_scatter(
            num_rows, num_columns, row_offsets.typed_data(),
            column_indices.typed_data(), log_values.typed_data(),
            log_x.typed_data(), output.typed_data(),
            output_cotangent.typed_data(), input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_input_vjp_scatter");
}

ffi::Error LogsumexpInputVjpScatterUnweightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32Buffer log_x, F32Buffer output, F32Buffer output_cotangent,
    F32Result input_cotangent, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_rows{};
    error = validate_rows(row_offsets, output, output_cotangent, &num_rows);
    if (error.failure()) {
        return error;
    }
    std::int32_t num_columns{};
    error = checked_int32_size(input_cotangent->element_count(),
                               "number of columns", &num_columns);
    if (error.failure()) {
        return error;
    }
    if (log_x.element_count() != static_cast<std::size_t>(num_columns)) {
        return ffi::Error::InvalidArgument(
            "log_x must have one entry per column");
    }
    return cuda_error(
        csr_logsumexp_input_vjp_scatter(
            num_rows, num_columns, row_offsets.typed_data(),
            column_indices.typed_data(), log_x.typed_data(),
            output.typed_data(), output_cotangent.typed_data(),
            input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_input_vjp_scatter");
}

ffi::Error LogsumexpInputVjpScatterBatchedUnweightedImpl(
    cudaStream_t stream, S32Buffer row_offsets, S32Buffer column_indices,
    F32BufferR2 log_x, F32BufferR2 output,
    F32BufferR2 output_cotangent, F32ResultR2 input_cotangent,
    std::int32_t batch_size, std::int32_t max_nnz_per_row,
    std::int32_t subwarp_size)
{
    auto error = validate_kernel_options(max_nnz_per_row, subwarp_size);
    if (error.failure()) {
        return error;
    }
    if (batch_size < 0 || row_offsets.element_count() == 0) {
        return ffi::Error::InvalidArgument(
            "batch_size must be nonnegative and row_offsets nonempty");
    }
    std::int32_t num_rows{};
    error = checked_int32_size(row_offsets.element_count() - 1,
                               "number of rows", &num_rows);
    if (error.failure()) {
        return error;
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    const auto expected_outputs =
        batch * static_cast<std::size_t>(num_rows);
    if (output.element_count() != expected_outputs ||
        output_cotangent.element_count() != expected_outputs) {
        return ffi::Error::InvalidArgument(
            "output buffers must have batch_size times num_rows entries");
    }
    if (batch_size == 0) {
        if (log_x.element_count() != 0 ||
            input_cotangent->element_count() != 0) {
            return ffi::Error::InvalidArgument(
                "input buffers must be empty for an empty batch");
        }
        return ffi::Error::Success();
    }
    if (log_x.element_count() != input_cotangent->element_count() ||
        log_x.element_count() % batch != 0) {
        return ffi::Error::InvalidArgument(
            "input buffers must have equal batch-divisible sizes");
    }
    std::int32_t num_columns{};
    error = checked_int32_size(log_x.element_count() / batch,
                               "number of columns", &num_columns);
    if (error.failure()) {
        return error;
    }
    return cuda_error(
        csr_logsumexp_input_vjp_scatter_batched(
            batch_size, num_rows, num_columns, row_offsets.typed_data(),
            column_indices.typed_data(), log_x.typed_data(),
            output.typed_data(), output_cotangent.typed_data(),
            input_cotangent->typed_data(),
            CsrKernelOptions{max_nnz_per_row, subwarp_size}, stream),
        "csr_logsumexp_input_vjp_scatter_batched");
}

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpJvpUnweighted, LogsumexpJvpUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpJvpBatchedUnweighted,
    LogsumexpJvpBatchedUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32BufferR2>()
        .Arg<F32BufferR2>()
        .Arg<F32BufferR2>()
        .Ret<F32BufferR2>()
        .Attr<std::int32_t>("batch_size")
        .Attr<std::int32_t>("num_columns")
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpJvpWeighted, LogsumexpJvpWeightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpInputVjpUnweighted,
    LogsumexpInputVjpUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpInputVjpBatchedUnweighted,
    LogsumexpInputVjpBatchedUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32BufferR2>()
        .Arg<F32BufferR2>()
        .Arg<F32BufferR2>()
        .Ret<F32BufferR2>()
        .Attr<std::int32_t>("batch_size")
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpInputVjpWeighted, LogsumexpInputVjpWeightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpValuesVjp, LogsumexpValuesVjpImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpInputVjpScatter, LogsumexpInputVjpScatterImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpInputVjpScatterUnweighted,
    LogsumexpInputVjpScatterUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Arg<F32Buffer>()
        .Ret<F32Buffer>()
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

XLA_FFI_DEFINE_HANDLER(
    SpringmvLogsumexpInputVjpScatterBatchedUnweighted,
    LogsumexpInputVjpScatterBatchedUnweightedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<S32Buffer>()
        .Arg<S32Buffer>()
        .Arg<F32BufferR2>()
        .Arg<F32BufferR2>()
        .Arg<F32BufferR2>()
        .Ret<F32BufferR2>()
        .Attr<std::int32_t>("batch_size")
        .Attr<std::int32_t>("max_nnz_per_row")
        .Attr<std::int32_t>("subwarp_size"));

}  // namespace

XLA_FFI_Handler* csr_logsumexp_jvp_unweighted_handler() noexcept
{
    return SpringmvLogsumexpJvpUnweighted;
}

XLA_FFI_Handler* csr_logsumexp_jvp_batched_unweighted_handler() noexcept
{
    return SpringmvLogsumexpJvpBatchedUnweighted;
}

XLA_FFI_Handler* csr_logsumexp_jvp_weighted_handler() noexcept
{
    return SpringmvLogsumexpJvpWeighted;
}

XLA_FFI_Handler* csr_logsumexp_input_vjp_unweighted_handler() noexcept
{
    return SpringmvLogsumexpInputVjpUnweighted;
}

XLA_FFI_Handler*
csr_logsumexp_input_vjp_batched_unweighted_handler() noexcept
{
    return SpringmvLogsumexpInputVjpBatchedUnweighted;
}

XLA_FFI_Handler* csr_logsumexp_input_vjp_weighted_handler() noexcept
{
    return SpringmvLogsumexpInputVjpWeighted;
}

XLA_FFI_Handler* csr_logsumexp_values_vjp_handler() noexcept
{
    return SpringmvLogsumexpValuesVjp;
}

XLA_FFI_Handler* csr_logsumexp_input_vjp_scatter_handler() noexcept
{
    return SpringmvLogsumexpInputVjpScatter;
}

XLA_FFI_Handler*
csr_logsumexp_input_vjp_scatter_unweighted_handler() noexcept
{
    return SpringmvLogsumexpInputVjpScatterUnweighted;
}

XLA_FFI_Handler*
csr_logsumexp_input_vjp_scatter_batched_unweighted_handler() noexcept
{
    return SpringmvLogsumexpInputVjpScatterBatchedUnweighted;
}

}  // namespace springmv::jax_ffi
