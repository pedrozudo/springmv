// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "xla/ffi/api/ffi.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace springmv::jax_ffi {

namespace ffi = xla::ffi;

using S32Buffer = ffi::BufferR1<ffi::DataType::S32>;
using F32Buffer = ffi::BufferR1<ffi::DataType::F32>;
using F32BufferR2 = ffi::BufferR2<ffi::DataType::F32>;
using F32AnyBuffer = ffi::Buffer<ffi::DataType::F32>;
using PredBuffer = ffi::BufferR1<ffi::DataType::PRED>;
using PredBufferR2 = ffi::BufferR2<ffi::DataType::PRED>;
using PredAnyBuffer = ffi::Buffer<ffi::DataType::PRED>;
using S32Result = ffi::ResultBufferR1<ffi::DataType::S32>;
using F32Result = ffi::ResultBufferR1<ffi::DataType::F32>;
using F32ResultR2 = ffi::ResultBufferR2<ffi::DataType::F32>;
using PredResult = ffi::ResultBufferR1<ffi::DataType::PRED>;
using PredResultR2 = ffi::ResultBufferR2<ffi::DataType::PRED>;

inline ffi::Error cuda_error(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess) {
        return ffi::Error::Success();
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status) << " ("
            << static_cast<int>(status) << ')';
    if (status == cudaErrorInvalidValue ||
        status == cudaErrorInvalidDevicePointer) {
        return ffi::Error::InvalidArgument(message.str());
    }
    return ffi::Error::Internal(message.str());
}

inline ffi::Error checked_int32_size(std::size_t value, const char* name,
                                     std::int32_t* result)
{
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
        return ffi::Error::InvalidArgument(std::string{name} +
                                           " is too large for int32 CSR");
    }
    *result = static_cast<std::int32_t>(value);
    return ffi::Error::Success();
}

inline ffi::Error validate_kernel_options(std::int32_t max_nnz_per_row,
                                          std::int32_t subwarp_size)
{
    if (max_nnz_per_row < 0) {
        return ffi::Error::InvalidArgument(
            "max_nnz_per_row must be nonnegative");
    }
    if (subwarp_size != 0 && subwarp_size != 1 && subwarp_size != 2 &&
        subwarp_size != 4 && subwarp_size != 8 && subwarp_size != 16 &&
        subwarp_size != 32) {
        return ffi::Error::InvalidArgument(
            "subwarp_size must be 0, 1, 2, 4, 8, 16, or 32");
    }
    return ffi::Error::Success();
}

}  // namespace springmv::jax_ffi
