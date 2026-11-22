// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_batched.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expression)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " +
                                 cudaGetErrorString(status));
    }
}

#define CHECK_CUDA(expression) check_cuda((expression), #expression)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<T>& values) : size_{values.size()}
    {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_),
                              std::max<std::size_t>(size_, 1) * sizeof(T)));
        if (!values.empty()) {
            CHECK_CUDA(cudaMemcpy(data_, values.data(), size_ * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

    explicit DeviceBuffer(std::size_t size) : size_{size}
    {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_),
                              std::max<std::size_t>(size_, 1) * sizeof(T)));
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }

    std::vector<T> copy_to_host() const
    {
        std::vector<T> result(size_);
        if (size_ > 0) {
            CHECK_CUDA(cudaMemcpy(result.data(), data_, size_ * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return result;
    }

private:
    T* data_{};
    std::size_t size_{};
};

void compare(const std::string& name, const std::vector<float>& actual,
             const std::vector<float>& expected)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " size mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::isinf(expected[index])) {
            if (!std::isinf(actual[index]) ||
                std::signbit(actual[index]) != std::signbit(expected[index])) {
                throw std::runtime_error(name + " infinity mismatch");
            }
        } else if (std::abs(actual[index] - expected[index]) > 2.0e-6F) {
            throw std::runtime_error(name + " mismatch at " +
                                     std::to_string(index));
        }
    }
}

template <typename Operation>
std::vector<float> evaluate(Operation operation)
{
    const std::vector<std::int32_t> row_offsets{0, 2, 3, 3};
    const std::vector<std::int32_t> column_indices{0, 2, 1};
    const std::vector<float> values{0.5F, -1.0F, 2.0F};
    const std::vector<float> x{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    DeviceBuffer<std::int32_t> d_row_offsets(row_offsets);
    DeviceBuffer<std::int32_t> d_column_indices(column_indices);
    DeviceBuffer<float> d_values(values);
    DeviceBuffer<float> d_x(x);
    DeviceBuffer<float> d_output(6);
    CHECK_CUDA(operation(d_row_offsets.data(), d_column_indices.data(),
                         d_values.data(), d_x.data(), d_output.data()));
    CHECK_CUDA(cudaDeviceSynchronize());
    return d_output.copy_to_host();
}

void run()
{
    const springmv::CsrKernelOptions options{2, 0};
    compare(
        "arithmetic unweighted",
        evaluate([&](const auto* rows, const auto* columns, const float*,
                     const float* x, float* y) {
            return springmv::csr_spmv_batched(2, 3, rows, columns, x, 3, y,
                                               options);
        }),
        {4.0F, 2.0F, 0.0F, 10.0F, 5.0F, 0.0F});
    compare(
        "arithmetic weighted",
        evaluate([&](const auto* rows, const auto* columns,
                     const float* values, const float* x, float* y) {
            return springmv::csr_spmv_batched(
                2, 3, rows, columns, values, 0, x, 3, y, options);
        }),
        {-2.5F, 4.0F, 0.0F, -4.0F, 10.0F, 0.0F});

    const auto negative_infinity = -std::numeric_limits<float>::infinity();
    compare(
        "logsumexp unweighted",
        evaluate([&](const auto* rows, const auto* columns, const float*,
                     const float* x, float* y) {
            return springmv::csr_logsumexp_batched(
                2, 3, rows, columns, x, 3, y, options);
        }),
        {std::log(std::exp(1.0F) + std::exp(3.0F)), 2.0F,
         negative_infinity, std::log(std::exp(4.0F) + std::exp(6.0F)), 5.0F,
         negative_infinity});
    compare(
        "minplus weighted",
        evaluate([&](const auto* rows, const auto* columns,
                     const float* values, const float* x, float* y) {
            return springmv::csr_minplus_batched(
                2, 3, rows, columns, values, 0, x, 3, y, options);
        }),
        {1.5F, 4.0F, std::numeric_limits<float>::infinity(), 4.5F, 7.0F,
         std::numeric_limits<float>::infinity()});
    compare(
        "maxplus unweighted",
        evaluate([&](const auto* rows, const auto* columns, const float*,
                     const float* x, float* y) {
            return springmv::csr_maxplus_batched(
                2, 3, rows, columns, x, 3, y, options);
        }),
        {3.0F, 2.0F, negative_infinity, 6.0F, 5.0F,
         negative_infinity});
}

}  // namespace

int main()
{
    try {
        run();
        std::cout << "PASS batched CSR reductions\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
