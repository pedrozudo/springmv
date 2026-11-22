// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_spmv.hpp"
#include "springmv/csr_spmv_backward.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expression)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{expression} + ": " +
                                 cudaGetErrorString(status));
    }
}

#define CHECK_CUDA(expression) check_cuda((expression), #expression)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_{size}
    {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_),
                              std::max<std::size_t>(size, 1) * sizeof(T)));
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    void copy_from(const std::vector<T>& source)
    {
        if (!source.empty()) {
            CHECK_CUDA(cudaMemcpy(data_, source.data(),
                                  source.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

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

void compare(const std::vector<float>& actual,
             const std::vector<float>& expected, const std::string& operation,
             std::int32_t subwarp)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(operation + " returned the wrong size");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto tolerance = 2.0e-6F + 2.0e-6F * std::abs(expected[i]);
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(
                operation + " failed for subwarp " +
                std::to_string(subwarp) + " at entry " +
                std::to_string(i) + ": expected " +
                std::to_string(expected[i]) + ", got " +
                std::to_string(actual[i]));
        }
    }
}

void run_backward_case()
{
    constexpr std::int32_t num_rows = 4;
    constexpr std::int32_t num_cols = 5;
    const std::vector<std::int32_t> row_offsets{0, 3, 3, 5, 7};
    const std::vector<std::int32_t> column_indices{0, 2, 2, 1, 4, 0, 3};
    const std::vector<std::int32_t> transpose_row_offsets{0, 2, 3, 5, 6, 7};
    const std::vector<std::int32_t> transpose_column_indices{0, 3, 2, 0,
                                                             0, 3, 2};
    const std::vector<std::int32_t> transpose_permutation{0, 5, 3, 1,
                                                          2, 6, 4};
    const std::vector<float> values{2.0F, -1.0F, 0.5F, 3.0F,
                                    -2.0F, 1.5F, 0.25F};
    const std::vector<float> x{1.0F, -2.0F, 3.0F, 4.0F, -1.0F};
    const std::vector<float> output_cotangent{0.5F, -2.0F, 3.0F, -1.0F};
    const std::vector<float> expected_input{-0.5F, 9.0F, -0.25F, -0.25F,
                                             -6.0F};
    const std::vector<float> expected_unweighted{-0.5F, 3.0F, 1.0F, -1.0F,
                                                  3.0F};
    const std::vector<float> expected_values{0.5F, 1.5F, 1.5F, -6.0F,
                                              -3.0F, -1.0F, -4.0F};

    DeviceBuffer<std::int32_t> d_row_offsets(row_offsets.size());
    DeviceBuffer<std::int32_t> d_column_indices(column_indices.size());
    DeviceBuffer<std::int32_t> d_transpose_row_offsets(
        transpose_row_offsets.size());
    DeviceBuffer<std::int32_t> d_transpose_column_indices(
        transpose_column_indices.size());
    DeviceBuffer<std::int32_t> d_transpose_permutation(
        transpose_permutation.size());
    DeviceBuffer<float> d_values(values.size());
    DeviceBuffer<float> d_x(x.size());
    DeviceBuffer<float> d_output_cotangent(output_cotangent.size());
    DeviceBuffer<float> d_input_cotangent(num_cols);
    DeviceBuffer<float> d_values_cotangent(values.size());

    d_row_offsets.copy_from(row_offsets);
    d_column_indices.copy_from(column_indices);
    d_transpose_row_offsets.copy_from(transpose_row_offsets);
    d_transpose_column_indices.copy_from(transpose_column_indices);
    d_transpose_permutation.copy_from(transpose_permutation);
    d_values.copy_from(values);
    d_x.copy_from(x);
    d_output_cotangent.copy_from(output_cotangent);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
        CHECK_CUDA(springmv::csr_spmv_input_vjp(
            num_cols, d_transpose_row_offsets.data(),
            d_transpose_column_indices.data(), d_transpose_permutation.data(),
            d_values.data(), d_output_cotangent.data(),
            d_input_cotangent.data(), {2, subwarp}, stream));
        CHECK_CUDA(springmv::csr_spmv_values_vjp(
            num_rows, d_row_offsets.data(), d_column_indices.data(),
            d_x.data(), d_output_cotangent.data(), d_values_cotangent.data(),
            {3, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_input_cotangent.copy_to_host(), expected_input,
                "weighted input VJP", subwarp);
        compare(d_values_cotangent.copy_to_host(), expected_values,
                "values VJP", subwarp);

        CHECK_CUDA(springmv::csr_spmv(
            num_cols, d_transpose_row_offsets.data(),
            d_transpose_column_indices.data(), d_output_cotangent.data(),
            d_input_cotangent.data(), {2, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_input_cotangent.copy_to_host(), expected_unweighted,
                "unweighted input VJP", subwarp);
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void run_empty_case()
{
    DeviceBuffer<std::int32_t> row_offsets(3);
    DeviceBuffer<std::int32_t> transpose_row_offsets(4);
    DeviceBuffer<float> input_cotangent(3);
    row_offsets.copy_from({0, 0, 0});
    transpose_row_offsets.copy_from({0, 0, 0, 0});

    CHECK_CUDA(springmv::csr_spmv_input_vjp(
        3, transpose_row_offsets.data(), nullptr, nullptr, nullptr, nullptr,
        input_cotangent.data(), {0, 0}));
    CHECK_CUDA(cudaDeviceSynchronize());
    compare(input_cotangent.copy_to_host(), {0.0F, 0.0F, 0.0F},
            "empty input VJP", 0);
    CHECK_CUDA(springmv::csr_spmv_values_vjp(
        2, row_offsets.data(), nullptr, nullptr, nullptr, nullptr, {0, 0}));
}

}  // namespace

int main()
{
    try {
        run_backward_case();
        run_empty_case();
        std::cout << "All arithmetic SpMV backward tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
