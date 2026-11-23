// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_sampling.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static_assert(sizeof(bool) == sizeof(std::uint8_t));

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

    explicit DeviceBuffer(const std::vector<T>& values)
        : DeviceBuffer(values.size())
    {
        copy_from(values);
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    void copy_from(const std::vector<T>& values)
    {
        if (values.size() != size_) {
            throw std::runtime_error("device-buffer size mismatch");
        }
        if (!values.empty()) {
            CHECK_CUDA(cudaMemcpy(data_, values.data(),
                                  values.size() * sizeof(T),
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

template <typename T>
void compare(const std::string& name, const std::vector<T>& actual,
             const std::vector<T>& expected)
{
    if (actual != expected) {
        throw std::runtime_error(name + " mismatch");
    }
}

void run()
{
    const std::vector<std::int32_t> row_offsets{0, 3, 3, 5, 7, 10};
    const std::vector<std::int32_t> column_indices{
        0, 2, 2, 1, 3, 0, 3, 1, 2, 3};
    const std::vector<std::int32_t> transpose_row_offsets{0, 2, 4, 7, 10};
    const std::vector<std::int32_t> transpose_column_indices{
        0, 3, 2, 4, 0, 0, 4, 2, 3, 4};
    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> weights{
        0.0F, 1.0F, 1.0F, 0.0F, 0.0F, infinity, infinity, nan,
        -infinity, -infinity};
    const std::vector<float> gumbels{
        0.0F, 1.0F, 1.0F, 0.0F, 2.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    const std::vector<std::uint8_t> active_parent{1, 1, 1, 1, 0};
    const std::vector<std::int32_t> expected_selected{2, -1, 3, 0, 2};
    const std::vector<std::uint8_t> expected_active{1, 0, 1, 1};

    DeviceBuffer<std::int32_t> d_rows(row_offsets);
    DeviceBuffer<std::int32_t> d_columns(column_indices);
    DeviceBuffer<std::int32_t> d_transpose_rows(transpose_row_offsets);
    DeviceBuffer<std::int32_t> d_transpose_columns(
        transpose_column_indices);
    DeviceBuffer<float> d_weights(weights);
    DeviceBuffer<float> d_gumbels(gumbels);
    DeviceBuffer<std::uint8_t> d_active_parent(active_parent);
    DeviceBuffer<std::int32_t> d_selected(expected_selected.size());
    DeviceBuffer<std::uint8_t> d_active_child(expected_active.size());

    for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
        const springmv::CsrKernelOptions options{3, subwarp};
        const springmv::CsrKernelOptions transpose_options{3, subwarp};
        CHECK_CUDA(springmv::csr_logsumexp_sample_select(
            5, d_rows.data(), d_columns.data(), d_weights.data(),
            d_gumbels.data(), d_selected.data(), options));
        CHECK_CUDA(cudaDeviceSynchronize());
        compare("selected child", d_selected.copy_to_host(),
                expected_selected);

        CHECK_CUDA(springmv::csr_sample_propagate(
            5, 4, d_transpose_rows.data(), d_transpose_columns.data(),
            d_selected.data(),
            reinterpret_cast<const bool*>(d_active_parent.data()),
            reinterpret_cast<bool*>(d_active_child.data()),
            transpose_options));
        CHECK_CUDA(cudaDeviceSynchronize());
        compare("active child", d_active_child.copy_to_host(),
                expected_active);
    }

    const std::vector<float> batched_weights{
        weights[0], weights[1], weights[2], weights[3], weights[4],
        weights[5], weights[6], weights[7], weights[8], weights[9],
        5.0F, 0.0F, 0.0F, 3.0F, 0.0F,
        0.0F, 4.0F, nan, nan, nan};
    const std::vector<float> batched_gumbels{
        gumbels[0], gumbels[1], gumbels[2], gumbels[3], gumbels[4],
        gumbels[5], gumbels[6], gumbels[7], gumbels[8], gumbels[9],
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    const std::vector<std::uint8_t> batched_active{
        1, 1, 1, 1, 0, 1, 0, 1, 1, 1};
    const std::vector<std::int32_t> expected_batched_selected{
        2, -1, 3, 0, 2, 0, -1, 1, 3, -1};
    const std::vector<std::uint8_t> expected_batched_active{
        1, 0, 1, 1, 1, 1, 0, 1};

    DeviceBuffer<float> d_batched_weights(batched_weights);
    DeviceBuffer<float> d_batched_gumbels(batched_gumbels);
    DeviceBuffer<std::uint8_t> d_batched_active(batched_active);
    DeviceBuffer<std::int32_t> d_batched_selected(
        expected_batched_selected.size());
    DeviceBuffer<std::uint8_t> d_batched_output(
        expected_batched_active.size());

    CHECK_CUDA(springmv::csr_logsumexp_sample_batched(
        2, 5, 4, d_rows.data(), d_columns.data(), d_transpose_rows.data(),
        d_transpose_columns.data(), d_batched_weights.data(), 10,
        d_batched_gumbels.data(), 10,
        reinterpret_cast<const bool*>(d_batched_active.data()), 5,
        d_batched_selected.data(),
        reinterpret_cast<bool*>(d_batched_output.data()),
        springmv::CsrKernelOptions{3, 0},
        springmv::CsrKernelOptions{3, 0}));
    CHECK_CUDA(cudaDeviceSynchronize());
    compare("batched selected child", d_batched_selected.copy_to_host(),
            expected_batched_selected);
    compare("batched active child", d_batched_output.copy_to_host(),
            expected_batched_active);
}

}  // namespace

int main()
{
    try {
        run();
        std::cout << "PASS CSR sampling\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
