// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_logsumexp_backward.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

struct Reference {
    std::vector<float> output;
    std::vector<float> jvp;
    std::vector<float> values_vjp;
    std::vector<float> input_vjp;
};

float term(std::int32_t edge,
           const std::vector<std::int32_t>& column_indices,
           const std::vector<float>* log_values,
           const std::vector<float>& log_x)
{
    const auto input = log_x[column_indices[edge]];
    return log_values == nullptr ? input : (*log_values)[edge] + input;
}

float row_output(std::int32_t begin, std::int32_t end,
                 const std::vector<std::int32_t>& column_indices,
                 const std::vector<float>* log_values,
                 const std::vector<float>& log_x)
{
    if (begin == end) {
        return -std::numeric_limits<float>::infinity();
    }
    auto maximum = -std::numeric_limits<double>::infinity();
    for (auto edge = begin; edge < end; ++edge) {
        const auto value = term(edge, column_indices, log_values, log_x);
        if (std::isnan(value)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        maximum = std::max(maximum, static_cast<double>(value));
    }
    if (std::isinf(maximum)) {
        return static_cast<float>(maximum);
    }
    double scaled_sum{};
    for (auto edge = begin; edge < end; ++edge) {
        scaled_sum +=
            std::exp(static_cast<double>(term(edge, column_indices,
                                              log_values, log_x)) -
                     maximum);
    }
    return static_cast<float>(maximum + std::log(scaled_sum));
}

float factor(float value, float output, std::int32_t row_length,
             std::int32_t positive_infinities)
{
    if (std::isnan(value) || std::isnan(output)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (output == std::numeric_limits<float>::infinity()) {
        return value == std::numeric_limits<float>::infinity()
                   ? 1.0F / static_cast<float>(positive_infinities)
                   : 0.0F;
    }
    if (output == -std::numeric_limits<float>::infinity()) {
        return 1.0F / static_cast<float>(row_length);
    }
    return std::exp(value - output);
}

Reference reference(
    const std::vector<std::int32_t>& row_offsets,
    const std::vector<std::int32_t>& column_indices,
    const std::vector<float>* log_values, const std::vector<float>& log_x,
    const std::vector<float>* values_tangent,
    const std::vector<float>& input_tangent,
    const std::vector<float>& output_cotangent)
{
    const auto num_rows = row_offsets.size() - 1;
    Reference result;
    result.output.resize(num_rows);
    result.jvp.assign(num_rows, 0.0F);
    result.values_vjp.resize(column_indices.size());
    result.input_vjp.assign(log_x.size(), 0.0F);

    for (std::size_t row = 0; row < num_rows; ++row) {
        const auto begin = row_offsets[row];
        const auto end = row_offsets[row + 1];
        result.output[row] =
            row_output(begin, end, column_indices, log_values, log_x);
        std::int32_t positive_infinities{};
        for (auto edge = begin; edge < end; ++edge) {
            positive_infinities +=
                term(edge, column_indices, log_values, log_x) ==
                std::numeric_limits<float>::infinity();
        }
        for (auto edge = begin; edge < end; ++edge) {
            const auto probability =
                factor(term(edge, column_indices, log_values, log_x),
                       result.output[row], end - begin, positive_infinities);
            auto tangent = input_tangent[column_indices[edge]];
            if (values_tangent != nullptr) {
                tangent += (*values_tangent)[edge];
            }
            result.jvp[row] += probability * tangent;
            result.values_vjp[edge] = output_cotangent[row] * probability;
            result.input_vjp[column_indices[edge]] +=
                output_cotangent[row] * probability;
        }
    }
    return result;
}

void compare(const std::vector<float>& actual,
             const std::vector<float>& expected, const std::string& operation,
             std::int32_t subwarp)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(operation + " returned the wrong size");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected[i])) {
            if (!std::isnan(actual[i])) {
                throw std::runtime_error(operation + " did not propagate NaN");
            }
            continue;
        }
        if (actual[i] == expected[i]) {
            continue;
        }
        const auto tolerance = 5.0e-6F + 5.0e-6F * std::abs(expected[i]);
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

void run_case(const std::vector<std::int32_t>& row_offsets,
              const std::vector<std::int32_t>& column_indices,
              const std::vector<std::int32_t>& transpose_row_offsets,
              const std::vector<std::int32_t>& transpose_column_indices,
              const std::vector<std::int32_t>& transpose_permutation,
              const std::vector<float>& log_values,
              const std::vector<float>& log_x,
              const std::vector<float>& values_tangent,
              const std::vector<float>& input_tangent,
              const std::vector<float>& output_cotangent,
              std::int32_t max_row_length,
              std::int32_t transpose_max_row_length)
{
    const auto num_rows =
        static_cast<std::int32_t>(row_offsets.size() - 1);
    const auto num_columns =
        static_cast<std::int32_t>(transpose_row_offsets.size() - 1);
    const auto weighted =
        reference(row_offsets, column_indices, &log_values, log_x,
                  &values_tangent, input_tangent, output_cotangent);
    const auto unweighted =
        reference(row_offsets, column_indices, nullptr, log_x, nullptr,
                  input_tangent, output_cotangent);
    const auto duplicate = [](const std::vector<float>& values) {
        auto result = values;
        result.insert(result.end(), values.begin(), values.end());
        return result;
    };
    const auto batched_log_x = duplicate(log_x);
    const auto batched_output = duplicate(unweighted.output);
    const auto batched_input_tangent = duplicate(input_tangent);
    const auto batched_output_cotangent = duplicate(output_cotangent);
    const auto batched_jvp = duplicate(unweighted.jvp);
    const auto batched_input_vjp = duplicate(unweighted.input_vjp);

    DeviceBuffer<std::int32_t> d_row_offsets(row_offsets.size());
    DeviceBuffer<std::int32_t> d_column_indices(column_indices.size());
    DeviceBuffer<std::int32_t> d_transpose_row_offsets(
        transpose_row_offsets.size());
    DeviceBuffer<std::int32_t> d_transpose_column_indices(
        transpose_column_indices.size());
    DeviceBuffer<std::int32_t> d_transpose_permutation(
        transpose_permutation.size());
    DeviceBuffer<float> d_log_values(log_values.size());
    DeviceBuffer<float> d_log_x(log_x.size());
    DeviceBuffer<float> d_output(num_rows);
    DeviceBuffer<float> d_values_tangent(values_tangent.size());
    DeviceBuffer<float> d_input_tangent(input_tangent.size());
    DeviceBuffer<float> d_output_cotangent(output_cotangent.size());
    DeviceBuffer<float> d_output_tangent(num_rows);
    DeviceBuffer<float> d_values_cotangent(log_values.size());
    DeviceBuffer<float> d_input_cotangent(num_columns);
    DeviceBuffer<float> d_batched_log_x(batched_log_x.size());
    DeviceBuffer<float> d_batched_output(batched_output.size());
    DeviceBuffer<float> d_batched_input_tangent(
        batched_input_tangent.size());
    DeviceBuffer<float> d_batched_output_cotangent(
        batched_output_cotangent.size());
    DeviceBuffer<float> d_batched_output_tangent(batched_jvp.size());
    DeviceBuffer<float> d_batched_input_cotangent(
        batched_input_vjp.size());

    d_row_offsets.copy_from(row_offsets);
    d_column_indices.copy_from(column_indices);
    d_transpose_row_offsets.copy_from(transpose_row_offsets);
    d_transpose_column_indices.copy_from(transpose_column_indices);
    d_transpose_permutation.copy_from(transpose_permutation);
    d_log_values.copy_from(log_values);
    d_log_x.copy_from(log_x);
    d_values_tangent.copy_from(values_tangent);
    d_input_tangent.copy_from(input_tangent);
    d_output_cotangent.copy_from(output_cotangent);
    d_batched_log_x.copy_from(batched_log_x);
    d_batched_output.copy_from(batched_output);
    d_batched_input_tangent.copy_from(batched_input_tangent);
    d_batched_output_cotangent.copy_from(batched_output_cotangent);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
        d_output.copy_from(weighted.output);
        CHECK_CUDA(springmv::csr_logsumexp_jvp(
            num_rows, d_row_offsets.data(), d_column_indices.data(),
            d_log_values.data(), d_log_x.data(), d_output.data(),
            d_values_tangent.data(), d_input_tangent.data(),
            d_output_tangent.data(), {max_row_length, subwarp}, stream));
        CHECK_CUDA(springmv::csr_logsumexp_values_vjp(
            num_rows, d_row_offsets.data(), d_column_indices.data(),
            d_log_values.data(), d_log_x.data(), d_output.data(),
            d_output_cotangent.data(), d_values_cotangent.data(),
            {max_row_length, subwarp}, stream));
        CHECK_CUDA(springmv::csr_logsumexp_input_vjp(
            num_columns, d_row_offsets.data(), d_column_indices.data(),
            d_transpose_row_offsets.data(),
            d_transpose_column_indices.data(),
            d_transpose_permutation.data(), d_log_values.data(),
            d_log_x.data(), d_output.data(), d_output_cotangent.data(),
            d_input_cotangent.data(),
            {transpose_max_row_length, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_output_tangent.copy_to_host(), weighted.jvp,
                "weighted JVP", subwarp);
        compare(d_values_cotangent.copy_to_host(), weighted.values_vjp,
                "values VJP", subwarp);
        compare(d_input_cotangent.copy_to_host(), weighted.input_vjp,
                "weighted input VJP", subwarp);

        CHECK_CUDA(springmv::csr_logsumexp_input_vjp_scatter(
            num_rows, num_columns, d_row_offsets.data(),
            d_column_indices.data(), d_log_values.data(), d_log_x.data(),
            d_output.data(), d_output_cotangent.data(),
            d_input_cotangent.data(), {max_row_length, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_input_cotangent.copy_to_host(), weighted.input_vjp,
                "weighted input VJP scatter", subwarp);

        d_output.copy_from(unweighted.output);
        CHECK_CUDA(springmv::csr_logsumexp_jvp(
            num_rows, d_row_offsets.data(), d_column_indices.data(),
            d_log_x.data(), d_output.data(), d_input_tangent.data(),
            d_output_tangent.data(), {max_row_length, subwarp}, stream));
        CHECK_CUDA(springmv::csr_logsumexp_input_vjp(
            num_columns, d_row_offsets.data(), d_column_indices.data(),
            d_transpose_row_offsets.data(),
            d_transpose_column_indices.data(), d_log_x.data(),
            d_output.data(), d_output_cotangent.data(),
            d_input_cotangent.data(),
            {transpose_max_row_length, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_output_tangent.copy_to_host(), unweighted.jvp,
                "unweighted JVP", subwarp);
        compare(d_input_cotangent.copy_to_host(), unweighted.input_vjp,
                "unweighted input VJP", subwarp);

        CHECK_CUDA(springmv::csr_logsumexp_jvp_batched(
            2, num_rows, num_columns, d_row_offsets.data(),
            d_column_indices.data(), d_batched_log_x.data(),
            d_batched_output.data(), d_batched_input_tangent.data(),
            d_batched_output_tangent.data(), {max_row_length, subwarp},
            stream));
        CHECK_CUDA(springmv::csr_logsumexp_input_vjp_batched(
            2, num_rows, num_columns, d_row_offsets.data(),
            d_column_indices.data(), d_transpose_row_offsets.data(),
            d_transpose_column_indices.data(), d_batched_log_x.data(),
            d_batched_output.data(), d_batched_output_cotangent.data(),
            d_batched_input_cotangent.data(),
            {transpose_max_row_length, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_batched_output_tangent.copy_to_host(), batched_jvp,
                "batched unweighted JVP", subwarp);
        compare(d_batched_input_cotangent.copy_to_host(), batched_input_vjp,
                "batched unweighted input VJP", subwarp);

        CHECK_CUDA(springmv::csr_logsumexp_input_vjp_scatter(
            num_rows, num_columns, d_row_offsets.data(),
            d_column_indices.data(), d_log_x.data(), d_output.data(),
            d_output_cotangent.data(), d_input_cotangent.data(),
            {max_row_length, subwarp}, stream));
        CHECK_CUDA(springmv::csr_logsumexp_input_vjp_scatter_batched(
            2, num_rows, num_columns, d_row_offsets.data(),
            d_column_indices.data(), d_batched_log_x.data(),
            d_batched_output.data(), d_batched_output_cotangent.data(),
            d_batched_input_cotangent.data(), {max_row_length, subwarp},
            stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(d_input_cotangent.copy_to_host(), unweighted.input_vjp,
                "unweighted input VJP scatter", subwarp);
        compare(d_batched_input_cotangent.copy_to_host(), batched_input_vjp,
                "batched unweighted input VJP scatter", subwarp);
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void run_finite_case()
{
    run_case(
        {0, 3, 3, 5, 8}, {0, 2, 2, 1, 4, 0, 3, 4},
        {0, 2, 3, 5, 6, 8}, {0, 3, 2, 0, 0, 3, 2, 3},
        {0, 5, 3, 1, 2, 6, 4, 7},
        {0.2F, -0.7F, 0.4F, 1.1F, -0.3F, 0.5F, -1.2F, 0.8F},
        {0.3F, -0.9F, 1.4F, 0.2F, -0.5F},
        {0.1F, -0.2F, 0.3F, 0.5F, -0.7F, 0.9F, 1.1F, -0.4F},
        {-0.5F, 2.0F, 1.0F, -3.0F, 0.25F},
        {0.5F, -2.0F, 3.0F, -1.0F}, 3, 2);
}

void run_infinity_case()
{
    const auto infinity = std::numeric_limits<float>::infinity();
    run_case({0, 3, 5, 5}, {0, 1, 2, 2, 3}, {0, 1, 2, 4, 5},
             {0, 0, 0, 1, 1}, {0, 1, 2, 3, 4},
             {0.0F, 0.0F, 0.0F, 0.0F, 1.0F},
             {infinity, infinity, -infinity, -infinity},
             {0.0F, 1.0F, 2.0F, -3.0F, 4.0F},
             {1.0F, 3.0F, 5.0F, 7.0F}, {2.0F, -3.0F, 7.0F}, 3,
             2);
}

}  // namespace

int main()
{
    try {
        run_finite_case();
        run_infinity_case();
        std::cout << "All log-sum-exp backward tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
