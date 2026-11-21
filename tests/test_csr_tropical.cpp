// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_tropical.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
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
    explicit DeviceBuffer(std::size_t count) : count_{count}
    {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_),
                              std::max<std::size_t>(count, 1) * sizeof(T)));
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }

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
        std::vector<T> result(count_);
        if (count_ > 0) {
            CHECK_CUDA(cudaMemcpy(result.data(), data_, count_ * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return result;
    }

private:
    T* data_{};
    std::size_t count_{};
};

struct MatrixCase {
    std::string name;
    std::int32_t rows{};
    std::int32_t cols{};
    std::vector<std::int32_t> row_offsets;
    std::vector<std::int32_t> column_indices;
    std::vector<float> edge_values;
    std::vector<float> x;
};

std::int32_t max_row_length(const MatrixCase& matrix)
{
    std::int32_t result{};
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        result = std::max(result, matrix.row_offsets[row + 1] -
                                      matrix.row_offsets[row]);
    }
    return result;
}

float combine(float left, float right, bool minimum)
{
    if (std::isnan(left) || std::isnan(right)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return minimum ? std::min(left, right) : std::max(left, right);
}

std::vector<float> reference(const MatrixCase& matrix, bool weighted,
                             bool minimum)
{
    const auto identity = minimum
                              ? std::numeric_limits<float>::infinity()
                              : -std::numeric_limits<float>::infinity();
    std::vector<float> result(matrix.rows, identity);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        auto state = identity;
        for (auto edge = matrix.row_offsets[row];
             edge < matrix.row_offsets[row + 1]; ++edge) {
            auto term = matrix.x[matrix.column_indices[edge]];
            if (weighted) {
                term = matrix.edge_values[edge] + term;
            }
            state = combine(state, term, minimum);
        }
        result[row] = state;
    }
    return result;
}

std::vector<float> evaluate(const MatrixCase& matrix, bool weighted,
                            bool minimum, std::int32_t subwarp)
{
    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<float> edge_values(matrix.edge_values.size());
    DeviceBuffer<float> x(matrix.x.size());
    DeviceBuffer<float> output(matrix.rows);
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    edge_values.copy_from(matrix.edge_values);
    x.copy_from(matrix.x);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    const springmv::CsrKernelOptions options{max_row_length(matrix), subwarp};
    cudaError_t status{};
    if (minimum && weighted) {
        status = springmv::csr_minplus(
            matrix.rows, row_offsets.data(), column_indices.data(),
            edge_values.data(), x.data(), output.data(), options, stream);
    } else if (minimum) {
        status = springmv::csr_minplus(
            matrix.rows, row_offsets.data(), column_indices.data(), x.data(),
            output.data(), options, stream);
    } else if (weighted) {
        status = springmv::csr_maxplus(
            matrix.rows, row_offsets.data(), column_indices.data(),
            edge_values.data(), x.data(), output.data(), options, stream);
    } else {
        status = springmv::csr_maxplus(
            matrix.rows, row_offsets.data(), column_indices.data(), x.data(),
            output.data(), options, stream);
    }
    CHECK_CUDA(status);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return output.copy_to_host();
}

void compare_value(const std::string& context, std::size_t row, float actual,
                   float expected)
{
    if (std::isnan(expected)) {
        if (!std::isnan(actual)) {
            throw std::runtime_error(context + " row " +
                                     std::to_string(row) +
                                     " expected NaN");
        }
        return;
    }
    if (std::isinf(expected)) {
        if (!std::isinf(actual) ||
            std::signbit(actual) != std::signbit(expected)) {
            throw std::runtime_error(context + " row " +
                                     std::to_string(row) +
                                     " has wrong infinity");
        }
        return;
    }
    if (actual != expected) {
        throw std::runtime_error(context + " row " + std::to_string(row) +
                                 ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

void run_case(const MatrixCase& matrix)
{
    for (const bool minimum : {true, false}) {
        for (const bool weighted : {false, true}) {
            const auto expected = reference(matrix, weighted, minimum);
            for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
                const auto actual =
                    evaluate(matrix, weighted, minimum, subwarp);
                const auto context =
                    matrix.name + (minimum ? " minplus" : " maxplus") +
                    (weighted ? " weighted" : " unweighted") +
                    " subwarp " + std::to_string(subwarp);
                for (std::size_t row = 0; row < actual.size(); ++row) {
                    compare_value(context, row, actual[row], expected[row]);
                }
            }
        }
    }
    std::cout << "PASS " << matrix.name << '\n';
}

MatrixCase make_analytic_case()
{
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    return {"analytic_edge_cases",
            8,
            7,
            {0, 0, 1, 4, 6, 8, 10, 12, 14},
            {0, 0, 1, 1, 2, 0, 3, 0, 4, 0, 3, 0, 6, 6},
            {1.0F, 4.0F, 10.0F, -10.0F, 0.0F, 1.0F, 0.0F, 1.0F,
             0.0F, 1.0F, inf, 0.0F, 1.0F, -2.0F},
            {2.0F, -3.0F, inf, -inf, nan, 0.5F, -1.0F}};
}

MatrixCase make_irregular_case()
{
    MatrixCase matrix;
    matrix.name = "irregular_rows";
    matrix.rows = 12;
    matrix.cols = 521;
    const std::vector<std::int32_t> lengths{0,  1,  2,  3,  4,  7,
                                             8, 16, 31, 32, 33, 257};
    matrix.row_offsets.push_back(0);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        for (std::int32_t entry = 0; entry < lengths[row]; ++entry) {
            matrix.column_indices.push_back(
                (row * 29 + entry * 17) % matrix.cols);
            matrix.edge_values.push_back(
                static_cast<float>((entry % 23) - 11) * 0.375F);
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.edge_values.size()));
    }
    matrix.x.resize(matrix.cols);
    for (std::int32_t column = 0; column < matrix.cols; ++column) {
        matrix.x[column] = static_cast<float>((column % 19) - 9) * 0.5F;
    }
    return matrix;
}

MatrixCase make_random_case()
{
    MatrixCase matrix;
    matrix.name = "deterministic_random";
    matrix.rows = 257;
    matrix.cols = 509;
    matrix.row_offsets.push_back(0);
    std::mt19937 generator{811};
    std::uniform_int_distribution<std::int32_t> column(0, matrix.cols - 1);
    std::uniform_real_distribution<float> value(-100.0F, 100.0F);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const auto length = (row * 37) % 65;
        for (std::int32_t entry = 0; entry < length; ++entry) {
            matrix.column_indices.push_back(column(generator));
            matrix.edge_values.push_back(value(generator));
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.edge_values.size()));
    }
    matrix.x.resize(matrix.cols);
    std::generate(matrix.x.begin(), matrix.x.end(),
                  [&] { return value(generator); });
    return matrix;
}

void test_properties()
{
    const auto matrix = make_irregular_case();
    for (const bool weighted : {false, true}) {
        const auto minimum = evaluate(matrix, weighted, true, 0);

        auto negated = matrix;
        for (auto& value : negated.x) {
            value = -value;
        }
        for (auto& value : negated.edge_values) {
            value = -value;
        }
        const auto negative_maximum = evaluate(negated, weighted, false, 0);
        for (std::size_t row = 0; row < minimum.size(); ++row) {
            compare_value("min/max duality", row, minimum[row],
                          -negative_maximum[row]);
        }

        auto permuted = matrix;
        for (std::int32_t row = 0; row < permuted.rows; ++row) {
            const auto begin = permuted.row_offsets[row];
            const auto end = permuted.row_offsets[row + 1];
            std::reverse(permuted.column_indices.begin() + begin,
                         permuted.column_indices.begin() + end);
            std::reverse(permuted.edge_values.begin() + begin,
                         permuted.edge_values.begin() + end);
        }
        const auto permuted_minimum = evaluate(permuted, weighted, true, 0);
        for (std::size_t row = 0; row < minimum.size(); ++row) {
            compare_value("row permutation", row, permuted_minimum[row],
                          minimum[row]);
        }
    }

    auto zero_weighted = matrix;
    std::fill(zero_weighted.edge_values.begin(),
              zero_weighted.edge_values.end(), 0.0F);
    for (const bool minimum : {true, false}) {
        const auto weighted = evaluate(zero_weighted, true, minimum, 0);
        const auto unweighted = evaluate(zero_weighted, false, minimum, 0);
        for (std::size_t row = 0; row < weighted.size(); ++row) {
            compare_value("zero-weight equivalence", row, weighted[row],
                          unweighted[row]);
        }
    }
    std::cout << "PASS tropical_properties\n";
}

void test_argument_validation()
{
    if (springmv::csr_minplus(0, nullptr, nullptr, nullptr, nullptr, {}) !=
            cudaSuccess ||
        springmv::csr_maxplus(0, nullptr, nullptr, nullptr, nullptr, {}) !=
            cudaSuccess) {
        throw std::runtime_error("zero-row unweighted operation should succeed");
    }
    if (springmv::csr_minplus(0, nullptr, nullptr, nullptr, nullptr, nullptr,
                              {}) != cudaSuccess ||
        springmv::csr_maxplus(0, nullptr, nullptr, nullptr, nullptr, nullptr,
                              {}) != cudaSuccess) {
        throw std::runtime_error("zero-row weighted operation should succeed");
    }
    if (springmv::csr_minplus(-1, nullptr, nullptr, nullptr, nullptr, {}) !=
        cudaErrorInvalidValue) {
        throw std::runtime_error("negative row count should be rejected");
    }
    if (springmv::csr_maxplus(1, nullptr, nullptr, nullptr, nullptr, {1, 3}) !=
        cudaErrorInvalidValue) {
        throw std::runtime_error("invalid subwarp should be rejected");
    }
    std::cout << "PASS argument_validation\n";
}

}  // namespace

int main()
{
    try {
        test_argument_validation();
        run_case(make_analytic_case());
        run_case(make_irregular_case());
        run_case(make_random_case());
        test_properties();
        std::cout << "All tropical tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

