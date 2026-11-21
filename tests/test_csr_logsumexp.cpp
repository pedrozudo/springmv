// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_logsumexp.hpp"

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expression)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_cusparse(cusparseStatus_t status, const char* expression)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expression) + ": cuSPARSE error " +
                                 std::to_string(static_cast<int>(status)));
    }
}

#define CHECK_CUDA(expression) check_cuda((expression), #expression)
#define CHECK_CUSPARSE(expression) check_cusparse((expression), #expression)

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
    std::vector<float> log_values;
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

std::vector<long double> stable_host_reference(const MatrixCase& matrix)
{
    const auto negative_infinity =
        -std::numeric_limits<long double>::infinity();
    const auto positive_infinity =
        std::numeric_limits<long double>::infinity();
    const auto quiet_nan = std::numeric_limits<long double>::quiet_NaN();
    std::vector<long double> result(matrix.rows, negative_infinity);

    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        auto maximum = negative_infinity;
        bool has_nan = false;
        for (auto nz = matrix.row_offsets[row];
             nz < matrix.row_offsets[row + 1]; ++nz) {
            // Match the production kernel's float32 addition before using a
            // wider accumulator for the reference reduction.
            const auto term_float =
                matrix.log_values[nz] + matrix.x[matrix.column_indices[nz]];
            const auto term = static_cast<long double>(term_float);
            if (std::isnan(term)) {
                has_nan = true;
                break;
            }
            maximum = std::max(maximum, term);
        }
        if (has_nan) {
            result[row] = quiet_nan;
            continue;
        }
        if (maximum == positive_infinity || maximum == negative_infinity) {
            result[row] = maximum;
            continue;
        }

        long double scaled_sum{};
        for (auto nz = matrix.row_offsets[row];
             nz < matrix.row_offsets[row + 1]; ++nz) {
            const auto term = static_cast<long double>(
                matrix.log_values[nz] + matrix.x[matrix.column_indices[nz]]);
            scaled_sum += std::exp(term - maximum);
        }
        result[row] = maximum + std::log(scaled_sum);
    }
    return result;
}

void compare_value(std::string_view context, std::size_t row, float actual,
                   long double expected)
{
    if (std::isnan(expected)) {
        if (!std::isnan(actual)) {
            throw std::runtime_error(std::string(context) + " row " +
                                     std::to_string(row) +
                                     " expected NaN");
        }
        return;
    }
    if (std::isinf(expected)) {
        if (!std::isinf(actual) || std::signbit(actual) != std::signbit(expected)) {
            throw std::runtime_error(std::string(context) + " row " +
                                     std::to_string(row) +
                                     " has wrong infinity");
        }
        return;
    }

    const auto error =
        std::abs(static_cast<long double>(actual) - expected);
    const auto tolerance =
        8.0e-5L + 6.0e-7L * std::abs(expected);
    if (!std::isfinite(actual) || error > tolerance) {
        std::ostringstream message;
        message << context << " row " << row << ": expected "
                << static_cast<double>(expected) << ", got " << actual
                << ", error " << static_cast<double>(error)
                << ", tolerance " << static_cast<double>(tolerance);
        throw std::runtime_error(message.str());
    }
}

std::vector<double> cusparse_transformed_reference(const MatrixCase& matrix)
{
    std::vector<double> exp_values(matrix.log_values.size());
    std::vector<double> exp_x(matrix.x.size());
    std::transform(matrix.log_values.begin(), matrix.log_values.end(),
                   exp_values.begin(),
                   [](float value) { return std::exp(double{value}); });
    std::transform(matrix.x.begin(), matrix.x.end(), exp_x.begin(),
                   [](float value) { return std::exp(double{value}); });

    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<double> values(exp_values.size());
    DeviceBuffer<double> x(exp_x.size());
    DeviceBuffer<double> output(matrix.rows);
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    values.copy_from(exp_values);
    x.copy_from(exp_x);
    CHECK_CUDA(cudaMemset(output.data(), 0,
                          static_cast<std::size_t>(matrix.rows) *
                              sizeof(double)));

    cusparseHandle_t handle{};
    cusparseSpMatDescr_t sparse{};
    cusparseDnVecDescr_t input{};
    cusparseDnVecDescr_t result{};
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &sparse, matrix.rows, matrix.cols,
        static_cast<std::int64_t>(matrix.log_values.size()),
        row_offsets.data(), column_indices.data(), values.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&input, matrix.cols, x.data(), CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&result, matrix.rows, output.data(),
                                       CUDA_R_64F));
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    std::size_t workspace_size{};
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, input, &beta,
        result, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size));
    DeviceBuffer<std::byte> workspace(workspace_size);
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, input, &beta,
        result, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, workspace.data()));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto transformed = output.copy_to_host();
    for (auto& value : transformed) {
        value = std::log(value);
    }
    CHECK_CUSPARSE(cusparseDestroyDnVec(result));
    CHECK_CUSPARSE(cusparseDestroyDnVec(input));
    CHECK_CUSPARSE(cusparseDestroySpMat(sparse));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    return transformed;
}

std::vector<float> evaluate(const MatrixCase& matrix, std::int32_t subwarp)
{
    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<float> log_values(matrix.log_values.size());
    DeviceBuffer<float> x(matrix.x.size());
    DeviceBuffer<float> output(matrix.rows);
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    log_values.copy_from(matrix.log_values);
    x.copy_from(matrix.x);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(springmv::csr_logsumexp(
        matrix.rows, row_offsets.data(), column_indices.data(),
        log_values.data(), x.data(), output.data(),
        {max_row_length(matrix), subwarp}, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return output.copy_to_host();
}

std::vector<float> evaluate_unweighted(const MatrixCase& matrix,
                                       std::int32_t subwarp)
{
    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<float> x(matrix.x.size());
    DeviceBuffer<float> output(matrix.rows);
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    x.copy_from(matrix.x);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(springmv::csr_logsumexp(
        matrix.rows, row_offsets.data(), column_indices.data(), x.data(),
        output.data(), {max_row_length(matrix), subwarp}, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return output.copy_to_host();
}

void run_case(const MatrixCase& matrix, bool use_cusparse)
{
    const auto expected = stable_host_reference(matrix);
    if (use_cusparse) {
        const auto transformed = cusparse_transformed_reference(matrix);
        for (std::size_t row = 0; row < transformed.size(); ++row) {
            compare_value(matrix.name + " cuSPARSE", row,
                          static_cast<float>(transformed[row]), expected[row]);
        }
    }

    for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
        const auto actual = evaluate(matrix, subwarp);
        for (std::size_t row = 0; row < actual.size(); ++row) {
            compare_value(matrix.name + " subwarp " +
                              std::to_string(subwarp),
                          row, actual[row], expected[row]);
        }
    }
    std::cout << "PASS " << matrix.name << '\n';
}

MatrixCase make_analytic_case()
{
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    return {"analytic_edge_cases",
            9,
            8,
            {0, 0, 1, 4, 6, 8, 10, 12, 15, 16},
            {1, 2, 2, 2, 3, 3, 0, 4, 0, 3, 6, 5, 1, 1, 1, 7},
            {1.25F, 0.0F, 0.0F, 0.0F, 0.0F, -inf, 0.25F, 0.0F,
             -2.0F, 0.0F, 0.0F, 0.0F, 0.5F, 1.5F, 1.5F, 0.0F},
            {0.0F, -0.5F, 2.0F, -inf, inf, -10000.0F, 10000.0F,
             nan}};
}

MatrixCase make_irregular_case(bool extreme)
{
    MatrixCase matrix;
    matrix.name = extreme ? "extreme_irregular_rows" : "moderate_irregular_rows";
    matrix.rows = 12;
    matrix.cols = 521;
    const std::vector<std::int32_t> lengths{0,  1,  2,  3,  4,  7,
                                             8, 16, 31, 32, 33, 257};
    matrix.row_offsets.push_back(0);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        for (std::int32_t entry = 0; entry < lengths[row]; ++entry) {
            matrix.column_indices.push_back(
                (row * 29 + entry * 17) % matrix.cols);
            if (extreme) {
                const auto sign = entry % 3 == 0 ? 1.0F : -1.0F;
                matrix.log_values.push_back(
                    sign * (9000.0F + static_cast<float>(entry % 997)));
            } else {
                matrix.log_values.push_back(
                    static_cast<float>((entry % 21) - 10) * 0.75F);
            }
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.log_values.size()));
    }
    matrix.x.resize(matrix.cols);
    for (std::int32_t col = 0; col < matrix.cols; ++col) {
        matrix.x[col] = static_cast<float>((col % 19) - 9) * 0.5F;
    }
    return matrix;
}

MatrixCase make_random_case()
{
    MatrixCase matrix;
    matrix.name = "moderate_random_cusparse";
    matrix.rows = 257;
    matrix.cols = 509;
    matrix.row_offsets.push_back(0);
    std::mt19937 generator{1729};
    std::uniform_int_distribution<std::int32_t> column(0, matrix.cols - 1);
    std::uniform_real_distribution<float> value(-10.0F, 10.0F);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const auto length = (row * 37) % 65;
        for (std::int32_t entry = 0; entry < length; ++entry) {
            matrix.column_indices.push_back(column(generator));
            matrix.log_values.push_back(value(generator));
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.log_values.size()));
    }
    matrix.x.resize(matrix.cols);
    std::generate(matrix.x.begin(), matrix.x.end(),
                  [&] { return value(generator); });
    return matrix;
}

MatrixCase make_metamorphic_case()
{
    MatrixCase matrix;
    matrix.name = "metamorphic_base";
    matrix.rows = 33;
    matrix.cols = 47;
    matrix.row_offsets.push_back(0);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const auto length = 1 + (row * 7) % 40;
        for (std::int32_t entry = 0; entry < length; ++entry) {
            matrix.column_indices.push_back((row + entry * 11) % matrix.cols);
            matrix.log_values.push_back(
                static_cast<float>((row + entry) % 17 - 8) * 0.25F);
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.log_values.size()));
    }
    matrix.x.resize(matrix.cols);
    for (std::int32_t col = 0; col < matrix.cols; ++col) {
        matrix.x[col] = static_cast<float>(col % 13 - 6) * 0.5F;
    }
    return matrix;
}

void test_metamorphic_properties()
{
    const auto base = make_metamorphic_case();
    const auto base_output = evaluate(base, 0);

    auto shifted = base;
    constexpr float shift = 17.25F;
    for (auto& value : shifted.x) {
        value += shift;
    }
    const auto shifted_output = evaluate(shifted, 0);
    for (std::size_t row = 0; row < base_output.size(); ++row) {
        compare_value("shift equivariance", row, shifted_output[row],
                      static_cast<long double>(base_output[row]) + shift);
    }

    auto permuted = base;
    for (std::int32_t row = 0; row < permuted.rows; ++row) {
        const auto begin = permuted.row_offsets[row];
        const auto end = permuted.row_offsets[row + 1];
        std::reverse(permuted.column_indices.begin() + begin,
                     permuted.column_indices.begin() + end);
        std::reverse(permuted.log_values.begin() + begin,
                     permuted.log_values.begin() + end);
    }
    const auto permuted_output = evaluate(permuted, 0);
    for (std::size_t row = 0; row < base_output.size(); ++row) {
        compare_value("row permutation", row, permuted_output[row],
                      base_output[row]);
    }

    auto duplicated = base;
    duplicated.row_offsets.clear();
    duplicated.column_indices.clear();
    duplicated.log_values.clear();
    duplicated.row_offsets.push_back(0);
    for (std::int32_t row = 0; row < base.rows; ++row) {
        for (auto nz = base.row_offsets[row]; nz < base.row_offsets[row + 1];
             ++nz) {
            duplicated.column_indices.push_back(base.column_indices[nz]);
            duplicated.log_values.push_back(base.log_values[nz]);
            duplicated.column_indices.push_back(base.column_indices[nz]);
            duplicated.log_values.push_back(base.log_values[nz]);
        }
        duplicated.row_offsets.push_back(
            static_cast<std::int32_t>(duplicated.log_values.size()));
    }
    const auto duplicated_output = evaluate(duplicated, 0);
    const auto log_two = std::log(2.0L);
    for (std::size_t row = 0; row < base_output.size(); ++row) {
        compare_value("term duplication", row, duplicated_output[row],
                      static_cast<long double>(base_output[row]) + log_two);
    }

    auto increased = base;
    increased.x[0] += 0.5F;
    const auto increased_output = evaluate(increased, 0);
    for (std::size_t row = 0; row < base_output.size(); ++row) {
        if (increased_output[row] + 1.0e-6F < base_output[row]) {
            throw std::runtime_error("monotonicity failed at row " +
                                     std::to_string(row));
        }
    }
    std::cout << "PASS metamorphic_properties\n";
}

void test_unweighted_overload()
{
    for (auto matrix : {make_analytic_case(), make_metamorphic_case()}) {
        std::fill(matrix.log_values.begin(), matrix.log_values.end(), 0.0F);
        const auto expected = stable_host_reference(matrix);
        for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
            const auto unweighted = evaluate_unweighted(matrix, subwarp);
            const auto zero_weighted = evaluate(matrix, subwarp);
            for (std::size_t row = 0; row < unweighted.size(); ++row) {
                const auto context = matrix.name + " unweighted subwarp " +
                                     std::to_string(subwarp);
                compare_value(context, row, unweighted[row], expected[row]);
                compare_value(context + " zero-weight equivalence", row,
                              unweighted[row], zero_weighted[row]);
            }
        }
    }
    std::cout << "PASS unweighted_overload\n";
}

void test_argument_validation()
{
    if (springmv::csr_logsumexp(0, nullptr, nullptr, nullptr, nullptr,
                                nullptr, {}) != cudaSuccess) {
        throw std::runtime_error("zero-row operation should succeed");
    }
    if (springmv::csr_logsumexp(-1, nullptr, nullptr, nullptr, nullptr,
                                nullptr, {}) != cudaErrorInvalidValue) {
        throw std::runtime_error("negative row count should be rejected");
    }
    if (springmv::csr_logsumexp(1, nullptr, nullptr, nullptr, nullptr,
                                nullptr, {1, 3}) != cudaErrorInvalidValue) {
        throw std::runtime_error("invalid subwarp should be rejected");
    }
    if (springmv::csr_logsumexp(0, nullptr, nullptr, nullptr, nullptr, {}) !=
        cudaSuccess) {
        throw std::runtime_error("zero-row unweighted operation should succeed");
    }
    if (springmv::csr_logsumexp(-1, nullptr, nullptr, nullptr, nullptr, {}) !=
        cudaErrorInvalidValue) {
        throw std::runtime_error(
            "negative unweighted row count should be rejected");
    }
    std::cout << "PASS argument_validation\n";
}

}  // namespace

int main()
{
    try {
        test_argument_validation();
        run_case(make_analytic_case(), false);
        run_case(make_irregular_case(false), true);
        run_case(make_irregular_case(true), false);
        run_case(make_random_case(), true);
        test_metamorphic_properties();
        test_unweighted_overload();
        std::cout << "All log-sum-exp tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

