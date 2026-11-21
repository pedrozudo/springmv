// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_spmv.hpp"

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
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
        // A non-null allocation keeps zero-nnz CSR descriptors portable across
        // cuSPARSE versions.
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
    std::vector<float> values;
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

std::vector<float> cusparse_reference(
    const MatrixCase& matrix, DeviceBuffer<std::int32_t>& row_offsets,
    DeviceBuffer<std::int32_t>& column_indices, DeviceBuffer<float>& values,
    DeviceBuffer<float>& x)
{
    DeviceBuffer<float> output(matrix.rows);
    CHECK_CUDA(cudaMemset(output.data(), 0,
                          static_cast<std::size_t>(matrix.rows) *
                              sizeof(float)));

    cusparseHandle_t handle{};
    cusparseSpMatDescr_t sparse{};
    cusparseDnVecDescr_t input{};
    cusparseDnVecDescr_t result{};
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &sparse, matrix.rows, matrix.cols,
        static_cast<std::int64_t>(matrix.values.size()), row_offsets.data(),
        column_indices.data(), values.data(), CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&input, matrix.cols, x.data(), CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&result, matrix.rows, output.data(),
                                       CUDA_R_32F));

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    std::size_t workspace_size{};
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, input, &beta,
        result, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size));
    DeviceBuffer<std::byte> workspace(workspace_size);
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, input, &beta,
        result, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, workspace.data()));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto host_output = output.copy_to_host();
    CHECK_CUSPARSE(cusparseDestroyDnVec(result));
    CHECK_CUSPARSE(cusparseDestroyDnVec(input));
    CHECK_CUSPARSE(cusparseDestroySpMat(sparse));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    return host_output;
}

void compare(const MatrixCase& matrix, const std::vector<float>& actual,
             const std::vector<float>& expected, std::int32_t subwarp)
{
    for (std::size_t row = 0; row < expected.size(); ++row) {
        const auto tolerance =
            2.0e-5F + 2.0e-5F * std::abs(expected[row]);
        if (!std::isfinite(actual[row]) ||
            std::abs(actual[row] - expected[row]) > tolerance) {
            std::ostringstream message;
            message << matrix.name << " failed for subwarp " << subwarp
                    << " at row " << row << ": expected " << expected[row]
                    << ", got " << actual[row]
                    << ", tolerance " << tolerance;
            throw std::runtime_error(message.str());
        }
    }
}

void run_case(const MatrixCase& matrix)
{
    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<float> values(matrix.values.size());
    DeviceBuffer<float> x(matrix.x.size());
    DeviceBuffer<float> output(matrix.rows);
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    values.copy_from(matrix.values);
    x.copy_from(matrix.x);

    const auto expected = cusparse_reference(matrix, row_offsets,
                                              column_indices, values, x);
    const auto maximum = max_row_length(matrix);
    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
        CHECK_CUDA(cudaMemset(output.data(), 0xff,
                              static_cast<std::size_t>(matrix.rows) *
                                  sizeof(float)));
        CHECK_CUDA(springmv::csr_spmv(
            matrix.rows, row_offsets.data(), column_indices.data(),
            values.data(), x.data(), output.data(), {maximum, subwarp},
            stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(matrix, output.copy_to_host(), expected, subwarp);
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::cout << "PASS " << matrix.name << '\n';
}

void run_unweighted_case(MatrixCase matrix)
{
    matrix.name += "_unweighted";
    std::fill(matrix.values.begin(), matrix.values.end(), 1.0F);
    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<float> values(matrix.values.size());
    DeviceBuffer<float> x(matrix.x.size());
    DeviceBuffer<float> output(matrix.rows);
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    values.copy_from(matrix.values);
    x.copy_from(matrix.x);

    const auto expected = cusparse_reference(matrix, row_offsets,
                                              column_indices, values, x);
    const auto maximum = max_row_length(matrix);
    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    for (const std::int32_t subwarp : {0, 1, 2, 4, 8, 16, 32}) {
        CHECK_CUDA(cudaMemset(output.data(), 0xff,
                              static_cast<std::size_t>(matrix.rows) *
                                  sizeof(float)));
        CHECK_CUDA(springmv::csr_spmv(
            matrix.rows, row_offsets.data(), column_indices.data(), x.data(),
            output.data(), {maximum, subwarp}, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        compare(matrix, output.copy_to_host(), expected, subwarp);
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::cout << "PASS " << matrix.name << '\n';
}

MatrixCase make_small_case()
{
    return {"empty_rows_and_duplicates",
            7,
            6,
            {0, 3, 3, 5, 5, 6, 9, 9},
            {0, 2, 2, 1, 4, 5, 0, 3, 5},
            {1.0F, 2.0F, -0.5F, 4.0F, -2.0F, 3.0F, -1.0F, 0.25F,
             2.0F},
            {1.0F, -2.0F, 0.5F, 4.0F, -1.0F, 3.0F}};
}

MatrixCase make_irregular_case()
{
    MatrixCase matrix;
    matrix.name = "highly_irregular_rows";
    matrix.rows = 12;
    matrix.cols = 521;
    const std::vector<std::int32_t> lengths{0,  1,  2,  3,  7, 16,
                                             31, 32, 33, 65, 127, 257};
    matrix.row_offsets.push_back(0);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        for (std::int32_t entry = 0; entry < lengths[row]; ++entry) {
            matrix.column_indices.push_back(
                (row * 29 + entry * 17) % matrix.cols);
            matrix.values.push_back(
                static_cast<float>((entry % 11) - 5) * 0.125F);
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.values.size()));
    }
    matrix.x.resize(matrix.cols);
    for (std::int32_t col = 0; col < matrix.cols; ++col) {
        matrix.x[col] = static_cast<float>((col % 19) - 9) * 0.0625F;
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
    std::mt19937 generator{1729};
    std::uniform_int_distribution<std::int32_t> column(0, matrix.cols - 1);
    std::uniform_real_distribution<float> value(-1.0F, 1.0F);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const auto length = (row * 37) % 65;
        for (std::int32_t entry = 0; entry < length; ++entry) {
            matrix.column_indices.push_back(column(generator));
            matrix.values.push_back(value(generator));
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.values.size()));
    }
    matrix.x.resize(matrix.cols);
    std::generate(matrix.x.begin(), matrix.x.end(),
                  [&] { return value(generator); });
    return matrix;
}

MatrixCase make_zero_nnz_case()
{
    MatrixCase matrix;
    matrix.name = "all_empty_rows";
    matrix.rows = 9;
    matrix.cols = 5;
    matrix.row_offsets.assign(matrix.rows + 1, 0);
    matrix.x.assign(matrix.cols, 1.0F);
    return matrix;
}

void test_selection()
{
    const std::vector<std::pair<std::int32_t, std::int32_t>> cases{
        {0, 1},  {1, 1},  {2, 2},  {3, 2},   {4, 4},   {7, 4},
        {8, 8},  {15, 8}, {16, 16}, {31, 16}, {32, 32}, {1000, 32}};
    for (const auto& [length, expected] : cases) {
        const auto actual = springmv::select_subwarp_size(length);
        if (actual != expected) {
            throw std::runtime_error("unexpected automatic subwarp selection");
        }
    }
    std::cout << "PASS subwarp_selection\n";
}

void test_argument_validation()
{
    if (springmv::csr_spmv(0, nullptr, nullptr, nullptr, nullptr, nullptr,
                           {}) != cudaSuccess) {
        throw std::runtime_error("zero-row operation should succeed");
    }
    if (springmv::csr_spmv(-1, nullptr, nullptr, nullptr, nullptr, nullptr,
                           {}) != cudaErrorInvalidValue) {
        throw std::runtime_error("negative row count should be rejected");
    }
    if (springmv::csr_spmv(1, nullptr, nullptr, nullptr, nullptr, nullptr,
                           {1, 3}) != cudaErrorInvalidValue) {
        throw std::runtime_error("invalid subwarp should be rejected");
    }
    if (springmv::csr_spmv(0, nullptr, nullptr, nullptr, nullptr, {}) !=
        cudaSuccess) {
        throw std::runtime_error("zero-row unweighted operation should succeed");
    }
    if (springmv::csr_spmv(-1, nullptr, nullptr, nullptr, nullptr, {}) !=
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
        test_selection();
        test_argument_validation();
        run_case(make_small_case());
        run_case(make_irregular_case());
        run_case(make_random_case());
        run_case(make_zero_nnz_case());
        run_unweighted_case(make_small_case());
        run_unweighted_case(make_irregular_case());
        run_unweighted_case(make_random_case());
        run_unweighted_case(make_zero_nnz_case());
        std::cout << "All SpringMV tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

