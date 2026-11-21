// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_transpose.hpp"

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
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
    std::vector<float> values;
};

struct TransposeResult {
    std::vector<std::int32_t> row_offsets;
    std::vector<std::int32_t> column_indices;
    std::vector<float> values;
    std::vector<std::int32_t> permutation;
};

TransposeResult cusparse_reference(const MatrixCase& matrix,
                                   DeviceBuffer<std::int32_t>& row_offsets,
                                   DeviceBuffer<std::int32_t>& column_indices,
                                   DeviceBuffer<float>& values)
{
    const auto nnz = static_cast<std::int32_t>(matrix.values.size());
    DeviceBuffer<std::int32_t> result_offsets(matrix.cols + 1);
    DeviceBuffer<std::int32_t> result_columns(nnz);
    DeviceBuffer<float> result_values(nnz);
    cusparseHandle_t handle{};
    CHECK_CUSPARSE(cusparseCreate(&handle));
    std::size_t workspace_size{};
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
        handle, matrix.rows, matrix.cols, nnz, values.data(),
        row_offsets.data(), column_indices.data(), result_values.data(),
        result_offsets.data(), result_columns.data(), CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &workspace_size));
    DeviceBuffer<std::byte> workspace(workspace_size);
    CHECK_CUSPARSE(cusparseCsr2cscEx2(
        handle, matrix.rows, matrix.cols, nnz, values.data(),
        row_offsets.data(), column_indices.data(), result_values.data(),
        result_offsets.data(), result_columns.data(), CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, workspace.data()));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUSPARSE(cusparseDestroy(handle));
    return {result_offsets.copy_to_host(), result_columns.copy_to_host(),
            result_values.copy_to_host(), {}};
}

void require_equal(const std::vector<std::int32_t>& actual,
                   const std::vector<std::int32_t>& expected,
                   const std::string& context)
{
    if (actual != expected) {
        throw std::runtime_error(context + " differs");
    }
}

void run_case(const MatrixCase& matrix)
{
    const auto nnz = static_cast<std::int32_t>(matrix.values.size());
    DeviceBuffer<std::int32_t> row_offsets(matrix.row_offsets.size());
    DeviceBuffer<std::int32_t> column_indices(matrix.column_indices.size());
    DeviceBuffer<float> values(matrix.values.size());
    row_offsets.copy_from(matrix.row_offsets);
    column_indices.copy_from(matrix.column_indices);
    values.copy_from(matrix.values);

    const auto reference =
        cusparse_reference(matrix, row_offsets, column_indices, values);
    std::size_t workspace_size{};
    CHECK_CUDA(springmv::csr_transpose_workspace_size(
        matrix.rows, matrix.cols, nnz, &workspace_size));
    if (workspace_size == 0) {
        throw std::runtime_error("nonempty transpose requested no workspace");
    }
    DeviceBuffer<std::byte> workspace(workspace_size);
    DeviceBuffer<std::int32_t> result_offsets(matrix.cols + 1);
    DeviceBuffer<std::int32_t> result_columns(nnz);
    DeviceBuffer<float> result_values(nnz);
    DeviceBuffer<std::int32_t> permutation(nnz);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(springmv::csr_transpose(
        matrix.rows, matrix.cols, nnz, row_offsets.data(),
        column_indices.data(), values.data(), result_offsets.data(),
        result_columns.data(), result_values.data(), permutation.data(),
        workspace.data(), workspace_size, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const auto actual_offsets = result_offsets.copy_to_host();
    const auto actual_columns = result_columns.copy_to_host();
    const auto actual_values = result_values.copy_to_host();
    const auto actual_permutation = permutation.copy_to_host();
    require_equal(actual_offsets, reference.row_offsets,
                  matrix.name + " row offsets");
    require_equal(actual_columns, reference.column_indices,
                  matrix.name + " column indices");
    if (actual_values != reference.values) {
        throw std::runtime_error(matrix.name + " values differ");
    }

    std::vector<std::int32_t> original_rows(nnz);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        for (auto edge = matrix.row_offsets[row];
             edge < matrix.row_offsets[row + 1]; ++edge) {
            original_rows[edge] = row;
        }
    }
    auto sorted_permutation = actual_permutation;
    std::sort(sorted_permutation.begin(), sorted_permutation.end());
    for (std::int32_t edge = 0; edge < nnz; ++edge) {
        if (sorted_permutation[edge] != edge) {
            throw std::runtime_error(matrix.name +
                                     " permutation is not bijective");
        }
    }
    for (std::int32_t transposed_row = 0; transposed_row < matrix.cols;
         ++transposed_row) {
        std::int32_t previous_row = -1;
        for (auto index = actual_offsets[transposed_row];
             index < actual_offsets[transposed_row + 1]; ++index) {
            const auto edge = actual_permutation[index];
            if (matrix.column_indices[edge] != transposed_row ||
                original_rows[edge] != actual_columns[index] ||
                matrix.values[edge] != actual_values[index]) {
                throw std::runtime_error(matrix.name +
                                         " permutation mapping is invalid");
            }
            if (actual_columns[index] < previous_row) {
                throw std::runtime_error(matrix.name +
                                         " output ordering is not deterministic");
            }
            previous_row = actual_columns[index];
        }
    }

    DeviceBuffer<std::int32_t> structure_offsets(matrix.cols + 1);
    DeviceBuffer<std::int32_t> structure_columns(nnz);
    DeviceBuffer<std::int32_t> structure_permutation(nnz);
    CHECK_CUDA(springmv::csr_transpose(
        matrix.rows, matrix.cols, nnz, row_offsets.data(),
        column_indices.data(), structure_offsets.data(),
        structure_columns.data(), structure_permutation.data(), workspace.data(),
        workspace_size, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    require_equal(structure_offsets.copy_to_host(), actual_offsets,
                  matrix.name + " structure-only offsets");
    require_equal(structure_columns.copy_to_host(), actual_columns,
                  matrix.name + " structure-only columns");
    require_equal(structure_permutation.copy_to_host(), actual_permutation,
                  matrix.name + " structure-only permutation");

    CHECK_CUDA(springmv::csr_transpose(
        matrix.rows, matrix.cols, nnz, row_offsets.data(),
        column_indices.data(), structure_offsets.data(),
        structure_columns.data(), nullptr, workspace.data(), workspace_size,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    require_equal(structure_offsets.copy_to_host(), actual_offsets,
                  matrix.name + " optional-permutation offsets");
    require_equal(structure_columns.copy_to_host(), actual_columns,
                  matrix.name + " optional-permutation columns");

    if (springmv::csr_transpose(
            matrix.rows, matrix.cols, nnz, row_offsets.data(),
            column_indices.data(), structure_offsets.data(),
            structure_columns.data(), nullptr, workspace.data(),
            workspace_size - 1) != cudaErrorInvalidValue) {
        throw std::runtime_error(matrix.name +
                                 " accepted undersized workspace");
    }
    std::cout << "PASS " << matrix.name << '\n';
}

MatrixCase make_rectangular_case()
{
    return {"rectangular_duplicates_and_empty_rows",
            5,
            7,
            {0, 3, 3, 5, 8, 9},
            {4, 1, 4, 6, 0, 1, 1, 5, 0},
            {0.25F, 1.25F, 2.25F, 3.25F, 4.25F, 5.25F, 6.25F, 7.25F,
             8.25F}};
}

MatrixCase make_random_case()
{
    MatrixCase matrix;
    matrix.name = "random_rectangular";
    matrix.rows = 257;
    matrix.cols = 131;
    matrix.row_offsets.push_back(0);
    std::mt19937 generator{2027};
    std::uniform_int_distribution<std::int32_t> column(0, matrix.cols - 1);
    std::uniform_real_distribution<float> value(-2.0F, 2.0F);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const auto length = (row * 19) % 43;
        for (std::int32_t entry = 0; entry < length; ++entry) {
            matrix.column_indices.push_back(column(generator));
            matrix.values.push_back(value(generator));
        }
        matrix.row_offsets.push_back(
            static_cast<std::int32_t>(matrix.values.size()));
    }
    return matrix;
}

void test_empty_matrix()
{
    constexpr std::int32_t rows = 4;
    constexpr std::int32_t cols = 9;
    std::size_t workspace_size = 123;
    CHECK_CUDA(springmv::csr_transpose_workspace_size(rows, cols, 0,
                                                      &workspace_size));
    if (workspace_size != 0) {
        throw std::runtime_error("empty transpose should need no workspace");
    }
    DeviceBuffer<std::int32_t> offsets(cols + 1);
    CHECK_CUDA(cudaMemset(offsets.data(), 0xff,
                          (cols + 1) * sizeof(std::int32_t)));
    CHECK_CUDA(springmv::csr_transpose(rows, cols, 0, nullptr, nullptr,
                                       offsets.data(), nullptr, nullptr,
                                       nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto host_offsets = offsets.copy_to_host();
    if (!std::all_of(host_offsets.begin(), host_offsets.end(),
                     [](std::int32_t value) { return value == 0; })) {
        throw std::runtime_error("empty transpose offsets are not zero");
    }
    std::cout << "PASS empty_matrix\n";
}

void test_argument_validation()
{
    std::size_t workspace_size{};
    if (springmv::csr_transpose_workspace_size(-1, 1, 0,
                                                &workspace_size) !=
        cudaErrorInvalidValue) {
        throw std::runtime_error("negative dimensions should be rejected");
    }
    if (springmv::csr_transpose_workspace_size(1, 1, 0, nullptr) !=
        cudaErrorInvalidValue) {
        throw std::runtime_error("null workspace-size output should be rejected");
    }
    std::cout << "PASS argument_validation\n";
}

}  // namespace

int main()
{
    try {
        test_argument_validation();
        test_empty_matrix();
        run_case(make_rectangular_case());
        run_case(make_random_case());
        std::cout << "All transpose tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

