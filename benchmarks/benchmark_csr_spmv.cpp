// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_spmv.hpp"
#include "springmv/csr_logsumexp.hpp"
#include "springmv/csr_tropical.hpp"

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
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

    void copy_from(const std::vector<T>& source)
    {
        CHECK_CUDA(cudaMemcpy(data_, source.data(), source.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    std::vector<T> copy_to_host() const
    {
        std::vector<T> result(count_);
        CHECK_CUDA(cudaMemcpy(result.data(), data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost));
        return result;
    }

private:
    T* data_{};
    std::size_t count_{};
};

struct Options {
    std::int32_t rows = 1 << 18;
    std::int32_t cols = 1 << 18;
    std::int32_t nnz_per_row = 32;
    int iterations = 30;
    int warmup = 5;
    std::int32_t subwarp = 0;
};

int parse_integer(const char* text, std::string_view option)
{
    std::size_t consumed{};
    const auto value = std::stoll(text, &consumed);
    if (consumed != std::string{text}.size() || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string("missing value for ") +
                                        argv[index]);
        }
        const std::string_view name{argv[index]};
        const auto value = parse_integer(argv[index + 1], name);
        if (name == "--rows") {
            options.rows = value;
        } else if (name == "--cols") {
            options.cols = value;
        } else if (name == "--nnz-per-row") {
            options.nnz_per_row = value;
        } else if (name == "--iterations") {
            options.iterations = value;
        } else if (name == "--warmup") {
            options.warmup = value;
        } else if (name == "--subwarp") {
            options.subwarp = value;
        } else {
            throw std::invalid_argument("unknown option " + std::string(name));
        }
    }
    if (options.rows <= 0 || options.cols <= 0 || options.nnz_per_row <= 0 ||
        options.iterations <= 0 || options.nnz_per_row > options.cols) {
        throw std::invalid_argument(
            "rows, cols, nnz-per-row, and iterations must be positive; "
            "nnz-per-row must not exceed cols");
    }
    return options;
}

struct Measurements {
    float minimum{};
    float median{};
    float maximum{};
};

template <typename Operation>
Measurements measure(Operation&& operation, int warmup, int iterations)
{
    for (int iteration = 0; iteration < warmup; ++iteration) {
        operation();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<float> times;
    times.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CHECK_CUDA(cudaEventRecord(start));
        operation();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed{};
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        times.push_back(elapsed);
    }
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));

    std::sort(times.begin(), times.end());
    const auto middle = times.size() / 2;
    const auto median = times.size() % 2 == 0
                            ? 0.5F * (times[middle - 1] + times[middle])
                            : times[middle];
    return {times.front(), median, times.back()};
}

void print_result(std::string_view name, const Measurements& timing,
                  std::int64_t rows, std::int64_t nnz,
                  double flops_per_nonzero,
                  bool has_edge_values)
{
    const auto bytes =
        static_cast<double>(nnz) *
            (sizeof(std::int32_t) + sizeof(float) +
             (has_edge_values ? sizeof(float) : 0)) +
        static_cast<double>(rows + 1) * sizeof(std::int32_t) +
        static_cast<double>(rows) * sizeof(float);
    const auto billion_nnz_per_second =
        static_cast<double>(nnz) /
        (static_cast<double>(timing.median) * 1.0e6);
    const auto bandwidth =
        bytes / (static_cast<double>(timing.median) * 1.0e6);
    std::cout << std::left << std::setw(18) << name << std::right << std::fixed
              << std::setprecision(4) << " median " << std::setw(9)
              << timing.median << " ms  min " << std::setw(9) << timing.minimum
              << " ms  max " << std::setw(9) << timing.maximum << " ms  "
              << std::setprecision(2) << std::setw(8)
              << billion_nnz_per_second << " GNNZ/s  " << std::setw(9)
              << bandwidth << " GB/s";
    if (flops_per_nonzero > 0.0) {
        std::cout << "  " << std::setw(9)
                  << flops_per_nonzero * billion_nnz_per_second << " GFLOP/s";
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse_options(argc, argv);
        const auto nnz = static_cast<std::int64_t>(options.rows) *
                         options.nnz_per_row;
        if (nnz > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("nnz exceeds int32 CSR capacity");
        }

        std::vector<std::int32_t> row_offsets(options.rows + 1);
        std::vector<std::int32_t> column_indices(nnz);
        std::vector<float> values(nnz);
        std::vector<float> x(options.cols);
        for (std::int32_t row = 0; row <= options.rows; ++row) {
            row_offsets[row] = row * options.nnz_per_row;
        }
        for (std::int32_t row = 0; row < options.rows; ++row) {
            for (std::int32_t entry = 0; entry < options.nnz_per_row; ++entry) {
                const auto offset = row * options.nnz_per_row + entry;
                column_indices[offset] =
                    (static_cast<std::int64_t>(row) * 17 +
                     static_cast<std::int64_t>(entry) * 9973) %
                    options.cols;
                values[offset] =
                    static_cast<float>((offset % 17) - 8) * 0.03125F;
            }
        }
        for (std::int32_t col = 0; col < options.cols; ++col) {
            x[col] = static_cast<float>((col % 23) - 11) * 0.0625F;
        }

        DeviceBuffer<std::int32_t> d_row_offsets(row_offsets.size());
        DeviceBuffer<std::int32_t> d_column_indices(column_indices.size());
        DeviceBuffer<float> d_values(values.size());
        DeviceBuffer<float> d_x(x.size());
        DeviceBuffer<float> d_springmv(options.rows);
        DeviceBuffer<float> d_springmv_unweighted(options.rows);
        DeviceBuffer<float> d_logsumexp(options.rows);
        DeviceBuffer<float> d_logsumexp_unweighted(options.rows);
        DeviceBuffer<float> d_minplus(options.rows);
        DeviceBuffer<float> d_minplus_unweighted(options.rows);
        DeviceBuffer<float> d_maxplus(options.rows);
        DeviceBuffer<float> d_maxplus_unweighted(options.rows);
        DeviceBuffer<float> d_cusparse(options.rows);
        d_row_offsets.copy_from(row_offsets);
        d_column_indices.copy_from(column_indices);
        d_values.copy_from(values);
        d_x.copy_from(x);

        cusparseHandle_t handle{};
        cusparseSpMatDescr_t sparse{};
        cusparseDnVecDescr_t input{};
        cusparseDnVecDescr_t output{};
        CHECK_CUSPARSE(cusparseCreate(&handle));
        CHECK_CUSPARSE(cusparseCreateCsr(
            &sparse, options.rows, options.cols, nnz, d_row_offsets.data(),
            d_column_indices.data(), d_values.data(), CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&input, options.cols, d_x.data(),
                                           CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(
            &output, options.rows, d_cusparse.data(), CUDA_R_32F));
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        std::size_t workspace_size{};
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, input,
            &beta, output, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            &workspace_size));
        DeviceBuffer<std::byte> workspace(workspace_size);

        const auto springmv_operation = [&] {
            CHECK_CUDA(springmv::csr_spmv(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_values.data(), d_x.data(), d_springmv.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto cusparse_operation = [&] {
            CHECK_CUSPARSE(cusparseSpMV(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse,
                input, &beta, output, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                workspace.data()));
        };
        const auto springmv_unweighted_operation = [&] {
            CHECK_CUDA(springmv::csr_spmv(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_x.data(), d_springmv_unweighted.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto logsumexp_operation = [&] {
            CHECK_CUDA(springmv::csr_logsumexp(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_values.data(), d_x.data(), d_logsumexp.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto logsumexp_unweighted_operation = [&] {
            CHECK_CUDA(springmv::csr_logsumexp(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_x.data(), d_logsumexp_unweighted.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto minplus_operation = [&] {
            CHECK_CUDA(springmv::csr_minplus(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_values.data(), d_x.data(), d_minplus.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto minplus_unweighted_operation = [&] {
            CHECK_CUDA(springmv::csr_minplus(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_x.data(), d_minplus_unweighted.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto maxplus_operation = [&] {
            CHECK_CUDA(springmv::csr_maxplus(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_values.data(), d_x.data(), d_maxplus.data(),
                {options.nnz_per_row, options.subwarp}));
        };
        const auto maxplus_unweighted_operation = [&] {
            CHECK_CUDA(springmv::csr_maxplus(
                options.rows, d_row_offsets.data(), d_column_indices.data(),
                d_x.data(), d_maxplus_unweighted.data(),
                {options.nnz_per_row, options.subwarp}));
        };

        springmv_operation();
        springmv_unweighted_operation();
        cusparse_operation();
        logsumexp_operation();
        logsumexp_unweighted_operation();
        minplus_operation();
        minplus_unweighted_operation();
        maxplus_operation();
        maxplus_unweighted_operation();
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto springmv_output = d_springmv.copy_to_host();
        const auto springmv_unweighted_output =
            d_springmv_unweighted.copy_to_host();
        const auto cusparse_output = d_cusparse.copy_to_host();
        const auto logsumexp_output = d_logsumexp.copy_to_host();
        const auto logsumexp_unweighted_output =
            d_logsumexp_unweighted.copy_to_host();
        const auto minplus_output = d_minplus.copy_to_host();
        const auto minplus_unweighted_output =
            d_minplus_unweighted.copy_to_host();
        const auto maxplus_output = d_maxplus.copy_to_host();
        const auto maxplus_unweighted_output =
            d_maxplus_unweighted.copy_to_host();
        for (std::size_t row = 0; row < springmv_output.size(); ++row) {
            const auto tolerance =
                2.0e-5F + 2.0e-5F * std::abs(cusparse_output[row]);
            if (std::abs(springmv_output[row] - cusparse_output[row]) >
                tolerance) {
                throw std::runtime_error(
                    "SpringMV output differs from cuSPARSE at row " +
                    std::to_string(row));
            }

            double expected_unweighted_arithmetic{};
            for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
                expected_unweighted_arithmetic += x[column_indices[nz]];
            }
            const auto arithmetic_unweighted_tolerance =
                2.0e-5 +
                2.0e-5 * std::abs(expected_unweighted_arithmetic);
            if (std::abs(
                    static_cast<double>(springmv_unweighted_output[row]) -
                    expected_unweighted_arithmetic) >
                arithmetic_unweighted_tolerance) {
                throw std::runtime_error(
                    "SpringMV unweighted arithmetic output differs at row " +
                    std::to_string(row));
            }

            auto maximum = -std::numeric_limits<double>::infinity();
            for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
                const auto term = static_cast<double>(
                    values[nz] + x[column_indices[nz]]);
                maximum = std::max(maximum, term);
            }
            double scaled_sum{};
            for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
                const auto term = static_cast<double>(
                    values[nz] + x[column_indices[nz]]);
                scaled_sum += std::exp(term - maximum);
            }
            const auto expected_logsumexp = maximum + std::log(scaled_sum);
            const auto logsumexp_tolerance =
                8.0e-5 + 6.0e-7 * std::abs(expected_logsumexp);
            if (std::abs(static_cast<double>(logsumexp_output[row]) -
                         expected_logsumexp) > logsumexp_tolerance) {
                throw std::runtime_error(
                    "SpringMV log-sum-exp output differs at row " +
                    std::to_string(row));
            }

            auto unweighted_maximum =
                -std::numeric_limits<double>::infinity();
            for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
                unweighted_maximum = std::max(
                    unweighted_maximum,
                    static_cast<double>(x[column_indices[nz]]));
            }
            double unweighted_scaled_sum{};
            for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
                unweighted_scaled_sum += std::exp(
                    static_cast<double>(x[column_indices[nz]]) -
                    unweighted_maximum);
            }
            const auto expected_unweighted =
                unweighted_maximum + std::log(unweighted_scaled_sum);
            const auto unweighted_tolerance =
                8.0e-5 + 6.0e-7 * std::abs(expected_unweighted);
            if (std::abs(
                    static_cast<double>(logsumexp_unweighted_output[row]) -
                    expected_unweighted) > unweighted_tolerance) {
                throw std::runtime_error(
                    "SpringMV unweighted log-sum-exp output differs at row " +
                    std::to_string(row));
            }

            auto expected_minplus = std::numeric_limits<float>::infinity();
            auto expected_maxplus = -std::numeric_limits<float>::infinity();
            auto expected_minplus_unweighted =
                std::numeric_limits<float>::infinity();
            auto expected_maxplus_unweighted =
                -std::numeric_limits<float>::infinity();
            for (auto nz = row_offsets[row]; nz < row_offsets[row + 1]; ++nz) {
                const auto input = x[column_indices[nz]];
                const auto weighted_term = values[nz] + input;
                expected_minplus = std::min(expected_minplus, weighted_term);
                expected_maxplus = std::max(expected_maxplus, weighted_term);
                expected_minplus_unweighted =
                    std::min(expected_minplus_unweighted, input);
                expected_maxplus_unweighted =
                    std::max(expected_maxplus_unweighted, input);
            }
            if (minplus_output[row] != expected_minplus ||
                maxplus_output[row] != expected_maxplus ||
                minplus_unweighted_output[row] !=
                    expected_minplus_unweighted ||
                maxplus_unweighted_output[row] !=
                    expected_maxplus_unweighted) {
                throw std::runtime_error(
                    "SpringMV tropical output differs at row " +
                    std::to_string(row));
            }
        }

        int device{};
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
        const auto selected_subwarp =
            options.subwarp == 0
                ? springmv::select_subwarp_size(options.nnz_per_row)
                : options.subwarp;
        std::cout << "GPU: " << properties.name << '\n'
                  << "matrix: " << options.rows << " x " << options.cols
                  << ", nnz " << nnz << ", nnz/row "
                  << options.nnz_per_row << '\n'
                  << "SpringMV subwarp: " << selected_subwarp << '\n'
                  << "warmup: " << options.warmup
                  << ", iterations: " << options.iterations << "\n\n";

        const auto springmv_timing = measure(
            springmv_operation, options.warmup, options.iterations);
        const auto springmv_unweighted_timing = measure(
            springmv_unweighted_operation, options.warmup, options.iterations);
        const auto cusparse_timing = measure(
            cusparse_operation, options.warmup, options.iterations);
        const auto logsumexp_timing = measure(
            logsumexp_operation, options.warmup, options.iterations);
        const auto logsumexp_unweighted_timing =
            measure(logsumexp_unweighted_operation, options.warmup,
                    options.iterations);
        const auto minplus_timing =
            measure(minplus_operation, options.warmup, options.iterations);
        const auto minplus_unweighted_timing = measure(
            minplus_unweighted_operation, options.warmup, options.iterations);
        const auto maxplus_timing =
            measure(maxplus_operation, options.warmup, options.iterations);
        const auto maxplus_unweighted_timing = measure(
            maxplus_unweighted_operation, options.warmup, options.iterations);
        print_result("SpringMV arithmetic", springmv_timing, options.rows, nnz,
                     2.0, true);
        print_result("arithmetic unweighted", springmv_unweighted_timing,
                     options.rows, nnz, 1.0, false);
        print_result("cuSPARSE arithmetic", cusparse_timing, options.rows, nnz,
                     2.0, true);
        print_result("SpringMV logsumexp", logsumexp_timing, options.rows, nnz,
                     0.0, true);
        print_result("logsumexp unweighted", logsumexp_unweighted_timing,
                     options.rows, nnz, 0.0, false);
        print_result("SpringMV minplus", minplus_timing, options.rows, nnz,
                     0.0, true);
        print_result("minplus unweighted", minplus_unweighted_timing,
                     options.rows, nnz, 0.0, false);
        print_result("SpringMV maxplus", maxplus_timing, options.rows, nnz,
                     0.0, true);
        print_result("maxplus unweighted", maxplus_unweighted_timing,
                     options.rows, nnz, 0.0, false);

        CHECK_CUSPARSE(cusparseDestroyDnVec(output));
        CHECK_CUSPARSE(cusparseDestroyDnVec(input));
        CHECK_CUSPARSE(cusparseDestroySpMat(sparse));
        CHECK_CUSPARSE(cusparseDestroy(handle));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

