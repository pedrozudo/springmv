// Copyright 2026 SpringMV contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "springmv/csr_transpose.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

namespace springmv {
namespace {

constexpr int block_size = 256;
constexpr std::size_t workspace_alignment = 256;

struct WorkspaceLayout {
    std::size_t counts_offset{};
    std::size_t sorted_keys_offset{};
    std::size_t input_ids_offset{};
    std::size_t sorted_ids_offset{};
    std::size_t temporary_offset{};
    std::size_t total_size{};
    std::size_t temporary_size{};
};

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

cudaError_t make_workspace_layout(std::int32_t num_cols,
                                  std::int32_t num_nonzeros,
                                  WorkspaceLayout* layout) noexcept
{
    if (num_nonzeros == 0) {
        *layout = {};
        return cudaSuccess;
    }

    std::size_t scan_bytes{};
    auto status = cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, static_cast<const std::int32_t*>(nullptr),
        static_cast<std::int32_t*>(nullptr), num_cols);
    if (status != cudaSuccess) {
        return status;
    }

    std::size_t sort_bytes{};
    status = cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, static_cast<const std::int32_t*>(nullptr),
        static_cast<std::int32_t*>(nullptr),
        static_cast<const std::int32_t*>(nullptr),
        static_cast<std::int32_t*>(nullptr), num_nonzeros, 0, 32);
    if (status != cudaSuccess) {
        return status;
    }

    std::size_t offset{};
    layout->counts_offset = offset;
    offset += static_cast<std::size_t>(num_cols) * sizeof(std::int32_t);
    layout->sorted_keys_offset = offset;
    offset += static_cast<std::size_t>(num_nonzeros) * sizeof(std::int32_t);
    layout->input_ids_offset = offset;
    offset += static_cast<std::size_t>(num_nonzeros) * sizeof(std::int32_t);
    layout->sorted_ids_offset = offset;
    offset += static_cast<std::size_t>(num_nonzeros) * sizeof(std::int32_t);
    layout->temporary_offset = align_up(offset, workspace_alignment);
    layout->temporary_size = std::max(scan_bytes, sort_bytes);
    layout->total_size = layout->temporary_offset + layout->temporary_size;
    return cudaSuccess;
}

__global__ void initialize_edge_ids(std::int32_t num_nonzeros,
                                    std::int32_t* edge_ids)
{
    const auto index = static_cast<std::int32_t>(blockIdx.x) * blockDim.x +
                       static_cast<std::int32_t>(threadIdx.x);
    if (index < num_nonzeros) {
        edge_ids[index] = index;
    }
}

__global__ void count_columns(std::int32_t num_nonzeros,
                              const std::int32_t* column_indices,
                              std::int32_t* counts)
{
    const auto index = static_cast<std::int32_t>(blockIdx.x) * blockDim.x +
                       static_cast<std::int32_t>(threadIdx.x);
    if (index < num_nonzeros) {
        atomicAdd(&counts[column_indices[index]], 1);
    }
}

__global__ void set_final_offset(std::int32_t num_cols,
                                 std::int32_t num_nonzeros,
                                 std::int32_t* row_offsets)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        row_offsets[num_cols] = num_nonzeros;
    }
}

__device__ std::int32_t find_original_row(
    std::int32_t edge, std::int32_t num_rows,
    const std::int32_t* row_offsets)
{
    std::int32_t lower{};
    auto upper = num_rows;
    while (lower + 1 < upper) {
        const auto middle = lower + (upper - lower) / 2;
        if (row_offsets[middle] <= edge) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return lower;
}

template <bool weighted>
__global__ void gather_transpose(
    std::int32_t num_rows, std::int32_t num_nonzeros,
    const std::int32_t* row_offsets, const float* values,
    const std::int32_t* sorted_edge_ids,
    std::int32_t* transposed_column_indices, float* transposed_values,
    std::int32_t* edge_permutation)
{
    const auto index = static_cast<std::int32_t>(blockIdx.x) * blockDim.x +
                       static_cast<std::int32_t>(threadIdx.x);
    if (index < num_nonzeros) {
        const auto edge = sorted_edge_ids[index];
        transposed_column_indices[index] =
            find_original_row(edge, num_rows, row_offsets);
        if constexpr (weighted) {
            transposed_values[index] = values[edge];
        }
        if (edge_permutation != nullptr) {
            edge_permutation[index] = edge;
        }
    }
}

template <bool weighted>
cudaError_t transpose_impl(
    std::int32_t num_rows, std::int32_t num_cols,
    std::int32_t num_nonzeros, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* values,
    std::int32_t* transposed_row_offsets,
    std::int32_t* transposed_column_indices, float* transposed_values,
    std::int32_t* edge_permutation, void* workspace,
    std::size_t workspace_size, cudaStream_t stream) noexcept
{
    if (num_rows < 0 || num_cols < 0 || num_nonzeros < 0 ||
        (num_nonzeros > 0 && (num_rows == 0 || num_cols == 0))) {
        return cudaErrorInvalidValue;
    }
    if (transposed_row_offsets == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (num_nonzeros == 0) {
        return cudaMemsetAsync(
            transposed_row_offsets, 0,
            (static_cast<std::size_t>(num_cols) + 1) * sizeof(std::int32_t),
            stream);
    }
    if (row_offsets == nullptr || column_indices == nullptr ||
        transposed_column_indices == nullptr ||
        (weighted && (values == nullptr || transposed_values == nullptr))) {
        return cudaErrorInvalidDevicePointer;
    }

    WorkspaceLayout layout;
    auto status = make_workspace_layout(num_cols, num_nonzeros, &layout);
    if (status != cudaSuccess) {
        return status;
    }
    if (workspace == nullptr || workspace_size < layout.total_size) {
        return cudaErrorInvalidValue;
    }

    auto* base = static_cast<std::byte*>(workspace);
    auto* counts = reinterpret_cast<std::int32_t*>(base + layout.counts_offset);
    auto* sorted_keys =
        reinterpret_cast<std::int32_t*>(base + layout.sorted_keys_offset);
    auto* input_ids =
        reinterpret_cast<std::int32_t*>(base + layout.input_ids_offset);
    auto* sorted_ids =
        reinterpret_cast<std::int32_t*>(base + layout.sorted_ids_offset);
    auto* temporary = base + layout.temporary_offset;

    status = cudaMemsetAsync(
        counts, 0, static_cast<std::size_t>(num_cols) * sizeof(std::int32_t),
        stream);
    if (status != cudaSuccess) {
        return status;
    }
    const auto blocks = static_cast<unsigned int>(
        (static_cast<std::int64_t>(num_nonzeros) + block_size - 1) /
        block_size);
    count_columns<<<blocks, block_size, 0, stream>>>(num_nonzeros,
                                                     column_indices, counts);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return status;
    }

    auto temporary_size = layout.temporary_size;
    status = cub::DeviceScan::ExclusiveSum(
        temporary, temporary_size, counts, transposed_row_offsets, num_cols,
        stream);
    if (status != cudaSuccess) {
        return status;
    }
    set_final_offset<<<1, 1, 0, stream>>>(num_cols, num_nonzeros,
                                          transposed_row_offsets);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return status;
    }

    initialize_edge_ids<<<blocks, block_size, 0, stream>>>(num_nonzeros,
                                                            input_ids);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        return status;
    }
    temporary_size = layout.temporary_size;
    status = cub::DeviceRadixSort::SortPairs(
        temporary, temporary_size, column_indices, sorted_keys, input_ids,
        sorted_ids, num_nonzeros, 0, 32, stream);
    if (status != cudaSuccess) {
        return status;
    }

    gather_transpose<weighted><<<blocks, block_size, 0, stream>>>(
        num_rows, num_nonzeros, row_offsets, values, sorted_ids,
        transposed_column_indices, transposed_values, edge_permutation);
    return cudaPeekAtLastError();
}

}  // namespace

cudaError_t csr_transpose_workspace_size(std::int32_t num_rows,
                                         std::int32_t num_cols,
                                         std::int32_t num_nonzeros,
                                         std::size_t* workspace_size) noexcept
{
    if (workspace_size == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (num_rows < 0 || num_cols < 0 || num_nonzeros < 0 ||
        (num_nonzeros > 0 && (num_rows == 0 || num_cols == 0))) {
        return cudaErrorInvalidValue;
    }
    WorkspaceLayout layout;
    const auto status =
        make_workspace_layout(num_cols, num_nonzeros, &layout);
    if (status == cudaSuccess) {
        *workspace_size = layout.total_size;
    }
    return status;
}

cudaError_t csr_transpose(
    std::int32_t num_rows, std::int32_t num_cols,
    std::int32_t num_nonzeros, const std::int32_t* row_offsets,
    const std::int32_t* column_indices,
    std::int32_t* transposed_row_offsets,
    std::int32_t* transposed_column_indices,
    std::int32_t* edge_permutation, void* workspace,
    std::size_t workspace_size, cudaStream_t stream) noexcept
{
    return transpose_impl<false>(
        num_rows, num_cols, num_nonzeros, row_offsets, column_indices, nullptr,
        transposed_row_offsets, transposed_column_indices, nullptr,
        edge_permutation, workspace, workspace_size, stream);
}

cudaError_t csr_transpose(
    std::int32_t num_rows, std::int32_t num_cols,
    std::int32_t num_nonzeros, const std::int32_t* row_offsets,
    const std::int32_t* column_indices, const float* values,
    std::int32_t* transposed_row_offsets,
    std::int32_t* transposed_column_indices, float* transposed_values,
    std::int32_t* edge_permutation, void* workspace,
    std::size_t workspace_size, cudaStream_t stream) noexcept
{
    return transpose_impl<true>(
        num_rows, num_cols, num_nonzeros, row_offsets, column_indices, values,
        transposed_row_offsets, transposed_column_indices, transposed_values,
        edge_permutation, workspace, workspace_size, stream);
}

}  // namespace springmv

