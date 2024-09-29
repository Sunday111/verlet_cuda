#include <stdio.h>
#include <vector>
#include <span>
#include <ranges>
#include <algorithm>
#include <bit>

#include "util.hpp"
#include "time.hpp"
#include "cuda_algorithms.hpp"
#include "fmt/core.h"
#include "fmt/chrono.h"

__global__ void find_max_value_int(size_t n, int* values)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    auto& max_value = values[n];
    if (i < n) {
        const auto value = values[i];
        while (true) {
            if (const auto prev_max = max_value; value <= prev_max ||
                atomicCAS(reinterpret_cast<uint32_t*>(&max_value),
                    std::bit_cast<uint32_t>(prev_max),
                    std::bit_cast<uint32_t>(value))
                    == std::bit_cast<uint32_t>(prev_max))
            {
                break;
            }
        }
    }
}

int CudaAlgorithms::MaxElement(std::span<const int> values) {
    fmt::println("Cuda find max element");

    const size_t n = values.size();

    auto max_value = values.front();
    auto max_value_view = std::span{&max_value, 1};

    // Store one more value to use the same buffer for input and output
    auto device_values = MakeCudaArray<int>(values.size() + 1);
    const auto copy_to_device_duration = MeasureTime([&] {
        CopyCudaArrayToDevice(values, device_values);
        CopyCudaArrayToDevice(max_value_view, device_values.get() + values.size());
    });

    fmt::println("  Copy to device: {:.2}", copy_to_device_duration);

    constexpr uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(n) + threads_per_block - 1) / threads_per_block;
    const auto find_max_value_cas_duration = MeasureTime([&] {
        find_max_value_int<<<num_blocks, threads_per_block>>>(n, device_values.get());
    });

    fmt::println("  Find max value CAS: {:.2}", find_max_value_cas_duration);

    const auto copy_to_host_duration = MeasureTime([&] {
        CopyCudaArrayToHost(max_value_view, device_values.get() + values.size());
    });

    fmt::println("    Copy to host: {:.2}", copy_to_host_duration);
    fmt::println("    Max value: {}", max_value);

    return max_value;
}
