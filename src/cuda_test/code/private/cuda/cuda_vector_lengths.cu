#include "util.hpp"
#include "time.hpp"
#include "cuda_algorithms.hpp"
#include "fmt/core.h"
#include "fmt/chrono.h"

__global__ void compute_vectors_lengths(size_t n, const edt::Vec3f* vectors, float* lengths) {
    if (const size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n) {
        auto& v = vectors[i];
        lengths[i] = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    }
}

void CudaAlgorithms::VectorsLengths(std::span<const edt::Vec3f> vectors, std::span<float> out_lengths) {
    fmt::println("Cuda compute vector lengths");

    const size_t n = vectors.size();

    auto device_vectors = MakeCudaArray<edt::Vec3f>(vectors.size());
    auto device_lengths = MakeCudaArray<float>(vectors.size());
    const auto copy_to_device_duration = MeasureTime([&] {
        CopyCudaArrayToDevice(vectors, device_vectors);
    });

    fmt::println("  Copy to device: {:.2}", copy_to_device_duration);

    constexpr uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(n) + threads_per_block - 1) / threads_per_block;
    fmt::println("  Blocks: {}", num_blocks);
    fmt::println("  Threads per block: {}", threads_per_block);
    const auto compute_vectors_lengths_duration = MeasureTime([&] {
        compute_vectors_lengths<<<num_blocks, threads_per_block>>>(n, device_vectors.get(), device_lengths.get());
    });

    fmt::println("  Compute vectors lengths: {:.2}", compute_vectors_lengths_duration);

    const auto copy_to_host_duration = MeasureTime([&] {
        CopyCudaArrayToHost(out_lengths, device_lengths);
    });

    fmt::println("  Copy to host: {:.2}", copy_to_host_duration);
}
