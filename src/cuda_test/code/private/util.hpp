#pragma once

#include <memory>
#include <chrono>
#include <span>

using CudaDeleter = decltype([](auto* p){ cudaFree(p); });

template<typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

template<typename T>
CudaPtr<T> MakeCudaArray(size_t elements_count)
{
    T* device_ptr{};
    cudaMalloc(&device_ptr, sizeof(T) * elements_count);
    return CudaPtr<T>(device_ptr);
}

template<typename T, size_t extent>
void CopyCudaArrayToDevice(std::span<T, extent> host, const CudaPtr<std::remove_const_t<T>>& device)
{
    cudaMemcpy(device.get(), host.data(), host.size_bytes(), cudaMemcpyHostToDevice);
}

template<typename T, size_t extent>
void CopyCudaArrayToHost(std::span<T, extent> host, const CudaPtr<std::remove_const_t<T>>& device)
{
    cudaMemcpy(host.data(), device.get(), host.size_bytes(), cudaMemcpyDeviceToHost);
}

template<typename F>
auto MeasureMillis(F&& f)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float, std::milli>;
    auto start_time = Clock::now();
    f();
    auto finish_time = Clock::now();
    return std::chrono::duration_cast<Duration>(finish_time - start_time).count();
}
