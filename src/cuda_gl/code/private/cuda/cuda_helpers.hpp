#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <memory>
#include <span>

using CudaDeleter = decltype([](auto* p) { cudaFree(p); });

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
CudaPtr<T> MakeCudaArray(size_t elements_count)
{
    T* device_ptr{};
    cudaMalloc(&device_ptr, sizeof(T) * elements_count);
    return CudaPtr<T>(device_ptr);
}

template <typename T, size_t extent>
void CopyCudaArrayToDevice(std::span<T, extent> host, std::remove_const_t<T>* device)
{
    auto result = cudaMemcpy(device, host.data(), host.size_bytes(), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
}

template <typename T, size_t extent>
void CopyCudaArrayToDevice(std::span<T, extent> host, const CudaPtr<std::remove_const_t<T>>& device)
{
    CopyCudaArrayToDevice(host, device.get());
}

template <typename T, size_t extent>
void CopyCudaArrayToHost(std::span<T, extent> host, std::remove_const_t<T>* device)
{
    auto result = cudaMemcpy(host.data(), device, host.size_bytes(), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
}

template <typename T, size_t extent>
void CopyCudaArrayToHost(std::span<T, extent> host, const CudaPtr<std::remove_const_t<T>>& device)
{
    CopyCudaArrayToHost(host, device.get());
}
