#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <cassert>
#include <cpptrace/cpptrace.hpp>
#include <iterator>
#include <memory>
#include <span>

#include "klvk/error_handling.hpp"
#include "klvk/vulkan/vulkan_common.hpp"

namespace klvk
{
class DeviceContext;
}

namespace verlet
{
using CudaDeleter = decltype([](auto* p) { cudaFree(p); });

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

template <typename... Args>
static void CheckResult(cudaError_t result, fmt::format_string<Args...> format_string = "", Args&&... args)
{
    if (result != cudaSuccess)
    {
        std::string message =
            fmt::format("Cuda operation returned an error code. Error: {}. ", cudaGetErrorString(result));
        if (fmt::formatted_size(format_string, args...))
        {
            message.append("\nContext: ");
            fmt::format_to(std::back_inserter(message), format_string, std::forward<Args>(args)...);
        }

        throw cpptrace::runtime_error(std::move(message));
    }
}

// A buffer that Vulkan and CUDA both address: the Vulkan allocation is exported as an
// opaque fd and imported by CUDA, so kernels write the objects in place and the very
// same VkBuffer is bound as the instance vertex buffer - no copies between the APIs.
//
// This replaces the cudaGraphicsGLRegisterBuffer path used with OpenGL. Unlike the GL
// interop there is no map/unmap: the device pointer stays valid for the buffer's lifetime.
// Access has to be synchronized by the caller (see VerletCudaApp::Tick).
class CudaVkBuffer
{
public:
    CudaVkBuffer() = default;
    CudaVkBuffer(klvk::DeviceContext& context, size_t bytes);
    CudaVkBuffer(const CudaVkBuffer&) = delete;
    CudaVkBuffer(CudaVkBuffer&& other) noexcept;
    CudaVkBuffer& operator=(CudaVkBuffer&& other) noexcept;
    ~CudaVkBuffer();

    [[nodiscard]] bool IsValid() const noexcept { return buffer_ != VK_NULL_HANDLE; }
    [[nodiscard]] VkBuffer GetHandle() const noexcept { return buffer_; }
    [[nodiscard]] size_t GetSize() const noexcept { return size_; }

    // Device pointer to the same memory the VkBuffer is bound to.
    template <typename T>
    [[nodiscard]] std::span<T> GetDeviceSpan() const
    {
        klvk::ErrorHandling::Ensure(
            size_ % sizeof(T) == 0,
            "Possibly wrong conversion: converting {} bytes to an array of objects with size {}",
            size_,
            sizeof(T));
        return std::span{reinterpret_cast<T*>(device_ptr_), size_ / sizeof(T)};  // NOLINT
    }

private:
    void Destroy() noexcept;

    klvk::DeviceContext* context_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    cudaExternalMemory_t external_memory_{};
    void* device_ptr_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
[[nodiscard]] static CudaPtr<T> MakeCudaArray(size_t elements_count, cudaStream_t stream)
{
    void* device_ptr{};
    CheckResult(cudaMallocAsync(&device_ptr, elements_count * sizeof(T), stream));
    return CudaPtr<T>(reinterpret_cast<T*>(device_ptr));  // NOLINT
}

template <typename T, size_t extent = std::dynamic_extent>
static void CudaMemset(std::span<T, extent> data, uint8_t value, cudaStream_t stream)
{
    CheckResult(cudaMemsetAsync(data.data(), value, data.size_bytes(), stream));
}

template <typename T, size_t extent_src = std::dynamic_extent, size_t extent_dst = std::dynamic_extent>
static void
CudaMemcpy(std::span<T, extent_src> src, std::span<T, extent_dst> dst, cudaMemcpyKind kind, cudaStream_t stream)
{
    klvk::ErrorHandling::Ensure(
        src.size() == dst.size(),
        "Attempt to memcpy regions with different sizes. {} -> {}",
        src.size(),
        dst.size());
    CheckResult(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(), kind, stream));
}

}  // namespace verlet
