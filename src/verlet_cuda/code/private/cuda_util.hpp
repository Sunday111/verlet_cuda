#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

#include <cassert>
#include <cpptrace/cpptrace.hpp>
#include <memory>
#include <span>

#include "klgl/error_handling.hpp"
#include "klgl/opengl/identifiers.hpp"

struct cudaGraphicsResource;

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

template <typename T>
[[nodiscard]] static constexpr std::span<T> ReinterpretSpan(std::span<uint8_t> span)
{
    klgl::ErrorHandling::Ensure(
        span.size_bytes() % sizeof(T) == 0,
        "Possibly wrong conversion: converting {} bytes to an array of objects with size {}",
        span.size_bytes(),
        sizeof(T));
    return std::span{
        reinterpret_cast<T*>(span.data()),  // NOLINT
        span.size_bytes() / sizeof(T),
    };
}

struct CudaMappedGraphicsResourceDeleter
{
    inline void operator()(cudaGraphicsResource* resource);
};

struct CudaGraphicsResourceDeleter
{
    inline void operator()(cudaGraphicsResource* resource);
};

using CudaMappedGraphicsResourcePtr = std::unique_ptr<cudaGraphicsResource, CudaMappedGraphicsResourceDeleter>;
using CudaGraphicsResourcePtr = std::unique_ptr<cudaGraphicsResource, CudaGraphicsResourceDeleter>;

class CudaGlInterop
{
public:
    static cudaGraphicsResource* RegisterResource(klgl::GlBufferId buffer);
    static void UnregisterResource(cudaGraphicsResource* resource);

    static CudaMappedGraphicsResourcePtr MapResource(cudaGraphicsResource* resource);
    static void UnmapResource(cudaGraphicsResource* resource);

    static std::span<uint8_t> GetDeviceDataPtr(const CudaMappedGraphicsResourcePtr& mapped_resource);

    template <typename T>
    [[nodiscard]] static std::tuple<CudaMappedGraphicsResourcePtr, std::span<T>> MapAndGet(
        const CudaGraphicsResourcePtr& resource)
    {
        auto mapped_resource = MapResource(resource.get());
        auto data = GetDeviceDataPtr(mapped_resource);
        return {std::move(mapped_resource), ReinterpretSpan<T>(data)};
    }
};

void CudaGraphicsResourceDeleter::operator()(cudaGraphicsResource* resource)
{
    CudaGlInterop::UnregisterResource(resource);
}

void CudaMappedGraphicsResourceDeleter::operator()(cudaGraphicsResource* resource)
{
    CudaGlInterop::UnmapResource(resource);
}

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
    klgl::ErrorHandling::Ensure(
        src.size() == dst.size(),
        "Attempt to memcpy regions with different sizes. {} -> {}",
        src.size(),
        dst.size());
    CheckResult(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(), kind, stream));
}

}  // namespace verlet
