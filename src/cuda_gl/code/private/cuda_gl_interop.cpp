#include "cuda_gl_interop.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

cudaGraphicsResource* CudaGlInterop::RegisterBuffer(klgl::GlBufferId buffer)
{
    cudaGraphicsResource* resource = nullptr;
    auto err = cudaGraphicsGLRegisterBuffer(&resource, buffer.GetValue(), cudaGraphicsMapFlagsNone);
    assert(err == cudaSuccess);
    return resource;
}

void CudaGlInterop::UnregisterBuffer(cudaGraphicsResource* resource)
{
    auto err = cudaGraphicsUnregisterResource(resource);
    assert(err == cudaSuccess);
}

std::span<uint8_t> CudaGlInterop::MapResourceAndGetPtr(cudaGraphicsResource* resource)
{
    auto err = cudaGraphicsMapResources(1, &resource, 0);
    assert(err == cudaSuccess);

    void* device_ptr = nullptr;
    size_t num_bytes = 0;
    err = cudaGraphicsResourceGetMappedPointer(&device_ptr, &num_bytes, resource);
    assert(err == cudaSuccess);

    return std::span{
        reinterpret_cast<uint8_t*>(device_ptr),  // NOLINT
        num_bytes,
    };
}

void CudaGlInterop::UnmapResource(cudaGraphicsResource* resource)
{
    auto err = cudaGraphicsUnmapResources(1, &resource, 0);
    assert(err == cudaSuccess);
}
