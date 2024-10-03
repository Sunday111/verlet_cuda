#include "cuda_util.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace verlet
{

cudaGraphicsResource* CudaGlInterop::RegisterResource(klgl::GlBufferId buffer)
{
    cudaGraphicsResource* resource = nullptr;
    CheckResult(cudaGraphicsGLRegisterBuffer(&resource, buffer.GetValue(), cudaGraphicsMapFlagsNone));
    return resource;
}

void CudaGlInterop::UnregisterResource(cudaGraphicsResource* resource)
{
    auto err = cudaGraphicsUnregisterResource(resource);
    assert(err == cudaSuccess);
}

CudaMappedGraphicsResourcePtr CudaGlInterop::MapResource(cudaGraphicsResource* resource)
{
    CheckResult(cudaGraphicsMapResources(1, &resource, 0));
    return CudaMappedGraphicsResourcePtr{resource};
}

std::span<uint8_t> CudaGlInterop::GetDeviceDataPtr(const CudaMappedGraphicsResourcePtr& mapped_resource)
{
    void* device_ptr = nullptr;
    size_t num_bytes = 0;
    CheckResult(cudaGraphicsResourceGetMappedPointer(&device_ptr, &num_bytes, mapped_resource.get()));

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
}  // namespace verlet
