#include "cuda_util.hpp"

#include <cuda_runtime.h>

#include <utility>

#include "klvk/vulkan/device_context.hpp"
#include "klvk/vulkan/vulkan_api.hpp"

// Vulkan create-info structs are designed for partial designated initialization;
// unlisted fields must be zero.
#ifdef __clang__
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace verlet
{
namespace
{

uint32_t FindMemoryType(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t i = 0; i != memory_properties.memoryTypeCount; ++i)
    {
        const bool type_allowed = (type_bits & (1u << i)) != 0;
        const bool has_properties = (memory_properties.memoryTypes[i].propertyFlags & properties) == properties;
        if (type_allowed && has_properties) return i;
    }
    throw cpptrace::runtime_error("Failed to find a Vulkan memory type suitable for CUDA interop");
}

}  // namespace

CudaVkBuffer::CudaVkBuffer(klvk::DeviceContext& context, size_t bytes) : context_(&context), size_(bytes)
{
    klvk::ErrorHandling::Ensure(
        context.IsExternalMemoryFdEnabled(),
        "The Vulkan device does not support VK_KHR_external_memory_fd, which CUDA interop requires");

    const VkDevice device = context.GetDevice();

    // Declaring the handle type up front is what makes the allocation exportable.
    const VkExternalMemoryBufferCreateInfo external_buffer_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_buffer_info,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    klvk::CheckVkResult(vkCreateBuffer(device, &buffer_info, nullptr, &buffer_), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer_, &requirements);

    const VkExportMemoryAllocateInfo export_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    // NVIDIA wants interop allocations dedicated to the resource; CUDA is told the same
    // below through cudaExternalMemoryDedicated.
    const VkMemoryDedicatedAllocateInfo dedicated_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &export_info,
        .buffer = buffer_,
    };
    const VkMemoryAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated_info,
        .allocationSize = requirements.size,
        .memoryTypeIndex = FindMemoryType(
            context.GetPhysicalDevice(),
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    klvk::CheckVkResult(vkAllocateMemory(device, &allocate_info, nullptr, &memory_), "vkAllocateMemory");
    klvk::CheckVkResult(vkBindBufferMemory(device, buffer_, memory_, 0), "vkBindBufferMemory");

    const VkMemoryGetFdInfoKHR fd_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memory_,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    int fd = -1;
    klvk::CheckVkResult(vkGetMemoryFdKHR(device, &fd_info, &fd), "vkGetMemoryFdKHR");

    cudaExternalMemoryHandleDesc handle_desc{};
    handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    // CUDA's C ABI requires selecting the fd member of this tagged handle union.
    handle_desc.handle.fd = fd;  // NOLINT(cppcoreguidelines-pro-type-union-access)
    handle_desc.size = requirements.size;
    handle_desc.flags = cudaExternalMemoryDedicated;
    // On success CUDA takes ownership of the fd and closes it itself, so it must not be
    // closed here. On failure it stays leaked - but the throw below tears the app down anyway.
    CheckResult(cudaImportExternalMemory(&external_memory_, &handle_desc));

    cudaExternalMemoryBufferDesc buffer_desc{};
    buffer_desc.offset = 0;
    buffer_desc.size = bytes;
    buffer_desc.flags = 0;
    CheckResult(cudaExternalMemoryGetMappedBuffer(&device_ptr_, external_memory_, &buffer_desc));
}

CudaVkBuffer::CudaVkBuffer(CudaVkBuffer&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      memory_(std::exchange(other.memory_, VK_NULL_HANDLE)),
      external_memory_(std::exchange(other.external_memory_, cudaExternalMemory_t{})),
      device_ptr_(std::exchange(other.device_ptr_, nullptr)),
      size_(std::exchange(other.size_, 0))
{
}

CudaVkBuffer& CudaVkBuffer::operator=(CudaVkBuffer&& other) noexcept
{
    if (this != &other)
    {
        Destroy();
        context_ = std::exchange(other.context_, nullptr);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
        external_memory_ = std::exchange(other.external_memory_, cudaExternalMemory_t{});
        device_ptr_ = std::exchange(other.device_ptr_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

CudaVkBuffer::~CudaVkBuffer()
{
    Destroy();
}

void CudaVkBuffer::Destroy() noexcept
{
    if (device_ptr_) cudaFree(device_ptr_);
    if (external_memory_) cudaDestroyExternalMemory(external_memory_);
    if (context_)
    {
        const VkDevice device = context_->GetDevice();
        if (buffer_) vkDestroyBuffer(device, buffer_, nullptr);
        if (memory_) vkFreeMemory(device, memory_, nullptr);
    }
    device_ptr_ = nullptr;
    external_memory_ = {};
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    size_ = 0;
    context_ = nullptr;
}

}  // namespace verlet
