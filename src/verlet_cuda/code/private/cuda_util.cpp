#include "cuda_util.hpp"

#include <cuda_runtime.h>

#include <utility>

#include "klvk/vulkan/device_context.hpp"
#include "klvk/vulkan/vulkan_common.hpp"

namespace verlet
{
namespace
{

uint32_t FindMemoryType(vk::PhysicalDevice physical_device, uint32_t type_bits, vk::MemoryPropertyFlags properties)
{
    const vk::PhysicalDeviceMemoryProperties memory_properties = physical_device.getMemoryProperties();
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

    const vk::Device device = context.GetDevice();

    // Declaring the handle type up front is what makes the allocation exportable.
    vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_chain;
    buffer_chain.get<vk::ExternalMemoryBufferCreateInfo>().setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    buffer_chain.get<vk::BufferCreateInfo>()
        .setSize(bytes)
        .setUsage(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst)
        .setSharingMode(vk::SharingMode::eExclusive);
    buffer_ = device.createBuffer(buffer_chain.get<vk::BufferCreateInfo>());

    const vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(buffer_);

    // NVIDIA wants interop allocations dedicated to the resource; CUDA is told the same
    // below through cudaExternalMemoryDedicated.
    vk::StructureChain<vk::MemoryAllocateInfo, vk::MemoryDedicatedAllocateInfo, vk::ExportMemoryAllocateInfo>
        allocation_chain;
    allocation_chain.get<vk::ExportMemoryAllocateInfo>().setHandleTypes(
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    allocation_chain.get<vk::MemoryDedicatedAllocateInfo>().setBuffer(buffer_);
    allocation_chain.get<vk::MemoryAllocateInfo>()
        .setAllocationSize(requirements.size)
        .setMemoryTypeIndex(FindMemoryType(
            context.GetPhysicalDevice(),
            requirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal));
    memory_ = device.allocateMemory(allocation_chain.get<vk::MemoryAllocateInfo>());
    device.bindBufferMemory(buffer_, memory_, 0);

    const vk::MemoryGetFdInfoKHR fd_info =
        vk::MemoryGetFdInfoKHR{}.setMemory(memory_).setHandleType(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
    const int fd = device.getMemoryFdKHR(fd_info);

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
      buffer_(std::exchange(other.buffer_, nullptr)),
      memory_(std::exchange(other.memory_, nullptr)),
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
        buffer_ = std::exchange(other.buffer_, nullptr);
        memory_ = std::exchange(other.memory_, nullptr);
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
        const vk::Device device = context_->GetDevice();
        if (buffer_) device.destroyBuffer(buffer_);
        if (memory_) device.freeMemory(memory_);
    }
    device_ptr_ = nullptr;
    external_memory_ = {};
    buffer_ = nullptr;
    memory_ = nullptr;
    size_ = 0;
    context_ = nullptr;
}

}  // namespace verlet
