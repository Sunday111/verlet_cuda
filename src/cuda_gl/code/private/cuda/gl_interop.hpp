#pragma once

#include <span>
#include <cassert>
#include "klgl/opengl/identifiers.hpp"

struct cudaGraphicsResource;

template<typename T>
[[nodiscard]] std::span<T> ReinterpretSpan(std::span<uint8_t> span)
{
    assert(span.size_bytes() % sizeof(T) == 0);
    return std::span{
        reinterpret_cast<T*>(span.data()), // NOLINT
        span.size_bytes() / sizeof(T),
    };
}

class CudaGlInterop
{
public:
    static cudaGraphicsResource* RegisterBuffer(klgl::GlBufferId buffer);
    static void UnregisterBuffer(cudaGraphicsResource* resource);
    static void ModifyVBO(cudaGraphicsResource* registered_positions_vbo, float t);
    static std::span<uint8_t> MapResourceAndGetPtr(cudaGraphicsResource* resource);
    static void UnmapResource(cudaGraphicsResource* resource);
};
