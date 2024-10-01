#pragma once

#include <cassert>
#include <span>

#include "klgl/opengl/identifiers.hpp"

struct cudaGraphicsResource;

class CudaGlInterop
{
public:
    static cudaGraphicsResource* RegisterBuffer(klgl::GlBufferId buffer);
    static void UnregisterBuffer(cudaGraphicsResource* resource);
    static void ModifyVBO(cudaGraphicsResource* registered_positions_vbo, float t);
    static std::span<uint8_t> MapResourceAndGetPtr(cudaGraphicsResource* resource);
    static void UnmapResource(cudaGraphicsResource* resource);
};
