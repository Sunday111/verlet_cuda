#pragma once

#include "constants.hpp"
#include "driver_types.h"
#include "edt/math/matrix.hpp"
#include "verlet_object.hpp"

namespace verlet
{

class GridCell
{
public:
    [[nodiscard]] static constexpr Vec2<size_t> LocationToCell(const Vec2f& location)
    {
        return (constants::kWorldRange.Clamp(location) - constants::kWorldRange.Min()).Cast<size_t>() /
               constants::kGridCellSize;
    }

    [[nodiscard]] static constexpr size_t CellToCellIndex(const Vec2<size_t>& cell)
    {
        return cell.x() + cell.y() * constants::kGridSize.x();
    }

    [[nodiscard]] static constexpr size_t LocationToCellIndex(const Vec2f& location)
    {
        return CellToCellIndex(LocationToCell(location));
    }

    uint32_t first_object_index = kInvalidObjectIndex;
};

struct CollisionCacheMetadata
{
    uint32_t object_count;
    uint32_t last_occupied_cell;
};

struct CollisionCache
{
    VerletObject* objects = nullptr;
    uint32_t* original_indices = nullptr;
    CollisionCacheMetadata* metadata = nullptr;
    Vec2f* previous_positions = nullptr;
};

// C++ interface to invoke cuda kernels
class Kernels
{
public:
    [[nodiscard]] static cudaError_t PopulateGrid(
        cudaStream_t& stream,
        GridCell* cells,
        VerletObject* objects,
        size_t num_objects,
        const uint32_t* original_indices = nullptr,
        uint32_t* last_occupied_cell = nullptr);
    [[nodiscard]] static cudaError_t SolveCollisions(
        cudaStream_t& stream,
        GridCell* cells,
        VerletObject* objects,
        edt::Vec2<size_t> offset,
        const uint32_t* original_indices = nullptr,
        const uint32_t* last_occupied_cell = nullptr);
    [[nodiscard]] static cudaError_t CacheGrid(
        cudaStream_t stream,
        GridCell* cells,
        const VerletObject* objects,
        const Vec2f* previous_positions,
        CollisionCache cache);
    [[nodiscard]] static cudaError_t RestorePositions(
        cudaStream_t stream,
        size_t num_objects,
        VerletObject* objects,
        Vec2f* previous_positions,
        CollisionCache cache);
    [[nodiscard]] static cudaError_t
    UpdatePositions(cudaStream_t& stream, size_t num_objects, VerletObject* objects, Vec2f* previous_positions);
};
}  // namespace verlet
