#pragma once

#include "edt/math/matrix.hpp"
#include "constants.hpp"
#include "driver_types.h"
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

// C++ interface to invoke cuda kernels
class Kernels
{
public:
    // Return the launch status instead of reporting it here: this header is also included by
    // kernels.cu, which is compiled against a different standard library than the rest of the
    // project, so only trivial types may cross the boundary.
    [[nodiscard]] static cudaError_t
    PopulateGrid(cudaStream_t& stream, GridCell* cells, VerletObject* objects, size_t num_objects);
    [[nodiscard]] static cudaError_t
    SolveCollisions(cudaStream_t& stream, GridCell* cells, VerletObject* objects, edt::Vec2<size_t> offset);
    [[nodiscard]] static cudaError_t
    UpdatePositions(cudaStream_t& stream, size_t num_objects, VerletObject* objects);
};
}  // namespace verlet
