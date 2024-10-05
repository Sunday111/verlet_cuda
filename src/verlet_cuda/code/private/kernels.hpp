#pragma once

#include "EverydayTools/Math/Matrix.hpp"
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
    static void ClearGrid(cudaStream_t& stream, GridCell* cells);
    static void PopulateGrid(cudaStream_t& stream, GridCell* cells, std::span<VerletObject> objects);
    static void SolveCollisions(cudaStream_t& stream, GridCell* cells, VerletObject* objects, edt::Vec2<size_t> offset);
    static void UpdatePositions(cudaStream_t& stream, size_t num_objects, VerletObject* objects);
};
}  // namespace verlet
