#pragma once

#include "EverydayTools/Math/Matrix.hpp"
#include "constants.hpp"
#include "driver_types.h"

namespace verlet_cuda
{

using namespace edt::lazy_matrix_aliases;  // NOLINT

class DeviceGrid
{
public:
    [[nodiscard]] static constexpr Vec2<size_t> LocationToCell(const Vec2f& location)
    {
        return (constants::kWorldRange.Clamp(location) - constants::kWorldRange.Min()).Cast<size_t>() /
               constants::kGridCellSize;
    }

    [[nodiscard]] static constexpr size_t LocationToCellIndex(const Vec2f& location)
    {
        return CellToCellIndex(LocationToCell(location));
    }

    [[nodiscard]] static constexpr size_t CellToCellIndex(const Vec2<size_t>& cell)
    {
        return cell.x() + cell.y() * constants::kGridSize.x();
    }
};

class GridCell
{
public:
    std::array<uint32_t, constants::kGridMaxObjectsInCell> objects;
    uint32_t num_objects;
};

// C++ interface to invoke cuda kernels
class Kernels
{
public:
    static void ClearGrid(cudaStream_t& stream, GridCell* cells);
    static void PopulateGrid(cudaStream_t& stream, GridCell* cells, size_t num_objects, Vec2f* positions);
    static void
    SolveCollisions_OneRow(cudaStream_t& stream, GridCell* cells, Vec2f* positions, size_t cell_y, size_t offset_x);
    static void
    SolveCollisions_ManyRows(cudaStream_t& stream, GridCell* cells, Vec2f* positions, edt::Vec2<size_t> offset);
    static void UpdatePositions(cudaStream_t& stream, size_t num_objects, Vec2f* positions, Vec2f* old_positions);
};
}  // namespace verlet_cuda
