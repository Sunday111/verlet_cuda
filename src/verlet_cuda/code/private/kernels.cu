#include "kernels.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include "EverydayTools/Math/Math.hpp"

namespace verlet::kernels_impl
{

constexpr size_t GetChunkSize(size_t total_amount, size_t num_chunks, size_t chunk_index)
{
    assert(num_chunks > 0);
    assert(chunk_index < num_chunks);

    auto result = total_amount / num_chunks; 
    if (auto remainder = total_amount % num_chunks; chunk_index < remainder)
    {
        result += 1;;
    }

    return result;
}

static_assert(GetChunkSize(8, 3, 0) == 3);
static_assert(GetChunkSize(8, 3, 1) == 3);
static_assert(GetChunkSize(8, 3, 2) == 2);
static_assert(GetChunkSize(10, 3, 1) == 3);

constexpr edt::Vec2<size_t> GetChunkSize2D(edt::Vec2<size_t> total_amount, edt::Vec2<size_t> num_chunks, edt::Vec2<size_t> offset)
{
    return {
        GetChunkSize(total_amount.x(), num_chunks.x(), offset.x()),
        GetChunkSize(total_amount.y(), num_chunks.y(), offset.y()),
    };
}

static_assert(GetChunkSize2D({600, 600}, {3, 3}, {0, 0}) == Vec2<size_t>{200, 200});
static_assert(GetChunkSize2D({600, 600}, {3, 3}, {1, 1}) == Vec2<size_t>{200, 200});
static_assert(GetChunkSize2D({600, 600}, {3, 3}, {2, 2}) == Vec2<size_t>{200, 200});
static_assert(GetChunkSize2D({601, 601}, {3, 3}, {0, 0}) == Vec2<size_t>{201, 201});
static_assert(GetChunkSize2D({601, 601}, {3, 3}, {1, 1}) == Vec2<size_t>{200, 200});
static_assert(GetChunkSize2D({601, 601}, {3, 3}, {2, 2}) == Vec2<size_t>{200, 200});
static_assert(GetChunkSize2D({602, 602}, {3, 3}, {0, 0}) == Vec2<size_t>{201, 201});
static_assert(GetChunkSize2D({602, 602}, {3, 3}, {1, 1}) == Vec2<size_t>{201, 201});
static_assert(GetChunkSize2D({602, 602}, {3, 3}, {2, 2}) == Vec2<size_t>{200, 200});

__global__ void PopulateGrid(GridCell* cells, size_t num_objects, VerletObject* objects)
{
    const size_t object_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (object_index >= num_objects) return;

    edt::Vec2f& position = objects[object_index].position;
    const auto cell_index = GridCell::LocationToCellIndex(position);
    auto idx = atomicAdd(&cells[cell_index].num_objects, 1);
    if (idx < 4)
    {
        cells[cell_index].objects[idx] = object_index;
    }
    else
    {
        cells[cell_index].num_objects = 4;
        position *= 0.9999f;
    }
}

__device__ void SolveCollisionsFromCell(Vec2<size_t> cell, const GridCell* cells, VerletObject* objects)
{
    const size_t cell_index = cell.y() * constants::kGridSize.x() + cell.x();
    // printf("Solve collisions on cell %d %d\n", (int)cell.x(), (int)cell.y());

    constexpr float eps = 0.0001f;
    auto solve_collision_between_object_and_cell =
        [&](const size_t object_index, edt::Vec2f& object_position, const size_t origin_cell_index)
        {
            const size_t objects_in_cell = cells[origin_cell_index].num_objects % 4;
            for (size_t index_in_cell = 0; index_in_cell != objects_in_cell; ++index_in_cell)
            {
                const size_t another_object_index = cells[origin_cell_index].objects[index_in_cell];
                if (object_index != another_object_index)
                {
                    auto& another_object_position = objects[another_object_index].position;
                    const Vec2f axis = object_position - another_object_position;
                    const float dist_sq = axis.SquaredLength();
                    if (dist_sq < 1.0f && dist_sq > eps)
                    {
                        const float dist = sqrt(dist_sq);
                        const float delta = 0.5f - dist / 2;
                        const Vec2f col_vec = axis * (delta / dist);
                        const auto ac = 0.5f, bc = 0.5f; // mass coefficients
                        object_position += ac * col_vec;
                        another_object_position -= bc * col_vec;
                    }
                }
            }
        };

    const size_t grid_width = constants::kGridSize.x();
    const size_t objects_in_cell = cells[cell_index].num_objects % 4;
    for (size_t index_in_cell = 0; index_in_cell != objects_in_cell; ++index_in_cell)
    {
        size_t object_index = cells[cell_index].objects[index_in_cell];
        auto& object_position = objects[object_index].position;
        solve_collision_between_object_and_cell(object_index, object_position, cell_index);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index + 1);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index - 1);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index + grid_width);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index + grid_width + 1);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index + grid_width - 1);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index - grid_width);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index - grid_width + 1);
        solve_collision_between_object_and_cell(object_index, object_position, cell_index - grid_width - 1);
    }
}

__global__ void SolveCollisions_ManyRows(edt::Vec2<size_t> offset, const GridCell* cells, VerletObject* objects)
{
    const size_t job_index = threadIdx.x + blockIdx.x * blockDim.x;
    const auto sparse_grid_size = kernels_impl::GetChunkSize2D(constants::kGridSize - 2, {3, 3}, offset);
    const Vec2<size_t> sparse_grid_cell {Vec2<size_t>{ job_index % sparse_grid_size.x(), job_index / sparse_grid_size.x() }};
    const auto cell = sparse_grid_cell * 3 + offset + 1;
    if (sparse_grid_cell.x() >= sparse_grid_size.x() || sparse_grid_cell.y() >= sparse_grid_size.y()) return;
    SolveCollisionsFromCell(cell, cells, objects);
}

__global__ void UpdatePositions(size_t num_objects, VerletObject* objects)
{
    constexpr float margin = 2.0f;
    constexpr auto constraint_with_margin = constants::kWorldRange.Enlarged(-margin);
    constexpr float dt_2 = edt::Math::Sqr(constants::kTimeSubStepDurationSeconds);

    const size_t object_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (object_index >= num_objects) return;

    auto& position = objects[object_index].position;
    auto& old_position = objects[object_index].old_position;

    const auto last_update_move = position - old_position;

    // Save current position
    old_position = position;

    // Perform Verlet integration
    position += last_update_move + (constants::gravity - last_update_move * constants::kVelocityDampling) * dt_2;

    // Constraint
    position = constraint_with_margin.Clamp(position);
}
}

namespace verlet
{

void Kernels::PopulateGrid(cudaStream_t& stream, GridCell* cells, size_t num_objects, VerletObject* objects)
{
    const uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_objects) + threads_per_block - 1) / threads_per_block;
    kernels_impl::PopulateGrid<<<num_blocks, threads_per_block, 0, stream>>> (cells, num_objects, objects);
    const cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
}

void Kernels::SolveCollisions(cudaStream_t& stream, GridCell* cells, VerletObject* objects, edt::Vec2<size_t> offset)
{
    const auto sparse_grid_size = kernels_impl::GetChunkSize2D(constants::kGridSize - 2, { 3, 3 }, offset);
    const size_t num_jobs = sparse_grid_size.x() * sparse_grid_size.y();
    const uint32_t threads_per_block = 1024;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_jobs) + threads_per_block - 1) / threads_per_block;
    kernels_impl::SolveCollisions_ManyRows<<<num_blocks, threads_per_block, 0, stream>>>(offset, cells, objects);
    const cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
}

void Kernels::UpdatePositions(cudaStream_t& stream, size_t num_objects, VerletObject* objects)
{
    const uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_objects) + threads_per_block - 1) / threads_per_block;
    kernels_impl::UpdatePositions<<<num_blocks, threads_per_block, 0, stream>>>(num_objects, objects);
    const cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
}
}
