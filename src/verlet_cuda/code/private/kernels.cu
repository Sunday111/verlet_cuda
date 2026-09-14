#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cuda/atomic>
#include <limits>

#include "kernels.hpp"

namespace verlet::kernels_impl
{

constexpr size_t GetChunkSize(size_t total_amount, size_t num_chunks, size_t chunk_index)
{
    assert(num_chunks > 0);
    assert(chunk_index < num_chunks);

    auto result = total_amount / num_chunks;
    if (auto remainder = total_amount % num_chunks; chunk_index < remainder)
    {
        result += 1;
    }

    return result;
}

static_assert(GetChunkSize(8, 3, 0) == 3);
static_assert(GetChunkSize(8, 3, 1) == 3);
static_assert(GetChunkSize(8, 3, 2) == 2);
static_assert(GetChunkSize(10, 3, 1) == 3);

constexpr edt::Vec2<size_t>
GetChunkSize2D(edt::Vec2<size_t> total_amount, edt::Vec2<size_t> num_chunks, edt::Vec2<size_t> offset)
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

__global__ void PopulateGrid(GridCell* cells, VerletObject* objects, size_t num_objects)
{
    const size_t object_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (object_index >= num_objects) return;

    VerletObject& object = objects[object_index];
    const auto cell_index = GridCell::LocationToCellIndex(object.position);
    object.next_object_in_cell = kInvalidObjectIndex;
    auto* link = &cells[cell_index].first_object_index;
    uint32_t inserting = static_cast<uint32_t>(object_index);
    using AtomicLink = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>;
    while (true)
    {
        const uint32_t previous = AtomicLink{*link}.fetch_min(inserting, cuda::memory_order_acq_rel);
        if (previous == kInvalidObjectIndex) break;
        link = &objects[min(previous, inserting)].next_object_in_cell;
        inserting = max(previous, inserting);
    }
}

template <bool check_for_self_collision = false>
__device__ void SolveCollisionBetweenObjectAndCell(
    const GridCell* cells,
    VerletObject* objects,
    VerletObject& object,
    Vec2f& object_position,
    size_t origin_cell_index)
{
    uint32_t another_object_index = cells[origin_cell_index].first_object_index;  // NOLINT
    while (another_object_index != kInvalidObjectIndex)
    {
        VerletObject& another_object = objects[another_object_index];  // NOLINT
        another_object_index = another_object.next_object_in_cell;

        // Don't need this branch in all nine cases
        // only when colliding object with objects in the same cell
        if constexpr (check_for_self_collision)
        {
            // self-collision
            if (&object == &another_object)
            {
                continue;
            }
        }

        auto& another_object_position = another_object.position;
        const Vec2f axis = object_position - another_object_position;
        const float dist_sq = axis.SquaredLength();
        if (dist_sq < 1.0f)
        {
            const float dist = sqrt(dist_sq);
            const float delta = 0.5f - dist / 2;
            const Vec2f col_vec = dist_sq >= std::numeric_limits<float>::min()
                                      ? axis * (delta / dist)
                                      : Vec2f{&object < &another_object ? -delta : delta, 0.f};
            const auto ac = 0.5f, bc = 0.5f;  // mass coefficients
            object_position += ac * col_vec;
            another_object_position -= bc * col_vec;
        }
    }
}

__device__ void
SolveCollisionsFromCell(Vec2<size_t> cell, size_t grid_width, const GridCell* cells, VerletObject* objects)
{
    const size_t cell_index = cell.y() * grid_width + cell.x();
    uint32_t object_index = cells[cell_index].first_object_index;  // NOLINT
    while (object_index != kInvalidObjectIndex)
    {
        VerletObject& object = objects[object_index];  // NOLINT
        Vec2f object_position = object.position;
        SolveCollisionBetweenObjectAndCell<true>(cells, objects, object, object_position, cell_index);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index + 1);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index - 1);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index + grid_width);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index + grid_width + 1);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index + grid_width - 1);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index - grid_width);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index - grid_width + 1);
        SolveCollisionBetweenObjectAndCell(cells, objects, object, object_position, cell_index - grid_width - 1);

        object.position = object_position;
        object_index = object.next_object_in_cell;
    }
}

template <size_t pass>
__global__ void SolveCollisions_ManyRows(const GridCell* cells, VerletObject* objects)
{
    constexpr Vec2<size_t> offset{pass % 3, pass / 3};
    constexpr auto grid_size = constants::kGridSize;
    constexpr auto sparse_grid_size = GetChunkSize2D(grid_size - 2, {3, 3}, offset);
    const size_t job_index = threadIdx.x + blockIdx.x * blockDim.x;
    const Vec2<size_t> sparse_grid_cell{
        Vec2<size_t>{job_index % sparse_grid_size.x(), job_index / sparse_grid_size.x()}};
    const auto cell = sparse_grid_cell * 3 + offset + 1;
    if (sparse_grid_cell.x() >= sparse_grid_size.x() || sparse_grid_cell.y() >= sparse_grid_size.y()) return;
    SolveCollisionsFromCell(cell, grid_size.x(), cells, objects);
}

__global__ void UpdatePositions(
    size_t num_objects,
    VerletObject* objects,
    Vec2f* previous_positions,
    edt::Vec2f gravity,
    float velocity_damping)
{
    constexpr float margin = 2.0f;
    constexpr auto constraint_with_margin = constants::kWorldRange.Enlarged(-margin);
    constexpr float dt_2 = constants::kTimeSubStepDurationSeconds * constants::kTimeSubStepDurationSeconds;

    const size_t object_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (object_index >= num_objects) return;

    auto& position = objects[object_index].position;
    auto& old_position = previous_positions[object_index];

    const auto last_update_move = position - old_position;

    // Save current position
    old_position = position;

    // Perform Verlet integration
    position += last_update_move + (gravity - last_update_move * velocity_damping) * dt_2;

    // Constraint
    position = constraint_with_margin.Clamp(position);
}
}  // namespace verlet::kernels_impl

namespace verlet
{
namespace
{
constexpr std::array kCollisionKernels{
    kernels_impl::SolveCollisions_ManyRows<0>,
    kernels_impl::SolveCollisions_ManyRows<1>,
    kernels_impl::SolveCollisions_ManyRows<2>,
    kernels_impl::SolveCollisions_ManyRows<3>,
    kernels_impl::SolveCollisions_ManyRows<4>,
    kernels_impl::SolveCollisions_ManyRows<5>,
    kernels_impl::SolveCollisions_ManyRows<6>,
    kernels_impl::SolveCollisions_ManyRows<7>,
    kernels_impl::SolveCollisions_ManyRows<8>};
}

cudaError_t Kernels::PopulateGrid(cudaStream_t& stream, GridCell* cells, VerletObject* objects, size_t num_objects)
{
    if (num_objects == 0) return cudaSuccess;
    const uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_objects) + threads_per_block - 1) / threads_per_block;
    kernels_impl::PopulateGrid<<<num_blocks, threads_per_block, 0, stream>>>(cells, objects, num_objects);
    return cudaGetLastError();
}

cudaError_t
Kernels::SolveCollisions(cudaStream_t& stream, GridCell* cells, VerletObject* objects, edt::Vec2<size_t> offset)
{
    const auto sparse_grid_size = kernels_impl::GetChunkSize2D(constants::kGridSize - 2, {3, 3}, offset);
    const size_t num_jobs = sparse_grid_size.x() * sparse_grid_size.y();
    const uint32_t threads_per_block = 1024;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_jobs) + threads_per_block - 1) / threads_per_block;
    const size_t pass = offset.x() + offset.y() * 3;
    [[assume(pass < kCollisionKernels.size())]];
    kCollisionKernels[pass]<<<num_blocks, threads_per_block, 0, stream>>>(cells, objects);
    return cudaGetLastError();
}

cudaError_t
Kernels::UpdatePositions(cudaStream_t& stream, size_t num_objects, VerletObject* objects, Vec2f* previous_positions)
{
    const uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_objects) + threads_per_block - 1) / threads_per_block;
    kernels_impl::UpdatePositions<<<num_blocks, threads_per_block, 0, stream>>>(
        num_objects,
        objects,
        previous_positions,
        constants::kGravity,
        constants::kVelocityDamping);
    return cudaGetLastError();
}
}  // namespace verlet
