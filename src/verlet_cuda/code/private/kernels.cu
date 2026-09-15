#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <limits>

#include "kernels.hpp"

namespace verlet::kernels_impl
{

constexpr size_t GetChunkSize(size_t total_amount, size_t num_chunks, size_t chunk_index)
{
    assert(num_chunks > 0);
    [[assume(num_chunks > 0)]];
    assert(chunk_index < num_chunks);
    [[assume(chunk_index < num_chunks)]];

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

__device__ void UpdatePosition(Vec2f& position, Vec2f& old_position, Vec2f gravity, float velocity_damping)
{
    constexpr auto constraint_with_margin = constants::kWorldRange.Enlarged(-2.f);
    constexpr float dt_2 = constants::kTimeSubStepDurationSeconds * constants::kTimeSubStepDurationSeconds;
    const auto last_update_move = position - old_position;

    // Save current position
    old_position = position;

    // Perform Verlet integration
    position += last_update_move + (gravity - last_update_move * velocity_damping) * dt_2;

    // Constraint
    position = constraint_with_margin.Clamp(position);
}

template <bool cached = false, bool integrate = false>
__global__ void PopulateGrid(
    GridCell* cells,
    VerletObject* objects,
    size_t num_objects,
    const uint32_t* original_indices = nullptr,
    uint32_t* last_occupied_cell = nullptr,
    Vec2f* previous_positions = nullptr)
{
    const size_t object_index = threadIdx.x + blockIdx.x * blockDim.x;
    if constexpr (integrate)
    {
        if (object_index < num_objects)
            UpdatePosition(
                objects[object_index].position,
                previous_positions[object_index],
                constants::kGravity,
                constants::kVelocityDamping);
    }
    const auto cell_index =
        object_index < num_objects ? GridCell::LocationToCellIndex(objects[object_index].position) : 0;
    if (last_occupied_cell)
    {
        using Reduce = cub::BlockReduce<uint32_t, 256>;
        __shared__ Reduce::TempStorage storage;
        const auto last = Reduce(storage).Reduce(
            static_cast<uint32_t>(cell_index),
            [] __device__(uint32_t a, uint32_t b) { return max(a, b); });
        if (threadIdx.x == 0) atomicMax(last_occupied_cell, last);
    }
    if (object_index >= num_objects) return;

    VerletObject& object = objects[object_index];
    object.next_object_in_cell = kInvalidObjectIndex;
    auto* link = &cells[cell_index].first_object_index;
    uint32_t inserting = static_cast<uint32_t>(object_index);
    using AtomicLink = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>;
    while (true)
    {
        if constexpr (cached)
        {
            AtomicLink atomic_link{*link};
            uint32_t previous = atomic_link.load(cuda::memory_order_acquire);
            if (previous == kInvalidObjectIndex || original_indices[inserting] < original_indices[previous])
            {
                if (!atomic_link.compare_exchange_weak(
                        previous,
                        inserting,
                        cuda::memory_order_acq_rel,
                        cuda::memory_order_acquire))
                    continue;
                if (previous == kInvalidObjectIndex) break;
                link = &objects[inserting].next_object_in_cell;
                inserting = previous;
            }
            else
            {
                link = &objects[previous].next_object_in_cell;
            }
        }
        else
        {
            const uint32_t previous = AtomicLink{*link}.fetch_min(inserting, cuda::memory_order_acq_rel);
            if (previous == kInvalidObjectIndex) break;
            link = &objects[min(previous, inserting)].next_object_in_cell;
            inserting = max(previous, inserting);
        }
    }
}

template <bool check_for_self_collision = false, bool cached = false>
__device__ void SolveCollisionBetweenObjectAndCell(
    const GridCell* cells,
    VerletObject* objects,
    VerletObject& object,
    Vec2f& object_position,
    size_t origin_cell_index,
    const uint32_t* original_indices = nullptr)
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
            const auto precedes = [&]
            {
                if constexpr (cached)
                    return original_indices[&object - objects] < original_indices[&another_object - objects];
                else
                    return &object < &another_object;
            };
            const Vec2f col_vec = dist_sq >= std::numeric_limits<float>::min()
                                      ? axis * (delta / dist)
                                      : Vec2f{precedes() ? -delta : delta, 0.f};
            const auto ac = 0.5f, bc = 0.5f;  // mass coefficients
            object_position += ac * col_vec;
            another_object_position -= bc * col_vec;
        }
    }
}

template <bool cached = false>
__device__ void SolveCollisionsFromCell(
    Vec2<size_t> cell,
    size_t grid_width,
    const GridCell* cells,
    VerletObject* objects,
    const uint32_t* original_indices = nullptr)
{
    const size_t cell_index = cell.y() * grid_width + cell.x();
    uint32_t object_index = cells[cell_index].first_object_index;  // NOLINT
    while (object_index != kInvalidObjectIndex)
    {
        VerletObject& object = objects[object_index];  // NOLINT
        Vec2f object_position = object.position;
        SolveCollisionBetweenObjectAndCell<true, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index + 1,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index - 1,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index + grid_width,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index + grid_width + 1,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index + grid_width - 1,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index - grid_width,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index - grid_width + 1,
            original_indices);
        SolveCollisionBetweenObjectAndCell<false, cached>(
            cells,
            objects,
            object,
            object_position,
            cell_index - grid_width - 1,
            original_indices);

        object.position = object_position;
        object_index = object.next_object_in_cell;
    }
}

template <size_t pass, bool cached>
__global__ void SolveCollisions_ManyRows(
    const GridCell* cells,
    VerletObject* objects,
    const uint32_t* original_indices,
    const uint32_t* last_occupied_cell)
{
    constexpr Vec2<size_t> offset{pass % 3, pass / 3};
    constexpr auto grid_size = constants::kGridSize;
    constexpr auto sparse_grid_size = GetChunkSize2D(grid_size - 2, {3, 3}, offset);
    const size_t job_index = threadIdx.x + blockIdx.x * blockDim.x;
    const Vec2<size_t> sparse_grid_cell{
        Vec2<size_t>{job_index % sparse_grid_size.x(), job_index / sparse_grid_size.x()}};
    const auto cell = sparse_grid_cell * 3 + offset + 1;
    if (sparse_grid_cell.x() >= sparse_grid_size.x() || sparse_grid_cell.y() >= sparse_grid_size.y()) return;
    if (last_occupied_cell && cell.y() * grid_size.x() + cell.x() > *last_occupied_cell) return;
    SolveCollisionsFromCell<cached>(cell, grid_size.x(), cells, objects, original_indices);
}

__global__ void UpdatePositions(
    size_t num_objects,
    VerletObject* objects,
    Vec2f* previous_positions,
    Vec2f gravity,
    float velocity_damping)
{
    const size_t object_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (object_index >= num_objects) return;
    UpdatePosition(objects[object_index].position, previous_positions[object_index], gravity, velocity_damping);
}
__global__ void CacheCells(
    verlet::GridCell* grid,
    const verlet::VerletObject* objects,
    verlet::VerletObject* cached,
    uint32_t* originals,
    uint32_t* allocated,
    const Vec2f* previous_positions,
    Vec2f* cached_previous_positions)
{
    using Scan = cub::BlockScan<uint32_t, 256>;
    __shared__ Scan::TempStorage scan;
    __shared__ uint32_t start;
    const size_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = 0;
    uint32_t first = verlet::kInvalidObjectIndex;
    if (cell < verlet::constants::kGridNumCells) first = grid[cell].first_object_index;
    for (auto index = first; index != verlet::kInvalidObjectIndex; index = objects[index].next_object_in_cell) ++count;
    uint32_t offset, total;
    Scan(scan).ExclusiveSum(count, offset, total);
    if (threadIdx.x == 0) start = total ? atomicAdd(allocated, total) : 0;
    __syncthreads();
    offset += start;
    if (cell >= verlet::constants::kGridNumCells) return;
    grid[cell].first_object_index = count ? offset : verlet::kInvalidObjectIndex;
    for (auto index = first; index != verlet::kInvalidObjectIndex;)
    {
        const auto object = objects[index];
        originals[offset] = index;
        cached_previous_positions[offset] = previous_positions[index];
        cached[offset].position = object.position;
        cached[offset].next_object_in_cell = --count ? offset + 1 : verlet::kInvalidObjectIndex;
        index = object.next_object_in_cell;
        ++offset;
    }
}
template <bool restore_history>
__global__ void ScatterCache(
    const verlet::VerletObject* cached,
    verlet::VerletObject* objects,
    const uint32_t* originals,
    size_t count,
    const Vec2f* cached_previous_positions,
    Vec2f* previous_positions)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        auto& object = objects[originals[index]];
        object.position = cached[index].position;
        if constexpr (restore_history)
        {
            previous_positions[originals[index]] = cached_previous_positions[index];
            const auto next = cached[index].next_object_in_cell;
            object.next_object_in_cell = next == verlet::kInvalidObjectIndex ? next : originals[next];
        }
    }
}

}  // namespace verlet::kernels_impl

namespace verlet
{
namespace
{
template <bool cached>
constexpr std::array kCollisionKernels{
    kernels_impl::SolveCollisions_ManyRows<0, cached>,
    kernels_impl::SolveCollisions_ManyRows<1, cached>,
    kernels_impl::SolveCollisions_ManyRows<2, cached>,
    kernels_impl::SolveCollisions_ManyRows<3, cached>,
    kernels_impl::SolveCollisions_ManyRows<4, cached>,
    kernels_impl::SolveCollisions_ManyRows<5, cached>,
    kernels_impl::SolveCollisions_ManyRows<6, cached>,
    kernels_impl::SolveCollisions_ManyRows<7, cached>,
    kernels_impl::SolveCollisions_ManyRows<8, cached>};
}

namespace
{
template <bool integrate>
cudaError_t LaunchPopulateGrid(
    cudaStream_t& stream,
    GridCell* cells,
    VerletObject* objects,
    size_t num_objects,
    const uint32_t* original_indices,
    uint32_t* last_occupied_cell,
    Vec2f* previous_positions)
{
    if (num_objects == 0) return cudaSuccess;
    if (last_occupied_cell)
        if (const auto result = cudaMemsetAsync(last_occupied_cell, 0, sizeof(uint32_t), stream); result != cudaSuccess)
            return result;
    const uint32_t threads_per_block = 256;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_objects) + threads_per_block - 1) / threads_per_block;
    const auto kernel =
        original_indices ? kernels_impl::PopulateGrid<true, integrate> : kernels_impl::PopulateGrid<false, integrate>;
    kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        cells,
        objects,
        num_objects,
        original_indices,
        last_occupied_cell,
        previous_positions);
    return cudaGetLastError();
}

}  // namespace

cudaError_t Kernels::PopulateGrid(
    cudaStream_t& stream,
    GridCell* cells,
    VerletObject* objects,
    size_t num_objects,
    const uint32_t* original_indices,
    uint32_t* last_occupied_cell)
{
    return LaunchPopulateGrid<false>(
        stream,
        cells,
        objects,
        num_objects,
        original_indices,
        last_occupied_cell,
        nullptr);
}

cudaError_t Kernels::UpdateAndPopulateGrid(
    cudaStream_t& stream,
    GridCell* cells,
    VerletObject* objects,
    size_t num_objects,
    Vec2f* previous_positions,
    const uint32_t* original_indices,
    uint32_t* last_occupied_cell)
{
    return LaunchPopulateGrid<true>(
        stream,
        cells,
        objects,
        num_objects,
        original_indices,
        last_occupied_cell,
        previous_positions);
}

cudaError_t Kernels::SolveCollisions(
    cudaStream_t& stream,
    GridCell* cells,
    VerletObject* objects,
    edt::Vec2<size_t> offset,
    const uint32_t* original_indices,
    const uint32_t* last_occupied_cell)
{
    const auto sparse_grid_size = kernels_impl::GetChunkSize2D(constants::kGridSize - 2, {3, 3}, offset);
    const size_t num_jobs = sparse_grid_size.x() * sparse_grid_size.y();
    const uint32_t threads_per_block = 1024;
    const uint32_t num_blocks = (static_cast<uint32_t>(num_jobs) + threads_per_block - 1) / threads_per_block;
    const size_t pass = offset.x() + offset.y() * 3;
    assert(pass < kCollisionKernels<false>.size());
    [[assume(pass < kCollisionKernels<false>.size())]];
    const auto kernel = original_indices ? kCollisionKernels<true>[pass] : kCollisionKernels<false>[pass];
    kernel<<<num_blocks, threads_per_block, 0, stream>>>(cells, objects, original_indices, last_occupied_cell);
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
cudaError_t Kernels::CacheGrid(
    cudaStream_t stream,
    GridCell* cells,
    const VerletObject* objects,
    const Vec2f* previous_positions,
    CollisionCache cache)
{
    if (const auto result = cudaMemsetAsync(&cache.metadata->object_count, 0, sizeof(uint32_t), stream);
        result != cudaSuccess)
        return result;
    constexpr uint32_t threads_per_block = 256;
    constexpr auto num_blocks = (constants::kGridNumCells + threads_per_block - 1) / threads_per_block;
    kernels_impl::CacheCells<<<num_blocks, threads_per_block, 0, stream>>>(
        cells,
        objects,
        cache.objects,
        cache.original_indices,
        &cache.metadata->object_count,
        previous_positions,
        cache.previous_positions);
    return cudaGetLastError();
}

cudaError_t Kernels::RestorePositions(
    cudaStream_t stream,
    size_t num_objects,
    VerletObject* objects,
    Vec2f* previous_positions,
    CollisionCache cache)
{
    if (num_objects == 0) return cudaSuccess;
    constexpr uint32_t threads_per_block = 256;
    const auto num_blocks = (num_objects + threads_per_block - 1) / threads_per_block;
    const auto kernel = previous_positions ? kernels_impl::ScatterCache<true> : kernels_impl::ScatterCache<false>;
    kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        cache.objects,
        objects,
        cache.original_indices,
        num_objects,
        cache.previous_positions,
        previous_positions);
    return cudaGetLastError();
}
}  // namespace verlet
