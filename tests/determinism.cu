#include <algorithm>
#include <bit>
#include <cstdlib>
#include <print>
#include <random>
#include <vector>

#include "../src/verlet_cuda/code/private/kernels.cu"

namespace
{
void Check(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        std::println(stderr, "CUDA: {}", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

bool SameState(const verlet::VerletObject& left, const verlet::VerletObject& right)
{
    return std::bit_cast<uint32_t>(left.position.x()) == std::bit_cast<uint32_t>(right.position.x()) &&
           std::bit_cast<uint32_t>(left.position.y()) == std::bit_cast<uint32_t>(right.position.y()) &&
           left.next_object_in_cell == right.next_object_in_cell;
}
}  // namespace

int main()
{
    std::mt19937 random{1234};
    constexpr size_t count = 4096;
    std::vector<verlet::VerletObject> initial(count), reference(count), actual(count);
    std::vector<verlet::Vec2f> previous(count), reference_previous(count), actual_previous(count);
    for (size_t index = 0; index < count; ++index)
    {
        initial[index].position = {
            static_cast<float>(index % 64) * 0.7f - 22.f,
            static_cast<float>(index / 64) * 0.7f - 22.f};
        if (index < 32) initial[index].position = {0.f, 0.f};
        previous[index] = initial[index].position + verlet::Vec2f{0.001f, -0.002f};
    }
    verlet::VerletObject* objects = nullptr;
    verlet::Vec2f* old_positions = nullptr;
    verlet::GridCell* cells = nullptr;
    const size_t object_bytes = count * sizeof(*objects), previous_bytes = count * sizeof(*old_positions);
    const size_t grid_bytes = verlet::constants::kGridNumCells * sizeof(*cells);
    Check(cudaMalloc(&objects, object_bytes));
    Check(cudaMalloc(&old_positions, previous_bytes));
    Check(cudaMalloc(&cells, grid_bytes));
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreate(&stream));
    std::vector<verlet::GridCell> grid(verlet::constants::kGridNumCells);
    std::vector<verlet::GridCell> expected_grid(grid.size());
    for (size_t occupied_cells : {1, 2, 31, 256, 1024, 4096})
    {
        for (size_t trial = 0; trial < 4; ++trial)
        {
            for (size_t index = 0; index < count; ++index)
            {
                const auto cell = random() % occupied_cells;
                actual[index].position = {static_cast<float>(cell % 64) - 32.f, static_cast<float>(cell / 64) - 32.f};
                if (occupied_cells > 1 && index < 4)
                    actual[index].position = {
                        (index & 1) ? verlet::constants::kWorldRange.Max().x()
                                    : verlet::constants::kWorldRange.Min().x(),
                        (index & 2) ? verlet::constants::kWorldRange.Max().y()
                                    : verlet::constants::kWorldRange.Min().y()};
                actual[index].next_object_in_cell = static_cast<uint32_t>((index + 7) % count);
            }
            Check(cudaMemcpyAsync(objects, actual.data(), object_bytes, cudaMemcpyHostToDevice, stream));
            Check(cudaMemsetAsync(cells, 255, grid_bytes, stream));
            std::ranges::fill(expected_grid, verlet::GridCell{});
            reference = actual;
            const size_t active_count = count - trial;
            for (size_t index = active_count; index-- > 0;)
            {
                auto& cell = expected_grid[verlet::GridCell::LocationToCellIndex(reference[index].position)];
                reference[index].next_object_in_cell = cell.first_object_index;
                cell.first_object_index = static_cast<uint32_t>(index);
            }
            if (trial == 0)
            {
                Check(verlet::Kernels::PopulateGrid(stream, cells, objects, active_count));
            }
            else
            {
                const unsigned threads = 32u << trial;
                const auto blocks = static_cast<unsigned>((active_count + threads - 1) / threads);
                verlet::kernels_impl::PopulateGrid<<<blocks, threads, 0, stream>>>(cells, objects, active_count);
                Check(cudaGetLastError());
            }
            Check(cudaMemcpyAsync(actual.data(), objects, object_bytes, cudaMemcpyDeviceToHost, stream));
            Check(cudaMemcpyAsync(grid.data(), cells, grid_bytes, cudaMemcpyDeviceToHost, stream));
            Check(cudaStreamSynchronize(stream));
            if (!std::ranges::equal(actual, reference, SameState) ||
                !std::ranges::equal(
                    grid,
                    expected_grid,
                    [](const auto& left, const auto& right)
                    { return left.first_object_index == right.first_object_index; }))
            {
                std::println(stderr, "Grid mismatch with {} occupied cells, trial {}", occupied_cells, trial);
                return EXIT_FAILURE;
            }
        }
    }
    std::println("24 GPU grid cases matched CPU-built lists, including 4,096 particles in one cell");
    for (size_t trial = 0; trial < 8; ++trial)
    {
        Check(cudaMemcpyAsync(objects, initial.data(), object_bytes, cudaMemcpyHostToDevice, stream));
        Check(cudaMemcpyAsync(old_positions, previous.data(), previous_bytes, cudaMemcpyHostToDevice, stream));
        for (size_t step = 0; step < 32; ++step)
        {
            Check(cudaMemsetAsync(cells, 255, grid_bytes, stream));
            if (trial == 0)
            {
                Check(cudaMemcpyAsync(actual.data(), objects, object_bytes, cudaMemcpyDeviceToHost, stream));
                Check(cudaStreamSynchronize(stream));
                std::ranges::fill(grid, verlet::GridCell{});
                for (size_t index = count; index-- > 0;)
                {
                    auto& cell = grid[verlet::GridCell::LocationToCellIndex(actual[index].position)];
                    actual[index].next_object_in_cell = cell.first_object_index;
                    cell.first_object_index = static_cast<uint32_t>(index);
                }
                Check(cudaMemcpyAsync(objects, actual.data(), object_bytes, cudaMemcpyHostToDevice, stream));
                Check(cudaMemcpyAsync(cells, grid.data(), grid_bytes, cudaMemcpyHostToDevice, stream));
            }
            else
            {
                Check(verlet::Kernels::PopulateGrid(stream, cells, objects, count));
            }
            for (size_t y = 0; y < 3; ++y)
                for (size_t x = 0; x < 3; ++x) Check(verlet::Kernels::SolveCollisions(stream, cells, objects, {x, y}));
            Check(verlet::Kernels::UpdatePositions(stream, count, objects, old_positions));
        }
        Check(cudaMemcpyAsync(actual.data(), objects, object_bytes, cudaMemcpyDeviceToHost, stream));
        Check(cudaMemcpyAsync(actual_previous.data(), old_positions, previous_bytes, cudaMemcpyDeviceToHost, stream));
        Check(cudaStreamSynchronize(stream));
        if (trial == 0)
        {
            reference = actual;
            reference_previous = actual_previous;
        }
        else
        {
            for (size_t index = 0; index < count; ++index)
            {
                if (!SameState(actual[index], reference[index]) ||
                    !SameState({.position = actual_previous[index]}, {.position = reference_previous[index]}))
                {
                    std::println(stderr, "State mismatch in trial {}, particle {}", trial, index);
                    return EXIT_FAILURE;
                }
            }
        }
    }
    Check(cudaStreamDestroy(stream));
    Check(cudaFree(cells));
    Check(cudaFree(old_positions));
    Check(cudaFree(objects));
    std::println("Seven GPU replays matched the CPU-built grid reference bit for bit over 32 substeps");
}
