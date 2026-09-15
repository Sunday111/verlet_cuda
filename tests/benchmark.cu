#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <random>
#include <string_view>
#include <vector>

#include "../src/verlet_cuda/code/private/emitters/burst_layout.hpp"
#include "../src/verlet_cuda/code/private/kernels.cu"

namespace
{
void Check(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        std::println(stderr, "{}: {}", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}
}  // namespace

int main(int argc, char** argv)
{
    const auto usage = []
    {
        std::println(
            stderr,
            "Usage: benchmark 100000|1000000|1500000|2000000|3000000|4000000 "
            "sparse|dense|lattice|coincident|packed|burst "
            "grid|sweep|frame|evolve "
            "[samples] [direct|cached]");
        return EXIT_FAILURE;
    };
    if (argc < 4 || argc > 6) return usage();
    const auto parse = [](std::string_view argument, size_t& value)
    {
        const auto result = std::from_chars(argument.data(), argument.data() + argument.size(), value);
        return result.ec == std::errc{} && result.ptr == argument.data() + argument.size();
    };
    size_t count = 0;
    size_t samples = 50;
    const std::string_view scene{argv[2]}, mode{argv[3]};
    if (!parse(argv[1], count) || (count != 100000 && count != 1000000 && count != 1500000 && count != 2000000 &&
                                   count != 3000000 && count != 4000000))
        return usage();
    if (scene != "sparse" && scene != "dense" && scene != "lattice" && scene != "coincident" && scene != "packed" &&
        scene != "burst")
        return usage();
    if (mode != "grid" && mode != "sweep" && mode != "frame" && mode != "evolve") return usage();
    if (argc >= 5 && (!parse(argv[4], samples) || samples == 0 || samples > 100000)) return usage();

    const std::string_view backend =
        argc == 6 ? argv[5] : (count >= verlet::constants::kCollisionCacheMinObjects ? "cached" : "direct");
    if (backend != "direct" && backend != "cached") return usage();
    const bool use_cache = backend == "cached";

    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0), "Get CUDA device properties");
    Check(cudaSetDevice(0), "Select CUDA device");
    std::println(stderr, "GPU: {}; {} particles; {}; {}", properties.name, count, scene, mode);
    std::println(stderr, "Collision backend: {}", backend);

    std::vector<verlet::VerletObject> input(count);
    std::vector<verlet::GridCell> grid(verlet::constants::kGridNumCells);
    std::mt19937 random{1234};
    const auto uniform = [&]
    {
        return static_cast<float>(random() >> 8) * (1.f / 16777216.f);
    };
    const float width = scene == "dense" ? std::sqrt(static_cast<float>(count) / 1.5f * 1.83f) : 1900.f;
    const float height = scene == "dense" ? static_cast<float>(count) / (1.5f * width) : 1010.f;
    const size_t locations = scene == "coincident" ? count / 2 : count;
    constexpr auto centre_bounds = verlet::constants::kWorldRange.Enlarged(-2.f);
    constexpr float packed_spacing = 1.01f;
    const auto columns = scene == "packed"
                             ? static_cast<size_t>(centre_bounds.Extent().x() / packed_spacing)
                             : static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(locations) * 1.83)));
    const size_t rows = (locations + columns - 1) / columns;
    const float spacing =
        scene == "packed" ? packed_spacing
                          : std::min({1.1f, width / static_cast<float>(columns), height / static_cast<float>(rows)});
    const float row_spacing = std::sqrt(3.f) * 0.5f * packed_spacing;
    if (scene == "packed" && (static_cast<float>(rows - 1) * row_spacing > centre_bounds.Extent().y()))
    {
        std::println(stderr, "Packed population does not fit in the world");
        return EXIT_FAILURE;
    }
    const auto burst =
        scene == "burst" ? verlet::BurstLayout::Generate(count, true, true) : std::vector<verlet::Vec2f>{};
    if (scene == "burst" && burst.size() != count)
    {
        std::println(stderr, "Packed burst does not fit in the world");
        return EXIT_FAILURE;
    }
    for (size_t index = 0; index < count; ++index)
    {
        if (scene == "burst")
        {
            input[index].position = burst[index];
        }
        else if (scene == "packed")
        {
            const size_t row = index / columns;
            input[index].position = {
                (static_cast<float>(index % columns) - static_cast<float>(columns - 1) * 0.5f +
                 static_cast<float>(row & 1) * 0.5f) *
                    spacing,
                centre_bounds.Min().y() + static_cast<float>(row) * row_spacing};
            if (input[index].position != centre_bounds.Clamp(input[index].position))
            {
                std::println(stderr, "Packed particle {} is outside the centre bounds", index);
                return EXIT_FAILURE;
            }
            const auto check_neighbour = [&](size_t neighbour)
            {
                if ((input[index].position - input[neighbour].position).SquaredLength() < 1.f)
                {
                    std::println(stderr, "Packed particles {} and {} overlap", index, neighbour);
                    std::exit(EXIT_FAILURE);
                }
            };
            const size_t column = index % columns;
            if (column != 0) check_neighbour(index - 1);
            if (row != 0)
            {
                check_neighbour(index - columns);
                if ((row & 1) != 0 && column + 1 < columns) check_neighbour(index - columns + 1);
                if ((row & 1) == 0 && column != 0) check_neighbour(index - columns - 1);
            }
        }
        else if (scene == "lattice" || scene == "coincident")
        {
            const size_t location = scene == "coincident" ? index / 2 : index;
            input[index].position = {
                (static_cast<float>(location % columns) - static_cast<float>(columns - 1) * 0.5f) * spacing,
                (static_cast<float>(location / columns) - static_cast<float>(rows - 1) * 0.5f) * spacing};
        }
        else
        {
            input[index].position = {(uniform() - 0.5f) * width, (uniform() - 0.5f) * height};
        }
    }
    if (scene == "packed")
        std::println(
            stderr,
            "Packed layout: {} columns, {} rows, {} spacing; checked bounds and neighbouring pairs",
            columns,
            rows,
            packed_spacing);
    std::println(
        stderr,
        "Physics world: {} x {}",
        verlet::constants::kWorldRange.Extent().x(),
        verlet::constants::kWorldRange.Extent().y());
    for (size_t index = count; index-- > 0;)
    {
        auto& cell = grid[verlet::GridCell::LocationToCellIndex(input[index].position)];
        input[index].next_object_in_cell = cell.first_object_index;
        cell.first_object_index = static_cast<uint32_t>(index);
    }

    verlet::VerletObject *objects = nullptr, *seed_objects = nullptr;
    verlet::GridCell *cells = nullptr, *seed_cells = nullptr;
    const size_t object_bytes = count * sizeof(*objects), grid_bytes = grid.size() * sizeof(*cells);
    Check(cudaMalloc(&objects, object_bytes), "Allocate objects");
    Check(cudaMalloc(&seed_objects, object_bytes), "Allocate seed objects");
    Check(cudaMalloc(&cells, grid_bytes), "Allocate grid");
    Check(cudaMalloc(&seed_cells, grid_bytes), "Allocate seed grid");
    Check(cudaMemcpy(seed_objects, input.data(), object_bytes, cudaMemcpyHostToDevice), "Upload seed objects");
    Check(cudaMemcpy(seed_cells, grid.data(), grid_bytes, cudaMemcpyHostToDevice), "Upload seed grid");
    std::vector<verlet::Vec2f> previous(count);
    for (size_t index = 0; index < count; ++index) previous[index] = input[index].position;
    verlet::Vec2f *previous_positions = nullptr, *seed_previous_positions = nullptr;
    const size_t previous_bytes = count * sizeof(verlet::Vec2f);
    Check(cudaMalloc(&previous_positions, previous_bytes), "Allocate previous positions");
    Check(cudaMalloc(&seed_previous_positions, previous_bytes), "Allocate seed previous positions");
    Check(
        cudaMemcpy(seed_previous_positions, previous.data(), previous_bytes, cudaMemcpyHostToDevice),
        "Upload previous positions");
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreate(&stream), "Create stream");
    verlet::CollisionCache cache;
    if (use_cache)
    {
        Check(cudaMalloc(&cache.previous_positions, previous_bytes), "Allocate cached previous positions");
        Check(cudaMalloc(&cache.positions, previous_bytes), "Allocate collision objects");
        Check(cudaMalloc(&cache.links, count * sizeof(uint32_t)), "Allocate cached links");
        Check(cudaMalloc(&cache.original_indices, count * sizeof(uint32_t)), "Allocate collision indices");
        Check(cudaMalloc(&cache.metadata, sizeof(verlet::CollisionCacheMetadata)), "Allocate collision counter");
    }
    const auto build_cache = [&]
    {
        if (use_cache)
            Check(verlet::Kernels::CacheGrid(stream, cells, objects, previous_positions, cache), "Cache grid");
    };
    const auto restore_positions = [&]
    {
        if (use_cache)
            Check(
                verlet::Kernels::RestorePositions(stream, count, objects, previous_positions, cache),
                "Restore positions");
    };
    size_t cache_frame = 0;
    const auto reset = [&]
    {
        cache_frame = 0;
        Check(
            cudaMemcpyAsync(
                previous_positions,
                seed_previous_positions,
                previous_bytes,
                cudaMemcpyDeviceToDevice,
                stream),
            "Reset previous positions");
        Check(cudaMemcpyAsync(objects, seed_objects, object_bytes, cudaMemcpyDeviceToDevice, stream), "Reset objects");
        Check(cudaMemcpyAsync(cells, seed_cells, grid_bytes, cudaMemcpyDeviceToDevice, stream), "Reset grid");
    };
    const auto sweep = [&]
    {
        for (size_t y = 0; y < 3; ++y)
        {
            for (size_t x = 0; x < 3; ++x)
            {
                Check(
                    verlet::Kernels::SolveCollisions(
                        stream,
                        cells,
                        use_cache ? cache.GetObjects() : objects,
                        {x, y},
                        use_cache && mode != "sweep" ? &cache.metadata->last_occupied_cell : nullptr),
                    "Launch collision sweep");
            }
        }
    };
    const auto run = [&]
    {
        if (mode == "grid")
        {
            Check(cudaMemsetAsync(cells, 255, grid_bytes, stream), "Clear grid");
            Check(verlet::Kernels::PopulateGrid(stream, cells, objects, count), "Launch grid population");
            return;
        }
        if (mode == "sweep")
        {
            build_cache();
            sweep();
            restore_positions();
            return;
        }
        for (size_t substep = 0; substep < verlet::constants::kNumSubSteps; ++substep)
        {
            Check(cudaMemsetAsync(cells, 255, grid_bytes, stream), "Clear grid");
            if (substep == 0)
            {
                Check(
                    verlet::Kernels::PopulateGrid(
                        stream,
                        cells,
                        use_cache && cache_frame != 0 ? cache.GetObjects() : objects,
                        count,
                        use_cache ? &cache.metadata->last_occupied_cell : nullptr),
                    "Launch grid population");
            }
            else
            {
                Check(
                    verlet::Kernels::UpdateAndPopulateGrid(
                        stream,
                        cells,
                        use_cache ? cache.GetObjects() : objects,
                        count,
                        use_cache ? cache.previous_positions : previous_positions,
                        use_cache ? &cache.metadata->last_occupied_cell : nullptr),
                    "Integrate and populate grid");
            }
            if (substep == 0 && cache_frame == 0) build_cache();
            sweep();
        }
        Check(
            verlet::Kernels::UpdatePositions(
                stream,
                count,
                use_cache ? cache.GetObjects() : objects,
                use_cache ? cache.previous_positions : previous_positions),
            "Launch position update");
        cache_frame = (cache_frame + 1) % verlet::constants::kCollisionCacheFrames;
        if (use_cache)
            Check(
                verlet::Kernels::RestorePositions(
                    stream,
                    count,
                    objects,
                    cache_frame == 0 ? previous_positions : nullptr,
                    cache),
                "Restore render positions");
    };

    const auto warmup_start = std::chrono::steady_clock::now();
    do
    {
        reset();
        run();
        Check(cudaStreamSynchronize(stream), "Synchronise warmup");
    } while (std::chrono::steady_clock::now() - warmup_start < std::chrono::seconds(2));

    if (mode == "evolve")
    {
        reset();
        for (size_t frame = 0; frame < 120; ++frame) run();
        Check(cudaStreamSynchronize(stream), "Synchronise settling frames");
    }

    cudaEvent_t start = nullptr, end = nullptr;
    Check(cudaEventCreate(&start), "Create start event");
    Check(cudaEventCreate(&end), "Create end event");
    std::println("sample,mode,scene,count,ms");
    for (size_t sample = 0; sample < samples; ++sample)
    {
        if (mode != "evolve") reset();
        Check(cudaEventRecord(start, stream), "Record start event");
        run();
        Check(cudaEventRecord(end, stream), "Record end event");
        Check(cudaEventSynchronize(end), "Synchronise measured work");
        float milliseconds = 0.f;
        Check(cudaEventElapsedTime(&milliseconds, start, end), "Read elapsed time");
        std::println("{},{},{},{},{:.8f}", sample, mode, scene, count, milliseconds);
    }

    if (mode != "grid") restore_positions();
    Check(cudaMemcpy(input.data(), objects, object_bytes, cudaMemcpyDeviceToHost), "Read final positions");
    for (size_t index = 0; index < count; ++index)
    {
        const auto& position = input[index].position;
        if (!std::isfinite(position.x()) || !std::isfinite(position.y()) ||
            position != verlet::constants::kWorldRange.Clamp(position))
        {
            std::println(stderr, "Non-finite or out-of-bounds final position at particle {}", index);
            return EXIT_FAILURE;
        }
    }
    Check(
        cudaMemcpy(previous.data(), previous_positions, previous_bytes, cudaMemcpyDeviceToHost),
        "Read previous positions");
    uint64_t state_hash = 14695981039346656037ull;
    for (size_t index = 0; index < count; ++index)
    {
        for (uint32_t value :
             {std::bit_cast<uint32_t>(input[index].position.x()),
              std::bit_cast<uint32_t>(input[index].position.y()),
              input[index].next_object_in_cell,
              std::bit_cast<uint32_t>(previous[index].x()),
              std::bit_cast<uint32_t>(previous[index].y())})
        {
            state_hash ^= value;
            state_hash *= 1099511628211ull;
        }
    }
    std::println(stderr, "Final state hash: {:016x}", state_hash);
    Check(cudaEventDestroy(start), "Destroy start event");
    Check(cudaEventDestroy(end), "Destroy end event");
    Check(cudaStreamDestroy(stream), "Destroy stream");
    Check(cudaFree(previous_positions), "Free previous positions");
    Check(cudaFree(seed_previous_positions), "Free seed previous positions");
    Check(cudaFree(cache.previous_positions), "Free cached previous positions");
    Check(cudaFree(cache.links), "Free cached links");
    Check(cudaFree(cache.positions), "Free collision objects");
    Check(cudaFree(cache.original_indices), "Free collision indices");
    Check(cudaFree(cache.metadata), "Free collision counter");
    Check(cudaFree(objects), "Free objects");
    Check(cudaFree(seed_objects), "Free seed objects");
    Check(cudaFree(cells), "Free grid");
    Check(cudaFree(seed_cells), "Free seed grid");
    return EXIT_SUCCESS;
}
