#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <random>
#include <string_view>
#include <vector>

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
            "Usage: benchmark 100000|1000000|1500000|2000000 sparse|dense|lattice|coincident sweep|frame [samples]");
        return EXIT_FAILURE;
    };
    if (argc < 4 || argc > 5) return usage();
    const auto parse = [](std::string_view argument, size_t& value)
    {
        const auto result = std::from_chars(argument.data(), argument.data() + argument.size(), value);
        return result.ec == std::errc{} && result.ptr == argument.data() + argument.size();
    };
    size_t count = 0;
    size_t samples = 50;
    const std::string_view scene{argv[2]}, mode{argv[3]};
    if (!parse(argv[1], count) || (count != 100000 && count != 1000000 && count != 1500000 && count != 2000000))
        return usage();
    if (scene != "sparse" && scene != "dense" && scene != "lattice" && scene != "coincident") return usage();
    if (mode != "sweep" && mode != "frame") return usage();
    if (argc == 5 && (!parse(argv[4], samples) || samples == 0 || samples > 100000)) return usage();

    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0), "Get CUDA device properties");
    Check(cudaSetDevice(0), "Select CUDA device");
    std::println(stderr, "GPU: {}; {} particles; {}; {}", properties.name, count, scene, mode);

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
    const auto columns = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(locations) * 1.83)));
    const size_t rows = (locations + columns - 1) / columns;
    const float spacing = std::min({1.1f, width / static_cast<float>(columns), height / static_cast<float>(rows)});
    for (size_t index = 0; index < count; ++index)
    {
        if (scene == "lattice" || scene == "coincident")
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
        input[index].old_position = input[index].position;
    }
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
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreate(&stream), "Create stream");
    const auto reset = [&]
    {
        Check(cudaMemcpyAsync(objects, seed_objects, object_bytes, cudaMemcpyDeviceToDevice, stream), "Reset objects");
        Check(cudaMemcpyAsync(cells, seed_cells, grid_bytes, cudaMemcpyDeviceToDevice, stream), "Reset grid");
    };
    const auto sweep = [&]
    {
        for (size_t y = 0; y < 3; ++y)
        {
            for (size_t x = 0; x < 3; ++x)
            {
                Check(verlet::Kernels::SolveCollisions(stream, cells, objects, {x, y}), "Launch collision sweep");
            }
        }
    };
    const auto run = [&]
    {
        if (mode == "sweep")
        {
            sweep();
            return;
        }
        for (size_t substep = 0; substep < verlet::constants::kNumSubSteps; ++substep)
        {
            Check(cudaMemsetAsync(cells, 255, grid_bytes, stream), "Clear grid");
            Check(verlet::Kernels::PopulateGrid(stream, cells, objects, count), "Launch grid population");
            sweep();
            Check(verlet::Kernels::UpdatePositions(stream, count, objects), "Launch position update");
        }
    };

    const auto warmup_start = std::chrono::steady_clock::now();
    do
    {
        reset();
        run();
        Check(cudaStreamSynchronize(stream), "Synchronise warmup");
    } while (std::chrono::steady_clock::now() - warmup_start < std::chrono::seconds(2));

    cudaEvent_t start = nullptr, end = nullptr;
    Check(cudaEventCreate(&start), "Create start event");
    Check(cudaEventCreate(&end), "Create end event");
    std::println("sample,mode,scene,count,ms");
    for (size_t sample = 0; sample < samples; ++sample)
    {
        reset();
        Check(cudaEventRecord(start, stream), "Record start event");
        run();
        Check(cudaEventRecord(end, stream), "Record end event");
        Check(cudaEventSynchronize(end), "Synchronise measured work");
        float milliseconds = 0.f;
        Check(cudaEventElapsedTime(&milliseconds, start, end), "Read elapsed time");
        std::println("{},{},{},{},{:.8f}", sample, mode, scene, count, milliseconds);
    }

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
    Check(cudaEventDestroy(start), "Destroy start event");
    Check(cudaEventDestroy(end), "Destroy end event");
    Check(cudaStreamDestroy(stream), "Destroy stream");
    Check(cudaFree(objects), "Free objects");
    Check(cudaFree(seed_objects), "Free seed objects");
    Check(cudaFree(cells), "Free grid");
    Check(cudaFree(seed_cells), "Free seed grid");
    return EXIT_SUCCESS;
}
