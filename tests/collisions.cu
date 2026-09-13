#include <array>
#include <cmath>
#include <print>

#include "../src/verlet_cuda/code/private/kernels.cu"

__global__ void ResolvePair(verlet::VerletObject* objects, bool reverse, bool self_only = false)
{
    verlet::GridCell cell{.first_object_index = 0};
    objects[0].next_object_in_cell = self_only ? verlet::kInvalidObjectIndex : 1;
    objects[1].next_object_in_cell = verlet::kInvalidObjectIndex;
    auto& object = objects[reverse ? 1 : 0];
    auto position = object.position;
    verlet::kernels_impl::SolveCollisionBetweenObjectAndCell<true>(&cell, objects, object, position, 0);
    object.position = position;
}

__global__ void ResolveCell(verlet::VerletObject* objects)
{
    std::array<verlet::GridCell, 25> cells{};
    cells[12].first_object_index = 0;
    cells[13].first_object_index = 2;
    cells[11].first_object_index = 3;
    cells[17].first_object_index = 4;
    cells[7].first_object_index = 5;
    objects[0].next_object_in_cell = 1;
    verlet::kernels_impl::SolveCollisionsFromCell({2, 2}, 5, cells.data(), objects);
}

int main()
{
    verlet::VerletObject* objects = nullptr;
    if (cudaMallocManaged(&objects, 6 * sizeof(*objects)) != cudaSuccess) return 1;
    bool success = true;
    for (float distance :
         {0.f,
          1.e-30f,
          1.e-22f,
          1.e-20f,
          1.e-19f,
          0.005f,
          0.009f,
          std::nextafter(0.01f, 0.f),
          0.01f,
          std::nextafter(0.01f, 1.f),
          0.1f,
          0.5f,
          0.9f,
          1.f,
          2.f})
    {
        for (verlet::Vec2f direction :
             {verlet::Vec2f{1.f, 0.f}, verlet::Vec2f{-1.f, 0.f}, verlet::Vec2f{0.f, 1.f}, verlet::Vec2f{0.6f, 0.8f}})
        {
            for (bool reverse : {false, true})
            {
                const auto initial_separation = direction * distance;
                objects[0] = {.position = {0.f, 0.f}};
                objects[1] = {.position = initial_separation};
                ResolvePair<<<1, 1>>>(objects, reverse);
                if (cudaDeviceSynchronize() != cudaSuccess) return 1;
                const auto separation = objects[1].position - objects[0].position;
                const auto midpoint_error = objects[0].position + objects[1].position - initial_separation;
                const float expected = distance < 1.f ? distance + (1.f - distance) / 2 : distance;
                bool valid = std::abs(std::hypot(separation.x(), separation.y()) - expected) < 1.e-6f;
                valid &= std::abs(midpoint_error.x()) < 1.e-6f && std::abs(midpoint_error.y()) < 1.e-6f;
                if (initial_separation.SquaredLength() >= std::numeric_limits<float>::min())
                {
                    valid &= std::abs(separation.x() - direction.x() * expected) < 1.e-6f;
                    valid &= std::abs(separation.y() - direction.y() * expected) < 1.e-6f;
                }
                else
                {
                    valid &= separation.x() > 0.f && std::abs(separation.y()) < 1.e-6f;
                }
                if (!valid)
                {
                    std::println(
                        "Failed: distance={:g} direction=({:g},{:g}) reverse={:d} separation=({:g},{:g})",
                        distance,
                        direction.x(),
                        direction.y(),
                        reverse,
                        separation.x(),
                        separation.y());
                }
                success &= valid;
            }
        }
    }
    objects[0] = {.position = {2.f, 3.f}};
    ResolvePair<<<1, 1>>>(objects, false, true);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    success &= objects[0].position == verlet::Vec2f{2.f, 3.f};

    const std::array initial_positions{
        verlet::Vec2f{2.15f, 2.4f},
        verlet::Vec2f{2.7f, 2.6f},
        verlet::Vec2f{3.1f, 2.55f},
        verlet::Vec2f{1.8f, 2.4f},
        verlet::Vec2f{2.4f, 3.1f},
        verlet::Vec2f{2.6f, 1.8f}};
    std::array<edt::Vec2<double>, 6> expected_positions{};
    for (size_t index = 0; index < initial_positions.size(); ++index)
    {
        objects[index] = {.position = initial_positions[index]};
        expected_positions[index] = initial_positions[index].Cast<double>();
    }
    for (size_t origin = 0; origin < 2; ++origin)
    {
        for (size_t neighbour = 0; neighbour < expected_positions.size(); ++neighbour)
        {
            if (origin == neighbour) continue;
            const edt::Vec2<double> axis = expected_positions[origin] - expected_positions[neighbour];
            const double distance = std::hypot(axis.x(), axis.y());
            if (distance >= 1.) continue;
            const edt::Vec2<double> displacement = axis * (0.25 * (1. - distance) / distance);
            expected_positions[origin] += displacement;
            expected_positions[neighbour] -= displacement;
        }
    }
    ResolveCell<<<1, 1>>>(objects);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(objects);
        return 1;
    }
    for (size_t index = 0; index < expected_positions.size(); ++index)
    {
        const auto error = objects[index].position.Cast<double>() - expected_positions[index];
        const bool valid = std::abs(error.x()) < 1.e-6 && std::abs(error.y()) < 1.e-6;
        if (!valid)
        {
            std::println("Failed: multicell particle={} error=({:g},{:g})", index, error.x(), error.y());
        }
        success &= valid;
    }
    verlet::Vec2f* previous_positions = nullptr;
    if (cudaMallocManaged(&previous_positions, 6 * sizeof(*previous_positions)) != cudaSuccess)
    {
        cudaFree(objects);
        return 1;
    }
    const std::array integration_positions{
        verlet::Vec2f{0.f, 0.f},
        verlet::Vec2f{3.f, 4.f},
        verlet::Vec2f{959.f, 519.f},
        verlet::Vec2f{-959.f, -519.f},
        verlet::Vec2f{900.f, 0.f},
        verlet::Vec2f{0.f, 500.f}};
    const auto bounds = verlet::constants::kWorldRange.Enlarged(-2.f);
    for (size_t index = 0; index < integration_positions.size(); ++index)
    {
        const auto position = integration_positions[index];
        const verlet::Vec2f displacement{static_cast<float>(index) - 2.f, 3.f - static_cast<float>(index)};
        objects[index] = {.position = position};
        previous_positions[index] = position - displacement;
        const auto move = position.Cast<double>() - previous_positions[index].Cast<double>();
        const double dt_squared =
            verlet::constants::kTimeSubStepDurationSeconds * verlet::constants::kTimeSubStepDurationSeconds;
        const auto expected =
            position.Cast<double>() + move +
            (verlet::constants::kGravity.Cast<double>() - move * verlet::constants::kVelocityDamping) * dt_squared;
        expected_positions[index] = bounds.Clamp(expected.Cast<float>()).Cast<double>();
    }
    cudaStream_t stream = nullptr;
    if (verlet::Kernels::UpdatePositions(stream, integration_positions.size(), objects, previous_positions) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(previous_positions);
        cudaFree(objects);
        return 1;
    }
    for (size_t index = 0; index < integration_positions.size(); ++index)
    {
        const auto error = objects[index].position.Cast<double>() - expected_positions[index];
        const auto& expected = expected_positions[index];
        const bool valid = previous_positions[index] == integration_positions[index] &&
                           std::abs(error.x()) < 1.e-6 + std::abs(expected.x()) * 1.e-7 &&
                           std::abs(error.y()) < 1.e-6 + std::abs(expected.y()) * 1.e-7;
        if (!valid) std::println("Failed: integration particle={}", index);
        success &= valid;
    }
    cudaFree(previous_positions);
    cudaFree(objects);
    std::println(
        "{}",
        success ? "121 collision cases, multicell and integration regressions passed" : "Collision tests failed");
    return success ? 0 : 1;
}
