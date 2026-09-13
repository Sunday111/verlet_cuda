#include <cmath>
#include <cstdio>

#include "../src/verlet_cuda/code/private/kernels.cu"

__global__ void ResolvePair(verlet::VerletObject* objects, bool reverse, bool self_only = false)
{
    verlet::GridCell cell{.first_object_index = 0};
    objects[0].next_object_in_cell = self_only ? verlet::kInvalidObjectIndex : 1;
    objects[1].next_object_in_cell = verlet::kInvalidObjectIndex;
    verlet::kernels_impl::SolveCollisionBetweenObjectAndCell<true>(&cell, objects, objects[reverse ? 1 : 0], 0);
}

int main()
{
    verlet::VerletObject* objects = nullptr;
    if (cudaMallocManaged(&objects, 2 * sizeof(*objects)) != cudaSuccess) return 1;
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
                    std::printf(
                        "Failed: distance=%g direction=(%g,%g) reverse=%d separation=(%g,%g)\n",
                        static_cast<double>(distance),
                        static_cast<double>(direction.x()),
                        static_cast<double>(direction.y()),
                        reverse,
                        static_cast<double>(separation.x()),
                        static_cast<double>(separation.y()));
                }
                success &= valid;
            }
        }
    }
    objects[0] = {.position = {2.f, 3.f}};
    ResolvePair<<<1, 1>>>(objects, false, true);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    success &= objects[0].position == verlet::Vec2f{2.f, 3.f};
    cudaFree(objects);
    std::puts(success ? "121 collision cases passed" : "Collision tests failed");
    return success ? 0 : 1;
}
