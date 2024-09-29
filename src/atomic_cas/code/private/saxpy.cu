#include <stdio.h>
#include <vector>
#include <span>
#include <ranges>
#include <algorithm>
#include <bit>

#include "util.hpp"
#include "saxpy.hpp"

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
__global__ void find_max_error(int n, float* max_error, const float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float error = std::abs(y[i] - 4.f);
        while (true) {
            if (const float prev_max = *max_error; error <= prev_max ||
                atomicCAS(reinterpret_cast<uint32_t*>(max_error),
                    std::bit_cast<uint32_t>(prev_max),
                    std::bit_cast<uint32_t>(error))
                    == std::bit_cast<uint32_t>(prev_max))
            {
                break;
            }
        }
    }
}

template<bool kUseCudaFindMaxError>
void Main()
{
    constexpr size_t N = 1'000'000;

    std::vector<float> x(N, 1.f);
    auto d_x = MakeCudaArray<float>(x.size());

    std::vector<float> y(N, 2.f);
    auto d_y = MakeCudaArray<float>(y.size());

    printf("    n = %zu\n", N);

    float copy_to_device_ms = MeasureMillis([&] {
        CopyCudaArrayToDevice(std::span{x}, d_x);
        CopyCudaArrayToDevice(std::span{y}, d_y);
    });
    printf("    Copy to device: %.2fms\n", copy_to_device_ms);

    // Perform SAXPY on elements
    float execute_kernel_ms =  MeasureMillis([&]{
        saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x.get(), d_y.get());
    });
    printf("    SAXPY: %.2fms\n", execute_kernel_ms);

    float copy_to_host_ms = MeasureMillis([&] {
        CopyCudaArrayToHost(std::span{y}, d_y);
    });
    printf("    Copy to host: %.2fms\n", copy_to_host_ms);

    float max_error = -100.0f;
    float find_max_error_ms = MeasureMillis([&] {
        if constexpr (kUseCudaFindMaxError)
        {
            auto device_max_error = MakeCudaArray<float>(1);
            CopyCudaArrayToHost(std::span{&max_error, 1}, device_max_error);
            find_max_error<<<(N + 255) / 256, 256>>>(N, device_max_error.get(), d_y.get());
            CopyCudaArrayToHost(std::span{&max_error, 1}, device_max_error);
        }
        else
        {
            for (float v: y)
            {
                max_error = std::max(max_error, std::abs(v - 4.0f));
            }
        }
    });
    printf("    Find max error: %.2fms\n", find_max_error_ms);

    printf("    Max error: %.2f\n", max_error);
}

void RunCudaSaxpy()
{
    std::puts("Find Max On CPU");
    Main<false>();

    std::puts("\nFind Max On GPU");
    Main<true>();
}
