#include "cuda/cuda_algorithms.hpp"
#include <vector>

#include <algorithm>
#include <functional>
#include <execution>
#include <random>

#include "fmt/core.h"
#include "fmt/chrono.h"
#include "time.hpp"

[[nodiscard]] int StlMaxElement(std::span<const int> values)
{
    int result = 0;

    const auto duration = MeasureTime([&] {
        result = *std::max_element(std::execution::par_unseq, values.begin(), values.end());
    });

    fmt::println("STL find max element:");
    fmt::println("  duration: {}", duration);
    fmt::println("  result: {}", result);

    return result;
}

void StlVectorsLengths(std::span<const edt::Vec3f> vectors, std::span<float> out_lengths) {
    fmt::println("STL vectors lengths: {}", MeasureTime([&]{
        for (size_t i = 0; i != vectors.size(); ++i) {
            out_lengths[i] = vectors[i].Length();
        }
    }));
}

void TestMaxElement() {
    constexpr size_t max_n = 10'000'000;
    std::vector<int> values{0};
    values.reserve(max_n);

    for (size_t n = 1; n <= max_n; n *= 10) {
        while(values.size() != n) values.push_back(values.back() + 1);

        fmt::println("");
        fmt::println("n = {}", n);
        [[maybe_unused]] auto cuda_result = CudaAlgorithms::MaxElement(values);
        [[maybe_unused]] auto stl_result = StlMaxElement(values);
    }
}

void TestVectorLengths() {
    constexpr size_t max_n = 100'000'000;
    constexpr unsigned kSeed = 0;
    std::mt19937 rnd(kSeed);
    std::uniform_real_distribution<float> distribution(-100.f, 100.f);

    std::vector<edt::Vec3f> vectors;
    std::vector<float> lengths;
    const auto generate_vectors_duration = MeasureTime([&] {
        vectors.reserve(max_n);
        lengths.reserve(max_n);
        for (size_t i = 0; i != max_n; ++i) {
            vectors.push_back({
                distribution(rnd),
                distribution(rnd),
                distribution(rnd)
            });
        }
    });

    fmt::println("Generate {} vectors: {:.2}", max_n, generate_vectors_duration);
    lengths.resize(max_n);
    StlVectorsLengths(std::span{vectors}, std::span{lengths});
    CudaAlgorithms::VectorsLengths(std::span{vectors}, std::span{lengths});
}

int main()
{
    TestVectorLengths();
}
