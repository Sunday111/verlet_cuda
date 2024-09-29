#pragma once

#include <span>

#include "EverydayTools/Math/Matrix.hpp"

class CudaAlgorithms
{
public:
    [[nodiscard]] static int MaxElement(std::span<const int> values);
    static void VectorsLengths(std::span<const edt::Vec3f> vectors, std::span<float> out_lengths);
};
