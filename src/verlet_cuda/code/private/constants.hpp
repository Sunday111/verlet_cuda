#pragma once

#include "EverydayTools/Math/FloatRange.hpp"
#include "EverydayTools/Math/Matrix.hpp"
#include "device_types.h"

namespace verlet
{

using namespace edt::lazy_matrix_aliases;  // NOLINT

namespace constants
{
// Maximum number of objects for a single cell
inline constexpr size_t kGridMaxObjectsInCell = 4;

// The size of each grid cell in world coordinates
__constant__ constexpr Vec2<size_t> kGridCellSize{1, 1};
__constant__ constexpr float kObjectRadius = 0.5f;
static constexpr Vec2f init_corner{960, 520};
__constant__ constexpr edt::FloatRange2Df kWorldRange = edt::FloatRange2Df::FromMinMax(-init_corner, init_corner);
__constant__ constexpr auto kGridSize = 2 + kWorldRange.Extent().Cast<size_t>() / kGridCellSize;
__constant__ constexpr auto kGridNumCells = kGridSize.x() * kGridSize.y();
__constant__ constexpr float kTimeStepDurationSeconds = 1.f / 60.f;
__constant__ constexpr size_t kNumSubSteps = 8;
__constant__ constexpr float kTimeSubStepDurationSeconds = kTimeStepDurationSeconds / static_cast<float>(kNumSubSteps);
__constant__ constexpr edt::Vec2f gravity{0.0f, -20.f};
__constant__ constexpr float kVelocityDampling = 40.f;  // arbitrary, approximating air friction
}  // namespace constants
}  // namespace verlet
