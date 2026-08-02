#pragma once

#include "edt/math/float_range.hpp"
#include "edt/math/matrix.hpp"

namespace verlet
{

using namespace edt::lazy_matrix_aliases;  // NOLINT

namespace constants
{
// Maximum number of objects for a single cell
inline constexpr size_t kGridMaxObjectsInCell = 4;

// The size of each grid cell in world coordinates
inline constexpr Vec2<size_t> kGridCellSize{1, 1};
inline constexpr float kObjectRadius = 0.5f;
static constexpr Vec2f init_corner{960, 520};
inline constexpr edt::FloatRange2Df kWorldRange = edt::FloatRange2Df::FromMinMax(-init_corner, init_corner);
inline constexpr auto kGridSize = 2 + kWorldRange.Extent().Cast<size_t>() / kGridCellSize;
inline constexpr auto kGridNumCells = kGridSize.x() * kGridSize.y();
inline constexpr float kTimeStepDurationSeconds = 1.f / 60.f;
inline constexpr size_t kNumSubSteps = 8;
inline constexpr float kTimeSubStepDurationSeconds = kTimeStepDurationSeconds / static_cast<float>(kNumSubSteps);
inline constexpr edt::Vec2f gravity{0.0f, -20.f};
inline constexpr float kVelocityDampling = 40.f;  // arbitrary, approximating air friction
}  // namespace constants
}  // namespace verlet
