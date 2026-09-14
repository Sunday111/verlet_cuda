#pragma once

#include "edt/math/float_range.hpp"
#include "edt/math/matrix.hpp"

#ifndef VERLET_WORLD_WIDTH
#define VERLET_WORLD_WIDTH 1920
#endif
#ifndef VERLET_WORLD_HEIGHT
#define VERLET_WORLD_HEIGHT 1040
#endif

#ifndef VERLET_MAX_OBJECTS
#define VERLET_MAX_OBJECTS 2000000
#endif

namespace verlet
{

using namespace edt::lazy_matrix_aliases;  // NOLINT

namespace constants
{
// Maximum number of objects for a single cell
inline constexpr size_t kGridMaxObjectsInCell = 4;
inline constexpr size_t kMaxObjects = VERLET_MAX_OBJECTS;

// The size of each grid cell in world coordinates
inline constexpr Vec2<size_t> kGridCellSize{1, 1};
inline constexpr float kObjectRadius = 0.5f;
inline constexpr Vec2u32 kVerletWorldSizeU = Vec2u32{VERLET_WORLD_WIDTH, VERLET_WORLD_HEIGHT};
inline constexpr Vec2f kVerletWorldSizeF = kVerletWorldSizeU.Cast<float>();
inline constexpr Vec2f kInitialCorner = kVerletWorldSizeF / 2;
inline constexpr edt::FloatRange2Df kWorldRange = edt::FloatRange2Df::FromMinMax(-kInitialCorner, kInitialCorner);
inline constexpr auto kGridSize = 2 + kWorldRange.Extent().Cast<size_t>() / kGridCellSize;
inline constexpr auto kGridNumCells = kGridSize.x() * kGridSize.y();
inline constexpr float kTimeStepDurationSeconds = 1.f / 60.f;
inline constexpr size_t kNumSubSteps = 8;
inline constexpr float kTimeSubStepDurationSeconds = kTimeStepDurationSeconds / static_cast<float>(kNumSubSteps);
inline constexpr edt::Vec2f kGravity{0.0f, -20.f};
inline constexpr float kVelocityDamping = 40.f;  // arbitrary, approximating air friction
}  // namespace constants
}  // namespace verlet
