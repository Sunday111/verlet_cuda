#pragma once
#include <functional>

#include "edt/math/matrix.hpp"

namespace verlet
{
class VerletObject;
using ObjectColorFunction = std::function<edt::Vec4f(const VerletObject&)>;
}  // namespace verlet
