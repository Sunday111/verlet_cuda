#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "constants.hpp"

namespace verlet
{
struct BurstLayout
{
    [[nodiscard]] static std::vector<Vec2f> Generate(size_t count, bool packing, bool shuffle_storage)
    {
        if (count == 0) return {};
        constexpr auto bounds = constants::kWorldRange.Enlarged(-2.f);
        constexpr auto extent = bounds.Extent();
        if (extent.Min() <= 0.f) return {};
        constexpr float spacing = 2.f * constants::kObjectRadius * 1.001f;
        const bool stagger = extent.x() >= spacing;
        const float row_spacing = spacing * (stagger ? std::sqrt(3.f) * 0.5f : 1.f);
        const size_t columns = packing ? std::max(size_t{1}, static_cast<size_t>(extent.x() / spacing))
                                       : std::clamp(
                                             static_cast<size_t>(std::ceil(
                                                 std::sqrt(
                                                     static_cast<double>(count) * static_cast<double>(extent.x()) /
                                                     static_cast<double>(extent.y())))),
                                             size_t{1},
                                             count);
        const size_t rows = (count - 1) / columns + 1;
        if (packing &&
            static_cast<double>(rows - 1) * static_cast<double>(row_spacing) > static_cast<double>(extent.y()))
            return {};

        std::vector<Vec2f> positions;
        positions.reserve(count);
        const Vec2f step = extent / Vec2f{static_cast<float>(columns), static_cast<float>(rows)};
        for (size_t index = 0; index < count; ++index)
        {
            const size_t row = index / columns, column = index % columns;
            if (packing)
            {
                positions.push_back(
                    {bounds.Min().x() + extent.x() * 0.5f +
                         (static_cast<float>(column) - static_cast<float>(columns - 1) * 0.5f +
                          (stagger ? static_cast<float>(row & 1) * 0.5f : 0.f)) *
                             spacing,
                     bounds.Min().y() + static_cast<float>(row) * row_spacing});
            }
            else
            {
                positions.push_back(
                    bounds.Min() + step * Vec2f{static_cast<float>(column) + 0.5f, static_cast<float>(row) + 0.5f});
            }
        }
        if (shuffle_storage)
        {
            std::mt19937 random{12345};
            std::shuffle(positions.begin(), positions.end(), random);
        }
        return positions;
    }
};
}  // namespace verlet
