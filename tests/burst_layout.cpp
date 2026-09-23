#include "../src/verlet_cuda/code/private/emitters/burst_layout.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
    const auto require = [](bool condition)
    {
        if (!condition) throw std::runtime_error("Burst layout regression");
    };
    require(verlet::BurstLayout::Generate(0, true, true).empty());
    constexpr auto bounds = verlet::constants::kWorldRange.Enlarged(-2.f);
    const auto less = [](const verlet::Vec2f& a, const verlet::Vec2f& b)
    {
        return a.y() != b.y() ? a.y() < b.y() : a.x() < b.x();
    };
    for (bool packing : {false, true})
    {
        for (size_t count : {size_t{1}, size_t{37}, size_t{10000}})
        {
            auto ordered = verlet::BurstLayout::Generate(count, packing, false);
            auto shuffled = verlet::BurstLayout::Generate(count, packing, true);
            require(ordered.size() == count && shuffled.size() == count);
            require(shuffled == verlet::BurstLayout::Generate(count, packing, true));
            if (count > 1) require(ordered != shuffled);
            for (const auto& position : ordered) require(position == bounds.Clamp(position));
            std::sort(ordered.begin(), ordered.end(), less);
            std::sort(shuffled.begin(), shuffled.end(), less);
            require(ordered == shuffled);
            if (packing)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    for (size_t j = i + 1; j < count && ordered[j].y() - ordered[i].y() < 1.f; ++j)
                        require((ordered[j] - ordered[i]).SquaredLength() >= 1.f);
                }
            }
        }
    }
    const auto too_many = static_cast<size_t>(bounds.Extent().x() * bounds.Extent().y() * 2.f);
    require(verlet::BurstLayout::Generate(too_many, true, false).empty());
    std::cout << "Burst layout checks passed\n";
}
