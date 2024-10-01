#pragma once

#include <chrono>

inline auto MeasureTime(auto&& f)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float, std::milli>;
    auto start_time = Clock::now();
    f();
    auto finish_time = Clock::now();
    return std::chrono::duration_cast<Duration>(finish_time - start_time);
}
