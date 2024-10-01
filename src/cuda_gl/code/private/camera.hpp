#pragma once

#include <EverydayTools/Math/FloatRange.hpp>
#include <algorithm>
#include <chrono>

namespace verlet
{

template <typename T>
struct ValueAnimation
{
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    // Returns true if animation is completed
    bool Update(T& value)
    {
        const auto now = Clock::now();
        if (now > begin_time + std::chrono::duration<float>(duration_seconds))
        {
            value = final_value;
            return true;
        }

        const auto dt =
            std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1>>>(now - begin_time).count();
        const float t = std::clamp(dt / duration_seconds, 0.f, 1.f);
        value = start_value + t * (final_value - start_value);

        return false;
    }

    T start_value{};
    T final_value{};
    float duration_seconds = 0.f;

    TimePoint begin_time = Clock::now();
};

class Camera
{
public:
    [[nodiscard]] const edt::FloatRange2Df& GetRange() const { return range_; }
    [[nodiscard]] float GetZoom() const { return zoom_; }
    [[nodiscard]] const edt::Vec2f& GetEye() const { return eye_; }

    void Zoom(const float delta);
    void Pan(const edt::Vec2f& delta);

    void Update(const edt::FloatRange2Df& world_range);

    float zoom_speed = 0.2f;
    float zoom_animation_diration_seconds = 0.3f;
    float pan_speed = 0.3f;
    float pan_animation_diration_seconds = 0.3f;
    bool animate = true;

private:
    edt::FloatRange2Df ComputeRange(const edt::FloatRange2Df& world_range) const;

private:
    edt::FloatRange2Df range_ = {};

    float zoom_ = 1.f;
    std::optional<ValueAnimation<float>> zoom_animation_;

    edt::Vec2f eye_{};
    std::optional<ValueAnimation<edt::Vec2f>> eye_animation_;
};
}  // namespace verlet
