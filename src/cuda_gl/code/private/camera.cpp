#include "camera.hpp"

namespace verlet
{
edt::FloatRange2Df Camera::ComputeRange(const edt::FloatRange2Df& world_range) const
{
    const auto half_world_extent = world_range.Extent() / 2;
    const auto half_camera_extent = half_world_extent / GetZoom();
    return edt::FloatRange2Df::FromMinMax(GetEye() - half_camera_extent, GetEye() + half_camera_extent);
}

void Camera::Update(const edt::FloatRange2Df& world_range)
{
    range_ = ComputeRange(world_range);
    if (zoom_animation_ && zoom_animation_->Update(zoom_)) zoom_animation_ = std::nullopt;
    if (eye_animation_ && eye_animation_->Update(eye_)) eye_animation_ = std::nullopt;
}

void Camera::Zoom(const float delta)
{
    if (animate)
    {
        float final_value = zoom_ + delta;
        if (zoom_animation_)
        {
            final_value = zoom_animation_->final_value + delta;
        }

        zoom_animation_ = ValueAnimation{
            .start_value = zoom_,
            .final_value = final_value,
            .duration_seconds = zoom_animation_diration_seconds,
        };
    }
    else
    {
        zoom_ += delta;
    }
}

void Camera::Pan(const edt::Vec2f& delta)
{
    if (animate)
    {
        edt::Vec2f final_value = eye_ + delta;
        if (eye_animation_)
        {
            final_value = eye_animation_->final_value + delta;
        }

        eye_animation_ = ValueAnimation{
            .start_value = eye_,
            .final_value = final_value,
            .duration_seconds = zoom_animation_diration_seconds,
        };
    }
    else
    {
        eye_ += delta;
    }
}

}  // namespace verlet
