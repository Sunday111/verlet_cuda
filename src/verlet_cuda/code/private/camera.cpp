#include "camera.hpp"

namespace verlet
{
edt::FloatRange2Df Camera::ComputeRange(const edt::FloatRange2Df& world_range, const edt::Vec2f& viewport_size) const
{
    auto camera_extent = edt::Vec2f{} + world_range.Extent().Min();
    if (viewport_size.x() > viewport_size.y())
    {
        camera_extent.x() *= viewport_size.x() / viewport_size.y();
    }
    else
    {
        camera_extent.y() *= viewport_size.y() / viewport_size.x();
    }

    camera_extent /= GetZoom();
    const auto half_camera_extent = camera_extent / 2;

    return edt::FloatRange2Df::FromMinMax(GetEye() - half_camera_extent, GetEye() + half_camera_extent);
}

void Camera::Update(const edt::FloatRange2Df& world_range, const edt::Vec2f& viewport_size)
{
    range_ = ComputeRange(world_range, viewport_size);
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
            .final_value = std::max(final_value, 0.000001f),
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
