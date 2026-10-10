#pragma once

#include <cstdint>
#include <optional>

namespace tritonic::cuda {

// Pad-offset rounding for the centered paste. DALI's rule is measured by spike T-3
// (specs/2026-10-01-gpu-preprocess-alternatives/plan.md) and kDefaultPadRounding
// follows the measurement.
enum class PadRounding { Floor, Ceil };

inline constexpr int kLetterboxCanvas = 640;
inline constexpr PadRounding kDefaultPadRounding = PadRounding::Floor;

struct LetterboxGeometry {
    int resized_height;
    int resized_width;
    int top;
    int left;
};

// Spec R-2 geometry: the longer side maps to the canvas with integer flooring, and the
// resized image is centered. Returns nullopt for non-positive sizes and for aspect ratios
// beyond canvas:1, where one side would collapse to zero pixels; callers reject those.
inline std::optional<LetterboxGeometry> ComputeLetterboxGeometry(
    int height, int width, int canvas = kLetterboxCanvas,
    PadRounding rounding = kDefaultPadRounding) {
    if (height <= 0 || width <= 0 || canvas <= 0) {
        return std::nullopt;
    }
    const int64_t extent = height > width ? height : width;
    const int resizedWidth = static_cast<int>(static_cast<int64_t>(width) * canvas / extent);
    const int resizedHeight = static_cast<int>(static_cast<int64_t>(height) * canvas / extent);
    if (resizedWidth == 0 || resizedHeight == 0) {
        return std::nullopt;
    }
    const int roundUp = rounding == PadRounding::Ceil ? 1 : 0;
    return LetterboxGeometry{resizedHeight, resizedWidth, (canvas - resizedHeight + roundUp) / 2,
                             (canvas - resizedWidth + roundUp) / 2};
}

}  // namespace tritonic::cuda
