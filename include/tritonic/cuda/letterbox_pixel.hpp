#pragma once

// Per-pixel spec R-2 letterbox math shared by the CUDA kernel and the host reference.
// Compiled as __host__ __device__ under nvcc and as plain C++ elsewhere, so the CPU unit
// tests exercise the exact arithmetic the kernel runs.

#include <math.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "tritonic/cuda/letterbox_geometry.hpp"

#if defined(__CUDACC__)
#define TRITONIC_HD __host__ __device__ __forceinline__
#else
#define TRITONIC_HD inline
#endif

namespace tritonic::cuda::detail {

struct Axis {
    int low;
    int high;
    float weight;
};

// Half-pixel source coordinate clamped to the input range: DALI linear resize without
// antialiasing, ONNX Resize half_pixel, and letterbox_reference.py all use this mapping.
TRITONIC_HD Axis SourceAxis(int index, int outputSize, int inputSize) {
    const float scale = static_cast<float>(inputSize) / static_cast<float>(outputSize);
    float coordinate = (static_cast<float>(index) + 0.5f) * scale - 0.5f;
    coordinate = fminf(fmaxf(coordinate, 0.0f), static_cast<float>(inputSize - 1));
    const int low = static_cast<int>(floorf(coordinate));
    const int high = low + 1 < inputSize ? low + 1 : inputSize - 1;
    return {low, high, coordinate - static_cast<float>(low)};
}

// Writes output pixel (x, y) of all three CHW planes.
TRITONIC_HD void LetterboxPixel(const uint8_t* source, int height, int width, size_t pitch,
                                bool bgr, float* destination, int canvas,
                                const LetterboxGeometry& geometry, int x, int y) {
    const size_t plane = static_cast<size_t>(canvas) * canvas;
    const size_t offset = static_cast<size_t>(y) * canvas + x;
    const int resizedX = x - geometry.left;
    const int resizedY = y - geometry.top;
    if (resizedX < 0 || resizedY < 0 || resizedX >= geometry.resized_width ||
        resizedY >= geometry.resized_height) {
        destination[offset] = 0.0f;
        destination[plane + offset] = 0.0f;
        destination[2 * plane + offset] = 0.0f;
        return;
    }

    const Axis row = SourceAxis(resizedY, geometry.resized_height, height);
    const Axis col = SourceAxis(resizedX, geometry.resized_width, width);
    const uint8_t* top = source + static_cast<size_t>(row.low) * pitch;
    const uint8_t* bottom = source + static_cast<size_t>(row.high) * pitch;
    const int left = col.low * 3;
    const int right = col.high * 3;

    for (int channel = 0; channel < 3; ++channel) {
        const int input = bgr ? 2 - channel : channel;
        const float upper = static_cast<float>(top[left + input]) * (1.0f - col.weight) +
                            static_cast<float>(top[right + input]) * col.weight;
        const float lower = static_cast<float>(bottom[left + input]) * (1.0f - col.weight) +
                            static_cast<float>(bottom[right + input]) * col.weight;
        // Round back to 8-bit like DALI's uint8 resize output, then scale.
        const float value =
            fminf(fmaxf(rintf(upper * (1.0f - row.weight) + lower * row.weight), 0.0f), 255.0f);
        destination[channel * plane + offset] = value * (1.0f / 255.0f);
    }
}

}  // namespace tritonic::cuda::detail

namespace tritonic::cuda {

// Host reference over the same per-pixel math; used by unit tests and parity tooling.
// Returns false for degenerate geometry. `destination` holds 3 * canvas * canvas floats.
inline bool LetterboxHost(const uint8_t* source, int height, int width, size_t pitch, bool bgr,
                          float* destination, int canvas = kLetterboxCanvas,
                          PadRounding rounding = kDefaultPadRounding) {
    const auto geometry = ComputeLetterboxGeometry(height, width, canvas, rounding);
    if (!geometry || pitch < static_cast<size_t>(width) * 3) {
        return false;
    }
    for (int y = 0; y < canvas; ++y) {
        for (int x = 0; x < canvas; ++x) {
            detail::LetterboxPixel(source, height, width, pitch, bgr, destination, canvas,
                                   *geometry, x, y);
        }
    }
    return true;
}

}  // namespace tritonic::cuda
