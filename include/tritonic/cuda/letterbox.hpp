#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "tritonic/cuda/letterbox_geometry.hpp"

namespace tritonic::cuda {

enum class ChannelOrder { RGB, BGR };

struct LetterboxSource {
    const uint8_t* data;  // device pointer, interleaved HWC, 3 channels
    int height;
    int width;
    size_t pitch_bytes;  // row stride in bytes (>= width * 3)
    ChannelOrder order;
};

// Writes the spec R-2 letterbox of `source` into `destination`: a device buffer of
// 3 * canvas * canvas floats, RGB CHW, in [0, 1]. Every output element is written exactly
// once, padding included, so the destination needs no prior memset. The launch is
// asynchronous on `stream`. Returns cudaErrorInvalidValue for degenerate geometry.
cudaError_t LaunchLetterbox(const LetterboxSource& source, float* destination, cudaStream_t stream,
                            int canvas = kLetterboxCanvas,
                            PadRounding rounding = kDefaultPadRounding);

}  // namespace tritonic::cuda
