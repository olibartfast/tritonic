#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace triton {
namespace backend {
namespace yolopreprocess {

// Expands a tightly packed single-channel image into interleaved RGB (R = G = B), so
// grayscale JPEGs follow the same letterbox path as color ones.
cudaError_t LaunchGrayToRgb(const uint8_t* gray, int height, int width, uint8_t* rgb,
                            cudaStream_t stream);

}  // namespace yolopreprocess
}  // namespace backend
}  // namespace triton
