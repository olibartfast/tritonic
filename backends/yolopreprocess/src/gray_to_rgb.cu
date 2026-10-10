#include "gray_to_rgb.hpp"

namespace triton {
namespace backend {
namespace yolopreprocess {

namespace {

__global__ void GrayToRgbKernel(const uint8_t* __restrict__ gray, size_t pixels,
                                uint8_t* __restrict__ rgb) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < pixels) {
        const uint8_t value = gray[index];
        rgb[index * 3] = value;
        rgb[index * 3 + 1] = value;
        rgb[index * 3 + 2] = value;
    }
}

}  // namespace

cudaError_t LaunchGrayToRgb(const uint8_t* gray, int height, int width, uint8_t* rgb,
                            cudaStream_t stream) {
    const size_t pixels = static_cast<size_t>(height) * width;
    if (pixels == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((pixels + block - 1) / block);
    GrayToRgbKernel<<<grid, block, 0, stream>>>(gray, pixels, rgb);
    return cudaGetLastError();
}

}  // namespace yolopreprocess
}  // namespace backend
}  // namespace triton
