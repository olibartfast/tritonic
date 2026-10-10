#include "tritonic/cuda/letterbox.hpp"
#include "tritonic/cuda/letterbox_pixel.hpp"

namespace tritonic::cuda {
namespace {

__global__ void LetterboxKernel(const uint8_t* __restrict__ source, int height, int width,
                                size_t pitch, bool bgr, float* __restrict__ destination, int canvas,
                                LetterboxGeometry geometry) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < canvas && y < canvas) {
        detail::LetterboxPixel(source, height, width, pitch, bgr, destination, canvas, geometry, x,
                               y);
    }
}

}  // namespace

cudaError_t LaunchLetterbox(const LetterboxSource& source, float* destination, cudaStream_t stream,
                            int canvas, PadRounding rounding) {
    const auto geometry = ComputeLetterboxGeometry(source.height, source.width, canvas, rounding);
    if (!geometry || source.data == nullptr || destination == nullptr ||
        source.pitch_bytes < static_cast<size_t>(source.width) * 3) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(32, 8);
    const dim3 grid((canvas + block.x - 1) / block.x, (canvas + block.y - 1) / block.y);
    LetterboxKernel<<<grid, block, 0, stream>>>(
        source.data, source.height, source.width, source.pitch_bytes,
        source.order == ChannelOrder::BGR, destination, canvas, *geometry);
    return cudaGetLastError();
}

}  // namespace tritonic::cuda
