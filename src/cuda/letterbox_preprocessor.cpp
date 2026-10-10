#include "tritonic/cuda/letterbox_preprocessor.hpp"

#include <stdexcept>
#include <string>

#include "tritonic/cuda/letterbox.hpp"

namespace tritonic::cuda {
namespace {

void Check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA preprocessing: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

}  // namespace

LetterboxPreprocessor::LetterboxPreprocessor(int deviceId, PadRounding rounding)
    : deviceId_(deviceId), rounding_(rounding) {
    Check(cudaSetDevice(deviceId_), "cudaSetDevice");
    Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
}

LetterboxPreprocessor::~LetterboxPreprocessor() {
    cudaFree(upload_);
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

size_t LetterboxPreprocessor::outputBytes() const {
    return 3 * static_cast<size_t>(kLetterboxCanvas) * kLetterboxCanvas * sizeof(float);
}

void LetterboxPreprocessor::preprocess(const cv::Mat& bgrFrame, void* deviceDestination) {
    if (bgrFrame.empty() || bgrFrame.type() != CV_8UC3) {
        throw std::invalid_argument("CUDA preprocessing expects a non-empty 8-bit BGR frame");
    }
    if (bgrFrame.rows > kMaxInputSide || bgrFrame.cols > kMaxInputSide) {
        throw std::invalid_argument("frame exceeds the " + std::to_string(kMaxInputSide) +
                                    " pixel side limit");
    }
    if (!ComputeLetterboxGeometry(bgrFrame.rows, bgrFrame.cols, kLetterboxCanvas, rounding_)) {
        throw std::invalid_argument(
            "frame aspect ratio is too extreme to letterbox: " + std::to_string(bgrFrame.cols) +
            "x" + std::to_string(bgrFrame.rows));
    }
    if (deviceDestination == nullptr) {
        throw std::invalid_argument("CUDA preprocessing destination is null");
    }

    Check(cudaSetDevice(deviceId_), "cudaSetDevice");
    const size_t rowBytes = static_cast<size_t>(bgrFrame.cols) * 3;
    const size_t bytes = rowBytes * bgrFrame.rows;
    if (bytes > uploadCapacity_) {
        cudaFree(upload_);
        upload_ = nullptr;
        uploadCapacity_ = 0;
        Check(cudaMalloc(reinterpret_cast<void**>(&upload_), bytes), "cudaMalloc");
        uploadCapacity_ = bytes;
    }
    Check(cudaMemcpy2DAsync(upload_, rowBytes, bgrFrame.data, bgrFrame.step, rowBytes,
                            bgrFrame.rows, cudaMemcpyHostToDevice, stream_),
          "upload");
    Check(LaunchLetterbox({upload_, bgrFrame.rows, bgrFrame.cols, rowBytes, ChannelOrder::BGR},
                          static_cast<float*>(deviceDestination), stream_, kLetterboxCanvas,
                          rounding_),
          "letterbox kernel");
    // Triton reads the region from another process, so the write must be complete first.
    Check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace tritonic::cuda
