#pragma once

#include <opencv2/core.hpp>

namespace tritonic::core {

/**
 * Client-side preprocessing that writes the model input tensor directly into device
 * memory (specs/2026-10-01-gpu-preprocess-alternatives, variant V2).
 *
 * The CUDA implementation lives in src/cuda/ and is only built with
 * TRITONIC_ENABLE_CUDA=ON; App depends on this interface alone.
 */
class IDevicePreprocessor {
public:
    virtual ~IDevicePreprocessor() = default;

    // Bytes the preprocessor writes into the destination buffer.
    virtual size_t outputBytes() const = 0;

    // Letterboxes a decoded BGR frame into `deviceDestination` and returns once the write is
    // complete. Throws std::invalid_argument for frames the contract rejects (empty,
    // non-8UC3, degenerate aspect ratio) and std::runtime_error for device failures.
    virtual void preprocess(const cv::Mat& bgrFrame, void* deviceDestination) = 0;
};

}  // namespace tritonic::core
