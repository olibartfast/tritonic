#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "tritonic/core/device_preprocessor.hpp"
#include "tritonic/cuda/letterbox_geometry.hpp"

namespace tritonic::cuda {

// Limit on either side of an input frame (spec R-11); larger frames are rejected.
inline constexpr int kMaxInputSide = 8192;

/**
 * V2 client-side preprocessing: uploads a decoded BGR frame and runs the shared letterbox
 * kernel into a caller-provided device buffer (the registered CUDA shared-memory input).
 * Owns one stream and one upload buffer, reused across frames; not thread-safe.
 */
class LetterboxPreprocessor final : public core::IDevicePreprocessor {
public:
    explicit LetterboxPreprocessor(int deviceId, PadRounding rounding = kDefaultPadRounding);
    ~LetterboxPreprocessor() override;

    LetterboxPreprocessor(const LetterboxPreprocessor&) = delete;
    LetterboxPreprocessor& operator=(const LetterboxPreprocessor&) = delete;

    size_t outputBytes() const override;
    void preprocess(const cv::Mat& bgrFrame, void* deviceDestination) override;

private:
    int deviceId_;
    PadRounding rounding_;
    cudaStream_t stream_{nullptr};
    uint8_t* upload_{nullptr};
    size_t uploadCapacity_{0};
};

}  // namespace tritonic::cuda
