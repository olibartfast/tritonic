#include <gtest/gtest.h>

#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"
#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

float ReadFloat(const std::vector<std::uint8_t>& bytes, std::size_t index) {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
    return value;
}

void ExpectPlaneValue(const std::vector<std::uint8_t>& bytes, std::size_t plane, std::size_t pixel,
                      float expected) {
    constexpr std::size_t kPixelsPerPlane = 16;
    EXPECT_FLOAT_EQ(ReadFloat(bytes, plane * kPixelsPerPlane + pixel), expected);
}

}  // namespace

TEST(YoloPreprocessingContractTest, EmitsRgbFloat32NchwNormalizedToUnitRange) {
    cv::Mat image(1, 1, CV_8UC3, cv::Scalar(10, 20, 30));  // BGR
    neuriplo_tasks::YoloPreprocessor preprocessor({4, 4});

    const auto bytes = preprocessor.preprocess(neuriplo_tasks::vision::opencv::toImageView(image));

    ASSERT_EQ(bytes.size(), 3U * 4U * 4U * sizeof(float));
    for (std::size_t pixel = 0; pixel < 16; ++pixel) {
        ExpectPlaneValue(bytes, 0, pixel, 30.0F / 255.0F);  // R
        ExpectPlaneValue(bytes, 1, pixel, 20.0F / 255.0F);  // G
        ExpectPlaneValue(bytes, 2, pixel, 10.0F / 255.0F);  // B
    }
}

TEST(YoloPreprocessingContractTest, LetterboxUsesZeroPaddingAndTruncatedScaledSize) {
    cv::Mat image(2, 3, CV_8UC3, cv::Scalar(10, 20, 30));
    neuriplo_tasks::YoloPreprocessor preprocessor({4, 4});

    const auto bytes = preprocessor.preprocess(neuriplo_tasks::vision::opencv::toImageView(image));

    // scale=min(4/3,4/2), so the implementation truncates 2*scale to 2 rows.
    // Centering places those rows at y=1 and y=2, leaving zero rows at y=0,3.
    for (std::size_t x = 0; x < 4; ++x) {
        for (std::size_t plane = 0; plane < 3; ++plane) {
            ExpectPlaneValue(bytes, plane, x, 0.0F);
            ExpectPlaneValue(bytes, plane, 12 + x, 0.0F);
        }
        ExpectPlaneValue(bytes, 0, 4 + x, 30.0F / 255.0F);
        ExpectPlaneValue(bytes, 0, 8 + x, 30.0F / 255.0F);
    }
}

TEST(YoloPreprocessingContractTest, OddPaddingRemainderFallsOnBottomAndRight) {
    cv::Mat image(1, 4, CV_8UC3, cv::Scalar(255, 255, 255));
    neuriplo_tasks::YoloPreprocessor preprocessor({4, 4});

    const auto bytes = preprocessor.preprocess(neuriplo_tasks::vision::opencv::toImageView(image));

    // Three vertical padding rows: integer centering puts one above and two below.
    for (std::size_t x = 0; x < 4; ++x) {
        ExpectPlaneValue(bytes, 0, x, 0.0F);
        ExpectPlaneValue(bytes, 0, 4 + x, 1.0F);
        ExpectPlaneValue(bytes, 0, 8 + x, 0.0F);
        ExpectPlaneValue(bytes, 0, 12 + x, 0.0F);
    }
}
