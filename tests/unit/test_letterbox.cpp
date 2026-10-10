#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"
#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"
#include "tritonic/cuda/letterbox_pixel.hpp"

// The CUDA kernel and LetterboxHost share letterbox_pixel.hpp, so these CPU tests cover the
// arithmetic the kernel runs. GPU execution itself is covered by spec V-5 / V-14.

using tritonic::cuda::ComputeLetterboxGeometry;
using tritonic::cuda::PadRounding;

TEST(LetterboxGeometryTest, LandscapeFillsWidthAndCentersVertically) {
    const auto geometry = ComputeLetterboxGeometry(1080, 1920);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->resized_width, 640);
    EXPECT_EQ(geometry->resized_height, 360);
    EXPECT_EQ(geometry->left, 0);
    EXPECT_EQ(geometry->top, 140);
}

TEST(LetterboxGeometryTest, PortraitFillsHeight) {
    const auto geometry = ComputeLetterboxGeometry(1080, 810);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->resized_height, 640);
    EXPECT_EQ(geometry->resized_width, 480);
    EXPECT_EQ(geometry->top, 0);
    EXPECT_EQ(geometry->left, 80);
}

TEST(LetterboxGeometryTest, OddPaddingFollowsRoundingRule) {
    // 481x640 keeps 481 rows: 159 padding rows split 79/80 (floor) or 80/79 (ceil).
    const auto floorGeometry = ComputeLetterboxGeometry(481, 640, 640, PadRounding::Floor);
    const auto ceilGeometry = ComputeLetterboxGeometry(481, 640, 640, PadRounding::Ceil);
    ASSERT_TRUE(floorGeometry && ceilGeometry);
    EXPECT_EQ(floorGeometry->resized_height, 481);
    EXPECT_EQ(floorGeometry->top, 79);
    EXPECT_EQ(ceilGeometry->top, 80);
}

TEST(LetterboxGeometryTest, RejectsDegenerateAndInvalidSizes) {
    EXPECT_FALSE(ComputeLetterboxGeometry(1, 1000));
    EXPECT_FALSE(ComputeLetterboxGeometry(1000, 1));
    EXPECT_FALSE(ComputeLetterboxGeometry(0, 10));
    EXPECT_FALSE(ComputeLetterboxGeometry(10, -1));
    EXPECT_TRUE(ComputeLetterboxGeometry(1, 640));
}

TEST(LetterboxHostTest, ConstantImageProducesExactPlanesAndZeroPadding) {
    const int height = 100;
    const int width = 200;
    std::vector<uint8_t> image(static_cast<size_t>(height) * width * 3);
    for (size_t i = 0; i < image.size(); i += 3) {
        image[i] = 10;       // B
        image[i + 1] = 20;   // G
        image[i + 2] = 255;  // R
    }
    std::vector<float> tensor(3 * 640 * 640, -1.0f);
    ASSERT_TRUE(
        tritonic::cuda::LetterboxHost(image.data(), height, width, width * 3, true, tensor.data()));
    const size_t plane = 640 * 640;
    // Content occupies rows [160, 480); padding rows are exactly zero in every channel.
    EXPECT_FLOAT_EQ(tensor[0 * plane + 300 * 640 + 320], 1.0f);
    EXPECT_FLOAT_EQ(tensor[1 * plane + 300 * 640 + 320], 20.0f / 255.0f);
    EXPECT_FLOAT_EQ(tensor[2 * plane + 300 * 640 + 320], 10.0f / 255.0f);
    for (int channel = 0; channel < 3; ++channel) {
        EXPECT_FLOAT_EQ(tensor[channel * plane + 159 * 640 + 320], 0.0f);
        EXPECT_FLOAT_EQ(tensor[channel * plane + 480 * 640 + 320], 0.0f);
        EXPECT_FLOAT_EQ(tensor[channel * plane + 160 * 640 + 320],
                        channel == 0 ? 1.0f : (channel == 1 ? 20.0f : 10.0f) / 255.0f);
    }
}

TEST(LetterboxHostTest, RejectsShortPitch) {
    std::vector<uint8_t> image(30);
    std::vector<float> tensor(3 * 640 * 640);
    EXPECT_FALSE(tritonic::cuda::LetterboxHost(image.data(), 2, 5, 14, false, tensor.data()));
}

// The CPU path (neuriplo-tasks) passed the DALI parity gate, so the shared kernel math
// must meet the same spec D-5 tolerances against it on the stock fixtures.
TEST(LetterboxHostTest, MatchesNeuriploTasksYoloPreprocessor) {
    neuriplo_tasks::YoloPreprocessor preprocessor({640, 640});
    for (const char* name : {"bus.jpg", "horses.jpg", "person.jpg", "mug.jpg"}) {
        SCOPED_TRACE(name);
        const cv::Mat image = cv::imread(std::string("data/images/") + name);
        ASSERT_FALSE(image.empty());

        const auto bytes =
            preprocessor.preprocess(neuriplo_tasks::vision::opencv::toImageView(image));
        ASSERT_EQ(bytes.size(), 3u * 640 * 640 * sizeof(float));
        const auto* expected = reinterpret_cast<const float*>(bytes.data());

        std::vector<float> actual(3 * 640 * 640);
        ASSERT_TRUE(tritonic::cuda::LetterboxHost(image.data, image.rows, image.cols, image.step,
                                                  true, actual.data()));
        double sum = 0.0;
        float maxAbs = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float diff = std::fabs(actual[i] - expected[i]);
            sum += diff;
            maxAbs = std::max(maxAbs, diff);
        }
        EXPECT_LE(maxAbs, 0.15f);
        EXPECT_LE(sum / static_cast<double>(actual.size()), 0.003);
    }
}
