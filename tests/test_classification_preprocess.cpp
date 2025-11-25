#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "vision-core/classification/classification_preprocessor.hpp"

namespace {

cv::Mat makeImage(int value) {
    return cv::Mat(224, 224, CV_8UC3, cv::Scalar(value, value, value));
}

} // namespace

TEST(TorchvisionPreprocessorTest, ProcessesBatchedInputs) {
    vision_core::TorchvisionPreprocessor preprocessor(cv::Size(224, 224));

    std::vector<cv::Mat> imgs = {makeImage(10), makeImage(20)};
    auto data = preprocessor.preprocess(imgs);

    // New API returns a vector of buffers, one per image
    ASSERT_EQ(data.size(), 2);
    const size_t expected_single_size = static_cast<size_t>(224) * 224 * 3 * sizeof(float);
    EXPECT_EQ(data[0].size(), expected_single_size);
    EXPECT_EQ(data[1].size(), expected_single_size);
}

TEST(TensorflowPreprocessorTest, ProcessesBatchedInputs) {
    vision_core::TensorflowPreprocessor preprocessor(cv::Size(224, 224));

    std::vector<cv::Mat> imgs = {makeImage(5), makeImage(15)};
    auto data = preprocessor.preprocess(imgs);

    ASSERT_EQ(data.size(), 2);
    // TensorflowPreprocessor uses UINT8
    const size_t expected_single_size = static_cast<size_t>(224) * 224 * 3 * sizeof(uint8_t);
    EXPECT_EQ(data[0].size(), expected_single_size);
    EXPECT_EQ(data[1].size(), expected_single_size);
}
