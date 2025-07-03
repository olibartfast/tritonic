#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "YOLO.hpp"
#include <opencv2/opencv.hpp>

class YOLOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a valid model info for YOLO
        modelInfo.input_shapes = {{1, 3, 640, 640}};
        modelInfo.input_formats = {"FORMAT_NCHW"};
        modelInfo.input_names = {"images"};
        modelInfo.output_names = {"output0"};
        modelInfo.input_types = {CV_32F};
        modelInfo.max_batch_size_ = 1;
        modelInfo.batch_size_ = 1;
        
        // Create test image
        testImage = cv::Mat::zeros(480, 640, CV_8UC3);
        // Add some test data to the image
        cv::rectangle(testImage, cv::Rect(100, 100, 200, 200), cv::Scalar(255, 255, 255), -1);
    }
    
    void TearDown() override {
        // Clean up
    }
    
    TritonModelInfo modelInfo;
    cv::Mat testImage;
};

TEST_F(YOLOTest, ConstructorWithValidModelInfo) {
    EXPECT_NO_THROW(YOLO yolo(modelInfo));
}

TEST_F(YOLOTest, ConstructorWithInvalidModelInfo) {
    TritonModelInfo invalidModel;
    invalidModel.input_shapes = {{1, 0}};  // Invalid dimensions
    invalidModel.input_formats = {"FORMAT_NCHW"};
    
    EXPECT_THROW(YOLO yolo(invalidModel), InputDimensionError);
}

TEST_F(YOLOTest, PreprocessWithValidImage) {
    YOLO yolo(modelInfo);
    std::vector<cv::Mat> images = {testImage};
    
    EXPECT_NO_THROW({
        auto preprocessed = yolo.preprocess(images);
        EXPECT_EQ(preprocessed.size(), 1);  // Should match number of inputs
        // Check that the preprocessed data has the correct size
        // For 640x640x3 input with float32, should be 640*640*3*4 bytes
        EXPECT_EQ(preprocessed[0].size(), 640 * 640 * 3 * sizeof(float));
    });
}

TEST_F(YOLOTest, PreprocessWithEmptyImageVector) {
    YOLO yolo(modelInfo);
    std::vector<cv::Mat> emptyImages;
    
    EXPECT_THROW(yolo.preprocess(emptyImages), std::runtime_error);
}

TEST_F(YOLOTest, PreprocessWithEmptyImage) {
    YOLO yolo(modelInfo);
    cv::Mat emptyImage;
    std::vector<cv::Mat> images = {emptyImage};
    
    EXPECT_THROW(yolo.preprocess(images), std::runtime_error);
}

TEST_F(YOLOTest, GetRectCorrectlyMapsCoordinates) {
    YOLO yolo(modelInfo);
    cv::Size imageSize(1280, 720);  // Different from model input size
    std::vector<float> bbox = {320, 240, 100, 80};  // center_x, center_y, width, height
    
    cv::Rect result = yolo.get_rect(imageSize, bbox);
    
    // Verify the result is reasonable
    EXPECT_GE(result.x, 0);
    EXPECT_GE(result.y, 0);
    EXPECT_GT(result.width, 0);
    EXPECT_GT(result.height, 0);
}

TEST_F(YOLOTest, PostprocessWithValidResults) {
    YOLO yolo(modelInfo);
    cv::Size frameSize(640, 640);
    
    // Create mock inference results (simplified)
    std::vector<std::vector<TensorElement>> inferResults;
    std::vector<std::vector<int64_t>> inferShapes;
    
    // Mock detection output: [batch, detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    inferResults.push_back(std::vector<TensorElement>(1 * 25200 * 85, 0.0f));
    inferShapes.push_back({1, 25200, 85});
    
    // Add a fake detection with high confidence
    inferResults[0][4] = 0.9f;  // High confidence
    inferResults[0][84] = 0.8f; // High class probability for last class
    
    EXPECT_NO_THROW({
        auto results = yolo.postprocess(frameSize, inferResults, inferShapes);
        // Results should be a vector of Detection objects
        EXPECT_GE(results.size(), 0);
    });
}

TEST_F(YOLOTest, GetTaskTypeReturnsDetection) {
    YOLO yolo(modelInfo);
    EXPECT_EQ(yolo.getTaskType(), TaskType::Detection);
}

// Test the coordinate mapping with different aspect ratios
TEST_F(YOLOTest, GetRectHandlesDifferentAspectRatios) {
    YOLO yolo(modelInfo);
    
    // Test with wider image
    cv::Size wideImage(1920, 1080);
    std::vector<float> bbox = {400, 300, 100, 80};
    cv::Rect result1 = yolo.get_rect(wideImage, bbox);
    EXPECT_GT(result1.width, 0);
    EXPECT_GT(result1.height, 0);
    
    // Test with taller image
    cv::Size tallImage(480, 854);
    cv::Rect result2 = yolo.get_rect(tallImage, bbox);
    EXPECT_GT(result2.width, 0);
    EXPECT_GT(result2.height, 0);
}
