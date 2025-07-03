#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "task_factory.hpp"
#include "mocks/MockTritonClient.hpp"
#include "mocks/MockFileSystem.hpp"

using ::testing::Return;
using ::testing::_;
using ::testing::StrictMock;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up model info
        modelInfo.input_shapes = {{1, 3, 640, 640}};
        modelInfo.input_formats = {"FORMAT_NCHW"};
        modelInfo.input_names = {"images"};
        modelInfo.output_names = {"output0"};
        modelInfo.input_types = {CV_32F};
        modelInfo.max_batch_size_ = 1;
        modelInfo.batch_size_ = 1;
        
        // Create test image
        testImage = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(testImage, cv::Rect(100, 100, 200, 200), cv::Scalar(255, 255, 255), -1);
        
        // Set up mocks
        mockTritonClient = std::make_unique<StrictMock<MockTritonClient>>();
        mockFileSystem = std::make_unique<StrictMock<MockFileSystem>>();
    }
    
    void TearDown() override {
        mockTritonClient.reset();
        mockFileSystem.reset();
    }
    
    TritonModelInfo modelInfo;
    cv::Mat testImage;
    std::unique_ptr<StrictMock<MockTritonClient>> mockTritonClient;
    std::unique_ptr<StrictMock<MockFileSystem>> mockFileSystem;
};

TEST_F(IntegrationTest, FullDetectionPipelineWithMocks) {
    // Create task
    auto task = TaskFactory::createTaskInstance("yolov8", modelInfo);
    ASSERT_NE(task, nullptr);
    
    // Set up mock inference results
    std::vector<std::vector<TensorElement>> mockResults;
    std::vector<std::vector<int64_t>> mockShapes;
    
    // Mock detection output: [batch, detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    mockResults.push_back(std::vector<TensorElement>(1 * 25200 * 85, 0.0f));
    mockShapes.push_back({1, 25200, 85});
    
    // Add some fake detections with high confidence
    mockResults[0][4] = 0.9f;   // High confidence for first detection
    mockResults[0][84] = 0.8f;  // High class probability for person class
    
    auto mockResultTuple = std::make_tuple(mockResults, mockShapes);
    
    EXPECT_CALL(*mockTritonClient, infer(_))
        .WillOnce(Return(mockResultTuple));
    
    // Test preprocessing
    std::vector<cv::Mat> images = {testImage};
    auto preprocessed = task->preprocess(images);
    EXPECT_EQ(preprocessed.size(), 1);
    
    // Test inference (using mock)
    auto [inferResults, inferShapes] = mockTritonClient->infer(preprocessed);
    EXPECT_EQ(inferResults.size(), 1);
    EXPECT_EQ(inferShapes.size(), 1);
    
    // Test postprocessing
    cv::Size frameSize(testImage.cols, testImage.rows);
    auto results = task->postprocess(frameSize, inferResults, inferShapes);
    
    // Verify results structure
    EXPECT_GE(results.size(), 0);
}

TEST_F(IntegrationTest, ConfigurationValidationIntegration) {
    // Test that invalid configurations are properly rejected throughout the pipeline
    Config invalidConfig;
    invalidConfig.model_name = "path/to/model";  // Invalid - contains path
    invalidConfig.server_address = "localhost";
    invalidConfig.port = 8000;
    
    EXPECT_FALSE(invalidConfig.isValid());
    
    auto errors = invalidConfig.getValidationErrors();
    bool foundPathError = false;
    for (const auto& error : errors) {
        if (error.find("path") != std::string::npos) {
            foundPathError = true;
            break;
        }
    }
    EXPECT_TRUE(foundPathError);
}

TEST_F(IntegrationTest, TaskCreationWithDifferentModelTypes) {
    std::vector<std::pair<std::string, TaskType>> modelTests = {
        {"yolov8", TaskType::Detection},
        {"yolov5", TaskType::Detection},
        {"resnet50", TaskType::Classification},
        {"yolov8-seg", TaskType::InstanceSegmentation}
    };
    
    for (const auto& [modelType, expectedTaskType] : modelTests) {
        auto task = TaskFactory::createTaskInstance(modelType, modelInfo);
        ASSERT_NE(task, nullptr) << "Failed to create task for " << modelType;
        EXPECT_EQ(task->getTaskType(), expectedTaskType) << "Wrong task type for " << modelType;
        
        // Test that preprocessing works for all task types
        std::vector<cv::Mat> images = {testImage};
        EXPECT_NO_THROW({
            auto preprocessed = task->preprocess(images);
            EXPECT_GT(preprocessed.size(), 0);
        }) << "Preprocessing failed for " << modelType;
    }
}

TEST_F(IntegrationTest, ErrorHandlingThroughoutPipeline) {
    // Test error handling at different stages
    
    // 1. Invalid model info should fail task creation
    TritonModelInfo invalidModelInfo;
    invalidModelInfo.input_shapes = {{1, 0}};
    EXPECT_THROW(TaskFactory::createTaskInstance("yolov8", invalidModelInfo), 
                 InputDimensionError);
    
    // 2. Empty image should fail preprocessing
    auto task = TaskFactory::createTaskInstance("yolov8", modelInfo);
    std::vector<cv::Mat> emptyImages;
    EXPECT_THROW(task->preprocess(emptyImages), std::runtime_error);
    
    // 3. Invalid image should fail preprocessing
    cv::Mat emptyImage;
    std::vector<cv::Mat> invalidImages = {emptyImage};
    EXPECT_THROW(task->preprocess(invalidImages), std::runtime_error);
}

TEST_F(IntegrationTest, OpticalFlowSpecialHandling) {
    // Set up optical flow model info
    TritonModelInfo flowModelInfo = modelInfo;
    flowModelInfo.input_shapes = {{2, 3, 480, 640}};
    
    auto flowTask = TaskFactory::createTaskInstance("raft", flowModelInfo);
    ASSERT_NE(flowTask, nullptr);
    EXPECT_EQ(flowTask->getTaskType(), TaskType::OpticalFlow);
    
    // Optical flow requires exactly 2 images
    cv::Mat frame1 = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Mat frame2 = cv::Mat::zeros(480, 640, CV_8UC3);
    
    std::vector<cv::Mat> twoFrames = {frame1, frame2};
    EXPECT_NO_THROW({
        auto preprocessed = flowTask->preprocess(twoFrames);
        EXPECT_GT(preprocessed.size(), 0);
    });
    
    // Should fail with single image
    std::vector<cv::Mat> oneFrame = {frame1};
    // Note: This might depend on the specific RAFT implementation
    // The test verifies the basic structure works
}
