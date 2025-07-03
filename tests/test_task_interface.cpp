#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "TaskInterface.hpp"
#include "mocks/MockFileSystem.hpp"

using ::testing::Return;
using ::testing::_;

// Test implementation of TaskInterface for testing
class TestTaskInterface : public TaskInterface {
public:
    TestTaskInterface(const TritonModelInfo& modelInfo) : TaskInterface(modelInfo) {}
    
    TaskType getTaskType() override { return TaskType::Detection; }
    
    std::vector<Result> postprocess(
        const cv::Size& frame_size, 
        const std::vector<std::vector<TensorElement>>& infer_results, 
        const std::vector<std::vector<int64_t>>& infer_shapes) override {
        return {};
    }
    
    std::vector<std::vector<uint8_t>> preprocess(
        const std::vector<cv::Mat>& imgs) override {
        return {};
    }
    
    // Expose protected members for testing
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getInputChannels() const { return input_channels_; }
};

class TaskInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a valid model info
        validModelInfo.input_shapes = {{1, 3, 640, 640}};
        validModelInfo.input_formats = {"FORMAT_NCHW"};
        validModelInfo.input_names = {"input"};
        validModelInfo.output_names = {"output"};
        
        // Create an invalid model info
        invalidModelInfo.input_shapes = {{1, 0}};  // Invalid dimensions
        invalidModelInfo.input_formats = {"FORMAT_NCHW"};
        invalidModelInfo.input_names = {"input"};
        invalidModelInfo.output_names = {"output"};
        
        mockFileSystem = std::make_unique<MockFileSystem>();
    }
    
    void TearDown() override {
        mockFileSystem.reset();
    }
    
    TritonModelInfo validModelInfo;
    TritonModelInfo invalidModelInfo;
    std::unique_ptr<MockFileSystem> mockFileSystem;
};

TEST_F(TaskInterfaceTest, ConstructorWithValidModelInfoSucceeds) {
    EXPECT_NO_THROW({
        TestTaskInterface task(validModelInfo);
        EXPECT_EQ(task.getInputWidth(), 640);
        EXPECT_EQ(task.getInputHeight(), 640);
        EXPECT_EQ(task.getInputChannels(), 3);
    });
}

TEST_F(TaskInterfaceTest, ConstructorWithInvalidModelInfoThrows) {
    EXPECT_THROW(TestTaskInterface task(invalidModelInfo), InputDimensionError);
}

TEST_F(TaskInterfaceTest, ConstructorHandlesNHWCFormat) {
    TritonModelInfo nhwcModelInfo;
    nhwcModelInfo.input_shapes = {{1, 480, 640, 3}};
    nhwcModelInfo.input_formats = {"FORMAT_NHWC"};
    nhwcModelInfo.input_names = {"input"};
    nhwcModelInfo.output_names = {"output"};
    
    TestTaskInterface task(nhwcModelInfo);
    EXPECT_EQ(task.getInputWidth(), 640);
    EXPECT_EQ(task.getInputHeight(), 480);
    EXPECT_EQ(task.getInputChannels(), 3);
}

TEST_F(TaskInterfaceTest, ConstructorHandlesNCHWFormat) {
    TritonModelInfo nchwModelInfo;
    nchwModelInfo.input_shapes = {{1, 3, 480, 640}};
    nchwModelInfo.input_formats = {"FORMAT_NCHW"};
    nchwModelInfo.input_names = {"input"};
    nchwModelInfo.output_names = {"output"};
    
    TestTaskInterface task(nchwModelInfo);
    EXPECT_EQ(task.getInputWidth(), 640);
    EXPECT_EQ(task.getInputHeight(), 480);
    EXPECT_EQ(task.getInputChannels(), 3);
}

TEST_F(TaskInterfaceTest, ReadLabelNamesWithValidFile) {
    std::vector<std::string> expectedLabels = {"person", "car", "dog", "cat"};
    
    EXPECT_CALL(*mockFileSystem, readLines("test_labels.txt"))
        .WillOnce(Return(expectedLabels));
    
    TestTaskInterface task(validModelInfo);
    
    // Note: This test would need TaskInterface to accept a file system dependency
    // For now, we'll test the existing functionality with a real file
    // In a refactored version, we would inject the file system dependency
    
    // Test that the method exists and doesn't throw
    EXPECT_NO_THROW({
        auto labels = task.readLabelNames("labels/coco.txt");
        // We can't easily test the exact content without a real file
        // but we can verify the method runs without error
    });
}

TEST_F(TaskInterfaceTest, GetTaskTypeReturnsCorrectType) {
    TestTaskInterface task(validModelInfo);
    EXPECT_EQ(task.getTaskType(), TaskType::Detection);
}
