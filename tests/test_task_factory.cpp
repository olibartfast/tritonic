#include <gtest/gtest.h>
#include "task_factory.hpp"

class TaskFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a valid model info
        modelInfo.input_shapes = {{1, 3, 640, 640}};
        modelInfo.input_formats = {"FORMAT_NCHW"};
        modelInfo.input_names = {"images"};
        modelInfo.output_names = {"output0"};
        modelInfo.input_types = {CV_32F};
        modelInfo.max_batch_size_ = 1;
        modelInfo.batch_size_ = 1;
    }
    
    void TearDown() override {
        // Clean up
    }
    
    TritonModelInfo modelInfo;
};

TEST_F(TaskFactoryTest, CreateYOLOTaskInstance) {
    auto task = TaskFactory::createTaskInstance("yolov8", modelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, CreateYOLOv5TaskInstance) {
    auto task = TaskFactory::createTaskInstance("yolov5", modelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, CreateYOLOv10TaskInstance) {
    auto task = TaskFactory::createTaskInstance("yolov10", modelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, CreateRTDetrTaskInstance) {
    auto task = TaskFactory::createTaskInstance("rtdetr", modelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, CreateClassificationTaskInstance) {
    // Modify model info for classification
    TritonModelInfo classModelInfo = modelInfo;
    classModelInfo.input_shapes = {{1, 3, 224, 224}};
    
    auto task = TaskFactory::createTaskInstance("resnet50", classModelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Classification);
}

TEST_F(TaskFactoryTest, CreateSegmentationTaskInstance) {
    auto task = TaskFactory::createTaskInstance("yolov8-seg", modelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::InstanceSegmentation);
}

TEST_F(TaskFactoryTest, CreateOpticalFlowTaskInstance) {
    // Modify model info for optical flow (needs 2 frames)
    TritonModelInfo flowModelInfo = modelInfo;
    flowModelInfo.input_shapes = {{2, 3, 480, 640}};
    
    auto task = TaskFactory::createTaskInstance("raft", flowModelInfo);
    
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::OpticalFlow);
}

TEST_F(TaskFactoryTest, CreateTaskWithInvalidModelType) {
    EXPECT_THROW(TaskFactory::createTaskInstance("invalid_model", modelInfo), 
                 std::invalid_argument);
}

TEST_F(TaskFactoryTest, CreateTaskWithEmptyModelType) {
    EXPECT_THROW(TaskFactory::createTaskInstance("", modelInfo), 
                 std::invalid_argument);
}

TEST_F(TaskFactoryTest, CreateTaskWithInvalidModelInfo) {
    TritonModelInfo invalidModelInfo;
    invalidModelInfo.input_shapes = {{1, 0}};  // Invalid dimensions
    invalidModelInfo.input_formats = {"FORMAT_NCHW"};
    
    EXPECT_THROW(TaskFactory::createTaskInstance("yolov8", invalidModelInfo), 
                 InputDimensionError);
}

// Test case-insensitive model type handling
TEST_F(TaskFactoryTest, HandlesCaseInsensitiveModelTypes) {
    auto task1 = TaskFactory::createTaskInstance("YOLOV8", modelInfo);
    auto task2 = TaskFactory::createTaskInstance("yolov8", modelInfo);
    auto task3 = TaskFactory::createTaskInstance("YoLoV8", modelInfo);
    
    ASSERT_NE(task1, nullptr);
    ASSERT_NE(task2, nullptr);
    ASSERT_NE(task3, nullptr);
    
    EXPECT_EQ(task1->getTaskType(), TaskType::Detection);
    EXPECT_EQ(task2->getTaskType(), TaskType::Detection);
    EXPECT_EQ(task3->getTaskType(), TaskType::Detection);
}

// Test that all supported model types can be created
TEST_F(TaskFactoryTest, AllSupportedModelTypesCanBeCreated) {
    std::vector<std::string> detectionModels = {
        "yolov8", "yolov5", "yolov10", "yolo-nas", 
        "rtdetr", "rtdetr-ultralytics", "rf-detr"
    };
    
    for (const auto& modelType : detectionModels) {
        EXPECT_NO_THROW({
            auto task = TaskFactory::createTaskInstance(modelType, modelInfo);
            EXPECT_NE(task, nullptr);
            EXPECT_EQ(task->getTaskType(), TaskType::Detection);
        }) << "Failed to create task for model type: " << modelType;
    }
}
