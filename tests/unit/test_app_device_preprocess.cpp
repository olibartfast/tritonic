#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "App.hpp"
#include "MockTriton.hpp"

// Client-side CUDA preprocessing (spec variant V2, plan T-13) through mocks: App must hand
// each frame to the device preprocessor, infer from the device input exactly once per frame,
// and never fall back to the host path (task preprocessing + ITriton::infer).

namespace {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

constexpr size_t kTensorBytes = 3 * 640 * 640 * sizeof(float);

class MockDeviceTriton : public MockTriton, public tritonic::triton::IDeviceInputInference {
public:
    MOCK_METHOD(void*, deviceInputBuffer, (size_t bytes), (override));
    MOCK_METHOD(std::vector<Tensor>, inferFromDeviceInput, (), (override));
};

class MockDevicePreprocessor : public tritonic::core::IDevicePreprocessor {
public:
    MOCK_METHOD(size_t, outputBytes, (), (const, override));
    MOCK_METHOD(void, preprocess, (const cv::Mat& frame, void* destination), (override));
};

TritonModelInfo Yolo26DetectionModel() {
    TritonModelInfo info;
    info.input_names = {"images"};
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_datatypes = {"FP32"};
    info.input_types = {CV_32F};
    info.output_names = {"output0"};
    info.output_datatypes = {"FP32"};
    info.output_shapes = {{1, 300, 6}};
    info.max_batch_size_ = 1;
    return info;
}

std::vector<Tensor> EmptyDetections() {
    return {Tensor(std::vector<tritonic::core::TensorElement>(300 * 6, 0.0f), {1, 300, 6})};
}

std::shared_ptr<InferenceConfig> DeviceConfig(const std::string& source) {
    auto config = std::make_shared<InferenceConfig>();
    config->SetModelName("yolo26det_trt");
    config->SetModelType("yolo26");
    config->SetLabelsFile("labels/coco.txt");
    config->SetSource(source);
    config->SetPreprocessDevice("cuda");
    config->SetSharedMemoryType("cuda");
    config->SetWriteFrame(false);
    return config;
}

std::shared_ptr<Logger> TestLogger() {
    return std::dynamic_pointer_cast<Logger>(LoggerManager::GetLogger("app_device_test"));
}

void ExpectServerReady(MockDeviceTriton& triton) {
    EXPECT_CALL(triton, isServerLive()).WillRepeatedly(Return(true));
    EXPECT_CALL(triton, isModelInRepository(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(triton, isModelReady(_, _)).WillRepeatedly(Return(true));
    EXPECT_CALL(triton, getModelInfo(_, _, _)).WillRepeatedly(Return(Yolo26DetectionModel()));
}

}  // namespace

TEST(AppDevicePreprocessTest, EachFrameIsPreprocessedOnDeviceAndInferredOnce) {
    auto triton = std::make_shared<NiceMock<MockDeviceTriton>>();
    auto preprocessor = std::make_shared<MockDevicePreprocessor>();
    ExpectServerReady(*triton);

    int region = 0;
    void* const regionPtr = &region;
    EXPECT_CALL(*preprocessor, outputBytes()).WillRepeatedly(Return(kTensorBytes));
    EXPECT_CALL(*triton, deviceInputBuffer(kTensorBytes))
        .Times(2)
        .WillRepeatedly(Return(regionPtr));
    EXPECT_CALL(*preprocessor, preprocess(_, regionPtr)).Times(2);
    EXPECT_CALL(*triton, inferFromDeviceInput()).Times(2).WillRepeatedly(Return(EmptyDetections()));
    EXPECT_CALL(*triton, infer(_)).Times(0);
    EXPECT_CALL(*triton, inferWithSharedMemory(_)).Times(0);

    App app(triton, DeviceConfig("data/images/bus.jpg,data/images/mug.jpg"), TestLogger());
    app.setDevicePreprocessor(preprocessor);
    EXPECT_EQ(app.run(), 0);
}

TEST(AppDevicePreprocessTest, FailsWithoutDevicePreprocessor) {
    auto triton = std::make_shared<NiceMock<MockDeviceTriton>>();
    ExpectServerReady(*triton);
    EXPECT_CALL(*triton, inferFromDeviceInput()).Times(0);
    EXPECT_CALL(*triton, infer(_)).Times(0);

    App app(triton, DeviceConfig("data/images/bus.jpg"), TestLogger());
    EXPECT_EQ(app.run(), 1);
}

TEST(AppDevicePreprocessTest, FailsWhenTritonClientLacksDeviceInput) {
    auto triton = std::make_shared<NiceMock<MockTriton>>();
    EXPECT_CALL(*triton, isServerLive()).WillRepeatedly(Return(true));
    EXPECT_CALL(*triton, isModelInRepository(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(*triton, isModelReady(_, _)).WillRepeatedly(Return(true));
    EXPECT_CALL(*triton, getModelInfo(_, _, _)).WillRepeatedly(Return(Yolo26DetectionModel()));
    EXPECT_CALL(*triton, infer(_)).Times(0);
    auto preprocessor = std::make_shared<MockDevicePreprocessor>();
    EXPECT_CALL(*preprocessor, preprocess(_, _)).Times(0);

    App app(triton, DeviceConfig("data/images/bus.jpg"), TestLogger());
    app.setDevicePreprocessor(preprocessor);
    EXPECT_EQ(app.run(), 1);
}

TEST(AppDevicePreprocessTest, CpuModeIgnoresInjectedPreprocessor) {
    auto triton = std::make_shared<NiceMock<MockDeviceTriton>>();
    ExpectServerReady(*triton);
    auto preprocessor = std::make_shared<MockDevicePreprocessor>();
    EXPECT_CALL(*preprocessor, preprocess(_, _)).Times(0);
    EXPECT_CALL(*triton, inferFromDeviceInput()).Times(0);
    EXPECT_CALL(*triton, infer(_)).Times(1).WillOnce(Return(EmptyDetections()));

    auto config = DeviceConfig("data/images/bus.jpg");
    config->SetPreprocessDevice("cpu");
    config->SetSharedMemoryType("none");
    App app(triton, config, TestLogger());
    app.setDevicePreprocessor(preprocessor);
    EXPECT_EQ(app.run(), 0);
}
