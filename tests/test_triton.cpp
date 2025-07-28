#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ITriton.hpp"
#include "mocks/MockTriton.hpp"

using ::testing::Return;
using ::testing::_;
using ::testing::InSequence;

class TritonInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        mockTriton = std::make_unique<MockTriton>();
    }
    
    void TearDown() override {
        mockTriton.reset();
    }
    
    std::unique_ptr<MockTriton> mockTriton;
};

TEST_F(TritonInterfaceTest, MockCanSetupModelInfoExpectations) {
    auto* mockPtr = mockTriton.get();
    
    // Set up model info expectation
    TritonModelInfo expectedModelInfo;
    expectedModelInfo.input_shapes = {{1, 3, 640, 640}};
    expectedModelInfo.input_names = {"images"};
    expectedModelInfo.output_names = {"output0"};
    
    EXPECT_CALL(*mockPtr, getModelInfo("test_model", "localhost:8000", _))
        .WillOnce(Return(expectedModelInfo));
    
    // Test the expectation
    auto modelInfo = mockPtr->getModelInfo("test_model", "localhost:8000", {});
    EXPECT_EQ(modelInfo.input_names[0], "images");
    EXPECT_EQ(modelInfo.output_names[0], "output0");
}

TEST_F(TritonInterfaceTest, MockCanSetupInferenceExpectations) {
    auto* mockPtr = mockTriton.get();
    
    // Set up inference expectation
    std::vector<std::vector<TensorElement>> expectedResults = {{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<int64_t>> expectedShapes = {{1, 3}};
    auto expectedTuple = std::make_tuple(expectedResults, expectedShapes);
    
    EXPECT_CALL(*mockPtr, infer(_))
        .WillOnce(Return(expectedTuple));
    
    // Test the expectation
    std::vector<std::vector<uint8_t>> inputData = {{1, 2, 3, 4}};
    auto [results, shapes] = mockPtr->infer(inputData);
    
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(shapes.size(), 1);
    EXPECT_EQ(shapes[0][1], 3);
}

TEST_F(TritonInterfaceTest, MockHandlesSetInputShapes) {
    auto* mockPtr = mockTriton.get();
    
    EXPECT_CALL(*mockPtr, setInputShapes(_))
        .Times(1);
    
    std::vector<std::vector<int64_t>> shapes = {{1, 3, 640, 640}};
    mockPtr->setInputShapes(shapes);
}

TEST_F(TritonInterfaceTest, MockHandlesSetInputShape) {
    auto* mockPtr = mockTriton.get();
    
    EXPECT_CALL(*mockPtr, setInputShape(_))
        .Times(1);
    
    std::vector<int64_t> shape = {1, 3, 640, 640};
    mockPtr->setInputShape(shape);
}

TEST_F(TritonInterfaceTest, MockHandlesPrintModelInfo) {
    auto* mockPtr = mockTriton.get();
    
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"test_input"};
    
    EXPECT_CALL(*mockPtr, printModelInfo(_))
        .Times(1);
    
    mockPtr->printModelInfo(modelInfo);
}

TEST_F(TritonInterfaceTest, MockHandlesCreateTritonClient) {
    auto* mockPtr = mockTriton.get();
    
    EXPECT_CALL(*mockPtr, createTritonClient())
        .Times(1);
    
    mockPtr->createTritonClient();
}

// Test that we can create multiple mock clients with different behaviors
TEST_F(TritonInterfaceTest, CanCreateMultipleMockClientsWithDifferentBehaviors) {
    auto mockClient1 = std::make_unique<MockTriton>();
    auto mockClient2 = std::make_unique<MockTriton>();
    
    auto* mock1Ptr = mockClient1.get();
    auto* mock2Ptr = mockClient2.get();
    
    // Set up different expectations for each client
    TritonModelInfo model1Info;
    model1Info.input_names = {"model1_input"};
    
    TritonModelInfo model2Info;
    model2Info.input_names = {"model2_input"};
    
    EXPECT_CALL(*mock1Ptr, getModelInfo("model1", _, _))
        .WillOnce(Return(model1Info));
    
    EXPECT_CALL(*mock2Ptr, getModelInfo("model2", _, _))
        .WillOnce(Return(model2Info));
    
    // Test that each client behaves differently
    auto info1 = mock1Ptr->getModelInfo("model1", "url", {});
    auto info2 = mock2Ptr->getModelInfo("model2", "url", {});
    
    EXPECT_EQ(info1.input_names[0], "model1_input");
    EXPECT_EQ(info2.input_names[0], "model2_input");
}
