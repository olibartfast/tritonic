#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ITriton.hpp"
#include "mocks/MockTriton.hpp"

using ::testing::Return;
using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;

class TritonInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use NiceMock to avoid warnings about uninteresting calls
        mockTriton = std::make_unique<NiceMock<MockTriton>>();
    }
    
    void TearDown() override {
        mockTriton.reset();
    }
    
    std::unique_ptr<NiceMock<MockTriton>> mockTriton;
};

TEST_F(TritonInterfaceTest, MockCanSetupModelInfoExpectations) {
    // Set up model info expectation
    TritonModelInfo expectedModelInfo;
    expectedModelInfo.input_shapes = {{1, 3, 640, 640}};
    expectedModelInfo.input_names = {"images"};
    expectedModelInfo.output_names = {"output0"};
    
    EXPECT_CALL(*mockTriton, getModelInfo("test_model", "localhost:8000", _))
        .WillOnce(Return(expectedModelInfo));
    
    // Test the expectation
    auto modelInfo = mockTriton->getModelInfo("test_model", "localhost:8000", {});
    EXPECT_EQ(modelInfo.input_names[0], "images");
    EXPECT_EQ(modelInfo.output_names[0], "output0");
}

TEST_F(TritonInterfaceTest, MockCanSetupInferenceExpectations) {
    // Set up inference expectation
    std::vector<std::vector<TensorElement>> expectedResults = {{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<int64_t>> expectedShapes = {{1, 3}};
    auto expectedTuple = std::make_tuple(expectedResults, expectedShapes);
    
    EXPECT_CALL(*mockTriton, infer(_))
        .WillOnce(Return(expectedTuple));
    
    // Test the expectation
    std::vector<std::vector<uint8_t>> inputData = {{1, 2, 3, 4}};
    auto [results, shapes] = mockTriton->infer(inputData);
    
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(shapes.size(), 1);
    EXPECT_EQ(shapes[0][1], 3);
}

TEST_F(TritonInterfaceTest, MockHandlesSetInputShapes) {
    EXPECT_CALL(*mockTriton, setInputShapes(_))
        .Times(1);
    
    std::vector<std::vector<int64_t>> shapes = {{1, 3, 640, 640}};
    mockTriton->setInputShapes(shapes);
}

TEST_F(TritonInterfaceTest, MockHandlesSetInputShape) {
    EXPECT_CALL(*mockTriton, setInputShape(_))
        .Times(1);
    
    std::vector<int64_t> shape = {1, 3, 640, 640};
    mockTriton->setInputShape(shape);
}

TEST_F(TritonInterfaceTest, MockHandlesPrintModelInfo) {
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"test_input"};
    
    EXPECT_CALL(*mockTriton, printModelInfo(_))
        .Times(1);
    
    mockTriton->printModelInfo(modelInfo);
}

TEST_F(TritonInterfaceTest, MockHandlesCreateTritonClient) {
    EXPECT_CALL(*mockTriton, createTritonClient())
        .Times(1);
    
    mockTriton->createTritonClient();
}

// Test that we can create multiple mock clients with different behaviors
TEST_F(TritonInterfaceTest, CanCreateMultipleMockClientsWithDifferentBehaviors) {
    auto mockClient1 = std::make_unique<MockTriton>();
    auto mockClient2 = std::make_unique<MockTriton>();
    
    // Set up different expectations for each client
    TritonModelInfo model1Info;
    model1Info.input_names = {"model1_input"};
    
    TritonModelInfo model2Info;
    model2Info.input_names = {"model2_input"};
    
    EXPECT_CALL(*mockClient1, getModelInfo("model1", _, _))
        .WillOnce(Return(model1Info));
    
    EXPECT_CALL(*mockClient2, getModelInfo("model2", _, _))
        .WillOnce(Return(model2Info));
    
    // Test that each client behaves differently
    auto info1 = mockClient1->getModelInfo("model1", "url", {});
    auto info2 = mockClient2->getModelInfo("model2", "url", {});
    
    EXPECT_EQ(info1.input_names[0], "model1_input");
    EXPECT_EQ(info2.input_names[0], "model2_input");
}
