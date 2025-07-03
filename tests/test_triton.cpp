#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "TritonClientAdapter.hpp"
#include "mocks/MockTritonClient.hpp"

using ::testing::Return;
using ::testing::_;
using ::testing::InSequence;

class TritonClientAdapterTest : public ::testing::Test {
protected:
    void SetUp() override {
        mockFactory = std::make_unique<MockTritonClientFactory>();
    }
    
    void TearDown() override {
        mockFactory.reset();
    }
    
    std::unique_ptr<MockTritonClientFactory> mockFactory;
};

TEST_F(TritonClientAdapterTest, FactoryCreatesClientWithCorrectParameters) {
    auto mockClient = std::make_unique<MockTritonClient>();
    
    EXPECT_CALL(*mockFactory, createClient("localhost:8000", 0, "test_model", "1", false))
        .WillOnce(Return(testing::ByMove(std::move(mockClient))));
    
    auto client = mockFactory->createClient("localhost:8000", 0, "test_model", "1", false);
    
    ASSERT_NE(client, nullptr);
}

TEST_F(TritonClientAdapterTest, MockClientCanSetupExpectations) {
    auto mockClient = std::make_unique<MockTritonClient>();
    auto* mockClientPtr = mockClient.get();
    
    // Set up model info expectation
    TritonModelInfo expectedModelInfo;
    expectedModelInfo.input_shapes = {{1, 3, 640, 640}};
    expectedModelInfo.input_names = {"images"};
    expectedModelInfo.output_names = {"output0"};
    
    EXPECT_CALL(*mockClientPtr, getModelInfo("test_model", "localhost:8000", _))
        .WillOnce(Return(expectedModelInfo));
    
    // Test the expectation
    auto modelInfo = mockClientPtr->getModelInfo("test_model", "localhost:8000", {});
    EXPECT_EQ(modelInfo.input_names[0], "images");
    EXPECT_EQ(modelInfo.output_names[0], "output0");
}

TEST_F(TritonClientAdapterTest, MockClientCanSetupInferenceExpectations) {
    auto mockClient = std::make_unique<MockTritonClient>();
    auto* mockClientPtr = mockClient.get();
    
    // Set up inference expectation
    std::vector<std::vector<TensorElement>> expectedResults = {{1.0f, 2.0f, 3.0f}};
    std::vector<std::vector<int64_t>> expectedShapes = {{1, 3}};
    auto expectedTuple = std::make_tuple(expectedResults, expectedShapes);
    
    EXPECT_CALL(*mockClientPtr, infer(_))
        .WillOnce(Return(expectedTuple));
    
    // Test the expectation
    std::vector<std::vector<uint8_t>> inputData = {{1, 2, 3, 4}};
    auto [results, shapes] = mockClientPtr->infer(inputData);
    
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(shapes.size(), 1);
    EXPECT_EQ(shapes[0][1], 3);
}

TEST_F(TritonClientAdapterTest, MockClientHandlesSetInputShapes) {
    auto mockClient = std::make_unique<MockTritonClient>();
    auto* mockClientPtr = mockClient.get();
    
    EXPECT_CALL(*mockClientPtr, setInputShapes(_))
        .Times(1);
    
    std::vector<std::vector<int64_t>> shapes = {{1, 3, 640, 640}};
    mockClientPtr->setInputShapes(shapes);
}

TEST_F(TritonClientAdapterTest, MockClientHandlesPrintModelInfo) {
    auto mockClient = std::make_unique<MockTritonClient>();
    auto* mockClientPtr = mockClient.get();
    
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"test_input"};
    
    EXPECT_CALL(*mockClientPtr, printModelInfo(_))
        .Times(1);
    
    mockClientPtr->printModelInfo(modelInfo);
}

// Test that we can create multiple mock clients with different behaviors
TEST_F(TritonClientAdapterTest, CanCreateMultipleMockClientsWithDifferentBehaviors) {
    auto mockClient1 = std::make_unique<MockTritonClient>();
    auto mockClient2 = std::make_unique<MockTritonClient>();
    
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
