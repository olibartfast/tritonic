#include <gtest/gtest.h>
#include "ITriton.hpp"
#include "TritonModelInfo.hpp"

// Basic tests to verify Triton interface and data types work correctly
// No concrete implementation - just interface and data structure tests

TEST(TritonInterfaceTest, TritonModelInfoCanBeCreated) {
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"test_input"};
    modelInfo.output_names = {"test_output"};
    modelInfo.input_shapes = {{1, 3, 640, 640}};

    EXPECT_EQ(modelInfo.input_names.size(), 1);
    EXPECT_EQ(modelInfo.output_names.size(), 1);
    EXPECT_EQ(modelInfo.input_shapes.size(), 1);
}

TEST(TritonInterfaceTest, TritonModelInfoCanStoreMultipleInputs) {
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"input1", "input2"};
    modelInfo.output_names = {"output1", "output2"};

    EXPECT_EQ(modelInfo.input_names.size(), 2);
    EXPECT_EQ(modelInfo.output_names.size(), 2);
}

TEST(TritonInterfaceTest, TritonModelInfoHasExpectedFields) {
    TritonModelInfo modelInfo;

    // Verify all expected fields exist and can be set
    modelInfo.input_names.push_back("test");
    modelInfo.output_names.push_back("test");
    modelInfo.input_shapes.push_back({1, 3, 224, 224});
    modelInfo.input_formats.push_back("FORMAT_NCHW");
    modelInfo.input_datatypes.push_back("FP32");
    modelInfo.max_batch_size_ = 1;
    modelInfo.batch_size_ = 1;

    EXPECT_EQ(modelInfo.input_names[0], "test");
    EXPECT_EQ(modelInfo.max_batch_size_, 1);
}

TEST(TritonInterfaceTest, TritonModelInfoCanStoreBytesDatatype) {
    TritonModelInfo modelInfo;
    modelInfo.input_names.push_back("text_input");
    modelInfo.output_names.push_back("text_output");
    modelInfo.input_shapes.push_back({1});
    modelInfo.input_formats.push_back("FORMAT_NONE");
    modelInfo.input_datatypes.push_back("BYTES");
    modelInfo.input_types.push_back(-1);  // Sentinel for string type

    EXPECT_EQ(modelInfo.input_datatypes[0], "BYTES");
    EXPECT_EQ(modelInfo.input_types[0], -1);
}

TEST(TritonInterfaceTest, TritonModelInfoVllmLayout) {
    // Typical vLLM model layout on Triton
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"text_input"};
    modelInfo.output_names = {"text_output"};
    modelInfo.input_shapes = {{1}};
    modelInfo.input_formats = {"FORMAT_NONE"};
    modelInfo.input_datatypes = {"BYTES"};
    modelInfo.input_types = {-1};
    modelInfo.max_batch_size_ = 0;

    EXPECT_EQ(modelInfo.input_names[0], "text_input");
    EXPECT_EQ(modelInfo.output_names[0], "text_output");
    EXPECT_EQ(modelInfo.input_datatypes[0], "BYTES");
}

// TensorElement string variant tests

TEST(TensorElementTest, CanHoldString) {
    TensorElement element = std::string("hello world");
    EXPECT_TRUE(std::holds_alternative<std::string>(element));
    EXPECT_EQ(std::get<std::string>(element), "hello world");
}

TEST(TensorElementTest, CanHoldFloat) {
    TensorElement element = 3.14f;
    EXPECT_TRUE(std::holds_alternative<float>(element));
    EXPECT_FLOAT_EQ(std::get<float>(element), 3.14f);
}

TEST(TensorElementTest, CanHoldInt32) {
    TensorElement element = int32_t(42);
    EXPECT_TRUE(std::holds_alternative<int32_t>(element));
    EXPECT_EQ(std::get<int32_t>(element), 42);
}

TEST(TensorElementTest, CanHoldInt64) {
    TensorElement element = int64_t(100000);
    EXPECT_TRUE(std::holds_alternative<int64_t>(element));
    EXPECT_EQ(std::get<int64_t>(element), 100000);
}

TEST(TensorElementTest, CanHoldUint8) {
    TensorElement element = uint8_t(255);
    EXPECT_TRUE(std::holds_alternative<uint8_t>(element));
    EXPECT_EQ(std::get<uint8_t>(element), 255);
}

TEST(TensorTest, CanStoreStringTensor) {
    Tensor tensor;
    tensor.data = {std::string("generated text output")};
    tensor.shape = {1};

    ASSERT_EQ(tensor.data.size(), 1);
    EXPECT_TRUE(std::holds_alternative<std::string>(tensor.data[0]));
    EXPECT_EQ(std::get<std::string>(tensor.data[0]), "generated text output");
}

TEST(TensorTest, CanStoreMultipleStringElements) {
    Tensor tensor;
    tensor.data = {std::string("line 1"), std::string("line 2"), std::string("line 3")};
    tensor.shape = {3};

    ASSERT_EQ(tensor.data.size(), 3);
    EXPECT_EQ(std::get<std::string>(tensor.data[0]), "line 1");
    EXPECT_EQ(std::get<std::string>(tensor.data[2]), "line 3");
}

// Note: Concrete Triton class tests should be in integration tests
// where heavy dependencies are acceptable