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

TEST(TritonInterfaceTest, TritonModelInfoOutputMetadata) {
    TritonModelInfo modelInfo;
    modelInfo.output_names = {"text_output"};
    modelInfo.output_datatypes = {"BYTES"};
    modelInfo.output_shapes = {{-1}};

    EXPECT_EQ(modelInfo.output_names.size(), 1);
    EXPECT_EQ(modelInfo.output_datatypes.size(), 1);
    EXPECT_EQ(modelInfo.output_shapes.size(), 1);
    EXPECT_EQ(modelInfo.output_datatypes[0], "BYTES");
    EXPECT_EQ(modelInfo.output_shapes[0][0], -1);
}

TEST(TritonInterfaceTest, TritonModelInfoStringInputType) {
    TritonModelInfo modelInfo;
    modelInfo.input_names = {"text_input", "sampling_parameters"};
    modelInfo.input_datatypes = {"BYTES", "BYTES"};
    modelInfo.input_types = {-1, -1};  // No OpenCV type for strings
    modelInfo.input_shapes = {{1}, {1}};
    modelInfo.input_formats = {"FORMAT_NONE", "FORMAT_NONE"};

    EXPECT_EQ(modelInfo.input_datatypes[0], "BYTES");
    EXPECT_EQ(modelInfo.input_types[0], -1);
}

TEST(TensorElementTest, StringVariant) {
    TensorElement elem = std::string("hello world");
    EXPECT_TRUE(std::holds_alternative<std::string>(elem));
    EXPECT_EQ(std::get<std::string>(elem), "hello world");
}

TEST(TensorElementTest, AllVariantTypes) {
    TensorElement f = 3.14f;
    TensorElement i32 = int32_t(42);
    TensorElement i64 = int64_t(100);
    TensorElement u8 = uint8_t(255);
    TensorElement str = std::string("test");

    EXPECT_TRUE(std::holds_alternative<float>(f));
    EXPECT_TRUE(std::holds_alternative<int32_t>(i32));
    EXPECT_TRUE(std::holds_alternative<int64_t>(i64));
    EXPECT_TRUE(std::holds_alternative<uint8_t>(u8));
    EXPECT_TRUE(std::holds_alternative<std::string>(str));
}

TEST(TensorTest, StringTensorData) {
    std::vector<TensorElement> data;
    data.emplace_back(std::string("Generated text output"));
    Tensor tensor(std::move(data), {1});

    ASSERT_EQ(tensor.data.size(), 1);
    EXPECT_TRUE(std::holds_alternative<std::string>(tensor.data[0]));
    EXPECT_EQ(std::get<std::string>(tensor.data[0]), "Generated text output");
}

// Note: Concrete Triton class tests should be in integration tests
// where heavy dependencies are acceptable