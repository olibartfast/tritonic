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

TEST(TritonInterfaceTest, TritonModelInfoStoresOutputMetadata) {
    TritonModelInfo modelInfo;
    modelInfo.output_names = {"text_output"};
    modelInfo.output_datatypes = {"BYTES"};
    modelInfo.output_shapes = {{-1}};
    modelInfo.input_datatypes = {"BYTES", "BOOL"};
    modelInfo.input_types = {TritonModelInfo::kStringTypeSentinel, CV_8U};

    EXPECT_EQ(modelInfo.output_datatypes[0], "BYTES");
    EXPECT_EQ(modelInfo.output_shapes[0][0], -1);
    EXPECT_EQ(modelInfo.input_types[0], TritonModelInfo::kStringTypeSentinel);
    EXPECT_EQ(modelInfo.input_types[1], CV_8U);
}

TEST(TensorElementTest, SupportsStringVariant) {
    TensorElement element = std::string("hello");
    EXPECT_TRUE(std::holds_alternative<std::string>(element));
    EXPECT_EQ(std::get<std::string>(element), "hello");
}

TEST(TensorTest, StoresStringData) {
    Tensor tensor({std::string("generated text")}, {1});
    ASSERT_EQ(tensor.data.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<std::string>(tensor.data[0]));
    EXPECT_EQ(std::get<std::string>(tensor.data[0]), "generated text");
}

// Note: Concrete Triton class tests should be in integration tests
// where heavy dependencies are acceptable
