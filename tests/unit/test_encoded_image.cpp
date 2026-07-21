#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "tritonic/core/encoded_image.hpp"

namespace {
tritonic::triton::ModelInfo MakeRequestModel() {
    tritonic::triton::ModelInfo info;
    info.input_names = {"IMAGE"};
    info.input_datatypes = {"UINT8"};
    info.input_shapes = {{1, 3}};
    info.output_names = {"output0"};
    info.output_datatypes = {"FP32"};
    info.output_shapes = {{84, 8400}};
    info.max_batch_size_ = 1;
    return info;
}
}  // namespace

TEST(EncodedImageRequestTest, PreservesFileBytesAndSetsShape) {
    const auto path = std::filesystem::temp_directory_path() / "tritonic_encoded_image_test.jpg";
    const std::vector<uint8_t> expected = {0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00,
                                           0x02, 0x00, 0x03, 0x03, 0x01, 0x11, 0x00, 0x02,
                                           0x11, 0x00, 0x03, 0x11, 0x00, 0xff, 0xd9};
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()),
                     static_cast<std::streamsize>(expected.size()));
    }

    const auto request = tritonic::core::BuildEncodedImageRequest(path.string());
    std::filesystem::remove(path);

    ASSERT_EQ(request.inputs.size(), 1u);
    EXPECT_EQ(request.inputs[0], expected);
    EXPECT_EQ(request.shapes, (std::vector<std::vector<int64_t>>{{1, 23}}));
    EXPECT_EQ(request.width, 3);
    EXPECT_EQ(request.height, 2);
}

TEST(EncodedImageRequestTest, RejectsEmptyFile) {
    const auto path = std::filesystem::temp_directory_path() / "tritonic_empty_image_test.jpg";
    std::ofstream(path).close();
    EXPECT_THROW(tritonic::core::BuildEncodedImageRequest(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(EncodedImageRequestTest, AcceptsCompatibleModelMetadata) {
    const auto request = MakeRequestModel();
    auto task = request;
    task.input_names = {"images"};
    task.input_datatypes = {"FP32"};
    EXPECT_NO_THROW(tritonic::core::ValidateEncodedImageModels(request, task));
}

TEST(EncodedImageRequestTest, RejectsOutputMismatch) {
    const auto request = MakeRequestModel();
    auto task = request;
    task.output_shapes = {{85, 8400}};
    EXPECT_THROW(tritonic::core::ValidateEncodedImageModels(request, task), std::runtime_error);
}
