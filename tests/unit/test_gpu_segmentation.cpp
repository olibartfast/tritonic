#include <gtest/gtest.h>

#include "tritonic/core/gpu_segmentation.hpp"

namespace {
tritonic::triton::ModelInfo MakeGpuModelInfo() {
    tritonic::triton::ModelInfo info;
    info.input_names = {"IMAGE"};
    info.input_datatypes = {"UINT8"};
    info.input_shapes = {{1, -1}};
    info.output_names = {"NUM_DETECTIONS", "BOXES",       "SCORES",   "CLASSES",
                         "MASK_OFFSETS",   "MASK_SHAPES", "MASK_DATA"};
    info.output_datatypes = {"INT32", "INT32", "FP32", "INT32", "INT64", "INT32", "UINT8"};
    info.output_shapes = {{1, 1}, {1, 100, 4}, {1, 100}, {1, 100}, {1, 101}, {1, 100, 2}, {1, -1}};
    info.max_batch_size_ = 1;
    return info;
}

std::vector<tritonic::core::Tensor> MakeGpuOutputs(bool nonempty_mask = true) {
    using tritonic::core::Tensor;
    using tritonic::core::TensorElement;
    std::vector<TensorElement> boxes(400, int32_t{0});
    boxes[0] = int32_t{2};
    boxes[1] = int32_t{3};
    boxes[2] = int32_t{2};
    boxes[3] = int32_t{2};
    std::vector<TensorElement> scores(100, 0.0F);
    scores[0] = 0.9F;
    std::vector<TensorElement> classes(100, int32_t{0});
    classes[0] = int32_t{5};
    std::vector<TensorElement> offsets(101, int64_t{4});
    offsets[0] = int64_t{0};
    std::vector<TensorElement> shapes(200, int32_t{0});
    shapes[0] = int32_t{2};
    shapes[1] = int32_t{2};
    std::vector<TensorElement> mask(4, uint8_t{0});
    if (nonempty_mask)
        mask[1] = uint8_t{255};
    return {
        Tensor({int32_t{1}}, {1, 1}),         Tensor(std::move(boxes), {1, 100, 4}),
        Tensor(std::move(scores), {1, 100}),  Tensor(std::move(classes), {1, 100}),
        Tensor(std::move(offsets), {1, 101}), Tensor(std::move(shapes), {1, 100, 2}),
        Tensor(std::move(mask), {1, 4}),
    };
}
}  // namespace

TEST(GpuSegmentationTest, AcceptsExpectedModelContract) {
    EXPECT_NO_THROW(tritonic::core::ValidateGpuSegmentationModel(MakeGpuModelInfo()));
}

TEST(GpuSegmentationTest, RejectsWrongMaskDatatype) {
    auto info = MakeGpuModelInfo();
    info.output_datatypes.back() = "INT32";
    EXPECT_THROW(tritonic::core::ValidateGpuSegmentationModel(info), std::runtime_error);
}

TEST(GpuSegmentationTest, DecodesStrictPackedMaskResult) {
    const auto info = MakeGpuModelInfo();
    const auto results =
        tritonic::core::DecodeGpuSegmentationResults(MakeGpuOutputs(), info.output_names, 10, 10);
    ASSERT_EQ(results.size(), 1U);
    const auto& result = std::get<neuriplo_tasks::InstanceSegmentation>(results[0]);
    EXPECT_EQ(result.bbox.x, 2);
    EXPECT_EQ(result.bbox.y, 3);
    EXPECT_EQ(result.bbox.width, 2);
    EXPECT_EQ(result.bbox.height, 2);
    EXPECT_FLOAT_EQ(result.class_confidence, 0.9F);
    EXPECT_FLOAT_EQ(result.class_id, 5.0F);
    EXPECT_EQ(result.mask_data, (std::vector<uint8_t>{0, 255, 0, 0}));
}

TEST(GpuSegmentationTest, RejectsEmptyGarbageMask) {
    const auto info = MakeGpuModelInfo();
    EXPECT_THROW(tritonic::core::DecodeGpuSegmentationResults(MakeGpuOutputs(false),
                                                              info.output_names, 10, 10),
                 std::runtime_error);
}
