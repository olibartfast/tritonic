#include <gtest/gtest.h>

#include "tritonic/core/gpu_segmentation.hpp"

namespace {
tritonic::triton::ModelInfo MakeGpuModelInfo() {
    tritonic::triton::ModelInfo info;
    info.input_names = {"IMAGE"};
    info.input_datatypes = {"UINT8"};
    info.input_shapes = {{1, -1}};
    info.output_names = {
        "NUM_DETECTIONS",     "BOXES",         "SCORES", "CLASSES", "INSTANCE_RING_OFFSETS",
        "RING_POINT_OFFSETS", "POLYGON_POINTS"};
    info.output_datatypes = {"INT32", "INT32", "FP32", "INT32", "INT64", "INT64", "INT32"};
    info.output_shapes = {{1, 1}, {1, 100, 4}, {1, 100}, {1, 100}, {1, 101}, {1, -1}, {1, -1, 2}};
    info.max_batch_size_ = 1;
    return info;
}

std::vector<tritonic::core::Tensor> MakeGpuOutputs(bool valid_polygon = true) {
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
    std::vector<TensorElement> instance_offsets(101, int64_t{1});
    instance_offsets[0] = int64_t{0};
    std::vector<TensorElement> ring_offsets = {int64_t{0}, int64_t{valid_polygon ? 4 : 2}};
    std::vector<TensorElement> points = {
        int32_t{2}, int32_t{3}, int32_t{4}, int32_t{3},
        int32_t{4}, int32_t{5}, int32_t{2}, int32_t{5},
    };
    return {
        Tensor({int32_t{1}}, {1, 1}),
        Tensor(std::move(boxes), {1, 100, 4}),
        Tensor(std::move(scores), {1, 100}),
        Tensor(std::move(classes), {1, 100}),
        Tensor(std::move(instance_offsets), {1, 101}),
        Tensor(std::move(ring_offsets), {1, 2}),
        Tensor(std::move(points), {1, 4, 2}),
    };
}
}  // namespace

TEST(GpuSegmentationTest, AcceptsExpectedPolygonModelContract) {
    EXPECT_NO_THROW(tritonic::core::ValidateGpuSegmentationModel(MakeGpuModelInfo()));
}

TEST(GpuSegmentationTest, RejectsWrongPolygonPointDatatype) {
    auto info = MakeGpuModelInfo();
    info.output_datatypes.back() = "FP32";
    EXPECT_THROW(tritonic::core::ValidateGpuSegmentationModel(info), std::runtime_error);
}

TEST(GpuSegmentationTest, DecodesStrictPackedPolygonResult) {
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
    EXPECT_TRUE(result.mask.empty());
    EXPECT_TRUE(result.mask_data.empty());
    ASSERT_EQ(result.polygons.size(), 1U);
    EXPECT_EQ(result.polygons[0].exterior.size(), 4U);
    EXPECT_TRUE(result.polygons[0].holes.empty());
}

TEST(GpuSegmentationTest, RejectsDegeneratePolygon) {
    const auto info = MakeGpuModelInfo();
    EXPECT_THROW(tritonic::core::DecodeGpuSegmentationResults(MakeGpuOutputs(false),
                                                              info.output_names, 10, 10),
                 std::runtime_error);
}
