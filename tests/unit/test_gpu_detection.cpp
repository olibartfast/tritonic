#include <gtest/gtest.h>

#include "tritonic/core/gpu_segmentation.hpp"

namespace {
tritonic::triton::ModelInfo MakeGpuDetectionModelInfo() {
    tritonic::triton::ModelInfo info;
    info.input_names = {"IMAGE"};
    info.input_datatypes = {"UINT8"};
    info.input_shapes = {{1, -1}};
    info.output_names = {"NUM_DETECTIONS", "BOXES", "SCORES", "CLASSES"};
    info.output_datatypes = {"INT32", "INT32", "FP32", "INT32"};
    info.output_shapes = {{1, 1}, {1, 100, 4}, {1, 100}, {1, 100}};
    info.max_batch_size_ = 1;
    return info;
}

// One valid detection: class 5, score 0.9, box (2,3,4,5).
std::vector<tritonic::core::Tensor> MakeGpuDetectionOutputs(int32_t count = 1) {
    using tritonic::core::Tensor;
    using tritonic::core::TensorElement;
    std::vector<TensorElement> boxes(400, int32_t{0});
    boxes[0] = int32_t{2};
    boxes[1] = int32_t{3};
    boxes[2] = int32_t{4};
    boxes[3] = int32_t{5};
    std::vector<TensorElement> scores(100, 0.0F);
    scores[0] = 0.9F;
    std::vector<TensorElement> classes(100, int32_t{0});
    classes[0] = int32_t{5};
    return {
        Tensor({count}, {1, 1}),
        Tensor(std::move(boxes), {1, 100, 4}),
        Tensor(std::move(scores), {1, 100}),
        Tensor(std::move(classes), {1, 100}),
    };
}
}  // namespace

TEST(GpuDetectionTest, AcceptsExpectedDetectionModelContract) {
    EXPECT_NO_THROW(tritonic::core::ValidateGpuDetectionModel(MakeGpuDetectionModelInfo()));
}

TEST(GpuDetectionTest, RejectsWrongScoreDatatype) {
    auto info = MakeGpuDetectionModelInfo();
    info.output_datatypes[2] = "INT32";
    EXPECT_THROW(tritonic::core::ValidateGpuDetectionModel(info), std::runtime_error);
}

TEST(GpuDetectionTest, RejectsIncompleteOutputContract) {
    auto info = MakeGpuDetectionModelInfo();
    info.output_names.pop_back();
    info.output_datatypes.pop_back();
    info.output_shapes.pop_back();
    EXPECT_THROW(tritonic::core::ValidateGpuDetectionModel(info), std::runtime_error);
}

TEST(GpuDetectionTest, RejectsPreprocessedInputContract) {
    auto info = MakeGpuDetectionModelInfo();
    info.input_datatypes = {"FP32"};
    EXPECT_THROW(tritonic::core::ValidateGpuDetectionModel(info), std::runtime_error);
}

TEST(GpuDetectionTest, DecodesStrictDetectionResult) {
    const auto info = MakeGpuDetectionModelInfo();
    const auto results =
        tritonic::core::DecodeGpuDetectionResults(MakeGpuDetectionOutputs(), info.output_names);
    ASSERT_EQ(results.size(), 1U);
    const auto& detection = std::get<neuriplo_tasks::Detection>(results[0]);
    EXPECT_EQ(detection.class_id, 5.0F);
    EXPECT_FLOAT_EQ(detection.class_confidence, 0.9F);
    EXPECT_EQ(detection.bbox.x, 2);
    EXPECT_EQ(detection.bbox.y, 3);
    EXPECT_EQ(detection.bbox.width, 4);
    EXPECT_EQ(detection.bbox.height, 5);
}

TEST(GpuDetectionTest, DecodesEmptyResult) {
    const auto info = MakeGpuDetectionModelInfo();
    const auto results =
        tritonic::core::DecodeGpuDetectionResults(MakeGpuDetectionOutputs(0), info.output_names);
    EXPECT_TRUE(results.empty());
}

// A count larger than the fixed 100-detection envelope must not be trusted to index
// into the buffers.
TEST(GpuDetectionTest, RejectsCountBeyondEnvelope) {
    const auto info = MakeGpuDetectionModelInfo();
    EXPECT_THROW(
        tritonic::core::DecodeGpuDetectionResults(MakeGpuDetectionOutputs(101), info.output_names),
        std::runtime_error);
}

TEST(GpuDetectionTest, RejectsNegativeCount) {
    const auto info = MakeGpuDetectionModelInfo();
    EXPECT_THROW(
        tritonic::core::DecodeGpuDetectionResults(MakeGpuDetectionOutputs(-1), info.output_names),
        std::runtime_error);
}

TEST(GpuDetectionTest, RejectsTruncatedBoxes) {
    const auto info = MakeGpuDetectionModelInfo();
    auto tensors = MakeGpuDetectionOutputs();
    tensors[1] = tritonic::core::Tensor(std::vector<tritonic::core::TensorElement>(8, int32_t{0}),
                                        {1, 2, 4});
    EXPECT_THROW(tritonic::core::DecodeGpuDetectionResults(tensors, info.output_names),
                 std::runtime_error);
}

TEST(GpuDetectionTest, RejectsNonFiniteScore) {
    const auto info = MakeGpuDetectionModelInfo();
    auto tensors = MakeGpuDetectionOutputs();
    tensors[2].data[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(tritonic::core::DecodeGpuDetectionResults(tensors, info.output_names),
                 std::runtime_error);
}

TEST(GpuDetectionTest, RejectsNegativeClassId) {
    const auto info = MakeGpuDetectionModelInfo();
    auto tensors = MakeGpuDetectionOutputs();
    tensors[3].data[0] = int32_t{-1};
    EXPECT_THROW(tritonic::core::DecodeGpuDetectionResults(tensors, info.output_names),
                 std::runtime_error);
}

TEST(GpuDetectionTest, RejectsDegenerateBox) {
    const auto info = MakeGpuDetectionModelInfo();
    auto tensors = MakeGpuDetectionOutputs();
    tensors[1].data[2] = int32_t{0};  // zero width
    EXPECT_THROW(tritonic::core::DecodeGpuDetectionResults(tensors, info.output_names),
                 std::runtime_error);
}

TEST(GpuDetectionTest, RejectsMissingOutputName) {
    auto info = MakeGpuDetectionModelInfo();
    info.output_names[1] = "BOXES_RENAMED";
    EXPECT_THROW(
        tritonic::core::DecodeGpuDetectionResults(MakeGpuDetectionOutputs(), info.output_names),
        std::runtime_error);
}
