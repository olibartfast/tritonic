#include <gtest/gtest.h>

#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

namespace {

using neuriplo_tasks::Detection;
using neuriplo_tasks::ModelInfo;
using neuriplo_tasks::Size;
using neuriplo_tasks::Tensor;
using neuriplo_tasks::TensorElement;

ModelInfo Yolo26DetectionModel() {
    ModelInfo info;
    info.input_names = {"images"};
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.max_batch_size_ = 1;
    return info;
}

}  // namespace

TEST(Yolo26PostprocessTest, PixelCoordinatesMapBackToOriginalFrame) {
    // Ultralytics end2end exports emit input-pixel xyxy; neuriplo-tasks v0.8.0 multiplied them by
    // 640 again.
    auto task = neuriplo_tasks::TaskFactory::createTaskInstance("yolo26", Yolo26DetectionModel());
    ASSERT_TRUE(task);

    // Create synthetic output tensor: 1 batch, 300 detections, 6 values per detection
    std::vector<TensorElement> output_data(1 * 300 * 6, TensorElement(0.0f));

    // Set row 0: {109.63f, 237.04f, 228.15f, 533.33f, 0.9f, 0.0f}
    // xyxy in 640-letterbox input pixels, score, class 0
    output_data[0] = TensorElement(109.63f);
    output_data[1] = TensorElement(237.04f);
    output_data[2] = TensorElement(228.15f);
    output_data[3] = TensorElement(533.33f);
    output_data[4] = TensorElement(0.9f);
    output_data[5] = TensorElement(0.0f);

    Tensor output_tensor(output_data, {1, 300, 6});

    // Original frame size: 810x1080
    Size original_size(810, 1080);

    auto results = task->postprocess(original_size, {output_tensor});

    // Expect exactly one detection
    ASSERT_EQ(results.size(), 1);

    // Extract detection from result variant
    auto detection = std::get<Detection>(results[0]);

    // Assert class 0
    EXPECT_EQ(static_cast<int>(detection.class_id), 0);

    // Assert confidence ~0.9
    EXPECT_FLOAT_EQ(detection.class_confidence, 0.9f);

    // Assert bbox within ±2 px of (50, 400, 200, 500)
    // Expected: x,y,w,h = (50, 400, 200, 500)
    EXPECT_NEAR(detection.bbox.x, 50, 2);
    EXPECT_NEAR(detection.bbox.y, 400, 2);
    EXPECT_NEAR(detection.bbox.width, 200, 2);
    EXPECT_NEAR(detection.bbox.height, 500, 2);
}
