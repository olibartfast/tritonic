#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include "vision-core/core/task_interface.hpp"
#include <variant>

namespace {

using vision_core::TensorElement;

TEST(RFDetrReproTest, VerifyBoxDecoding) {
    // Input size 640x640
    cv::Size input_size(640, 640);
    // Frame size 100x100 for easy calculation
    cv::Size frame_size(100, 100);
    
    // Thresholds
    float conf_thresh = 0.5f;
    float mask_thresh = 0.5f;
    
    vision_core::RfDetrSegmentationPostprocessor postprocessor(input_size, conf_thresh, mask_thresh);
    
    // Create mock output tensors
    // 0: boxes [1, 300, 4]
    // 1: labels [1, 300, num_classes]
    // 2: masks [1, 300, H, W]
    
    int num_dets = 1; // Just test 1 detection
    int num_classes = 80;
    int mask_h = 28;
    int mask_w = 28;
    
    std::vector<TensorElement> boxes_data;
    // Box at center with size 0.2x0.2 (normalized)
    // cx=0.5, cy=0.5, w=0.2, h=0.2
    // If interpreted as xyxy: x1=0.5, y1=0.5, x2=0.2, y2=0.2 -> Invalid box or tiny box at 0,0
    // If interpreted as cxcywh: x1=0.4, y1=0.4, x2=0.6, y2=0.6
    boxes_data.push_back(0.5f);
    boxes_data.push_back(0.5f);
    boxes_data.push_back(0.2f);
    boxes_data.push_back(0.2f);
    // Fill rest with 0
    for(int i=1; i<300; ++i) {
        boxes_data.push_back(0.0f); boxes_data.push_back(0.0f); boxes_data.push_back(0.0f); boxes_data.push_back(0.0f);
    }
    
    std::vector<TensorElement> labels_data;
    // Class 0 (person), high score
    // Logit for sigmoid(x) = 0.9 -> x approx 2.2
    float high_score_logit = 2.2f;
    // Class 1 in 1-based index is index 0 in 0-based? 
    // Reference says: max_class_idx -= 1. So if model outputs class 1, it is class 0.
    // Let's put high score at index 1 (which becomes class 0)
    for(int i=0; i<num_classes; ++i) {
        if (i == 1) labels_data.push_back(high_score_logit);
        else labels_data.push_back(-10.0f);
    }
    // Fill rest
    for(int i=1; i<300; ++i) {
        for(int c=0; c<num_classes; ++c) labels_data.push_back(-10.0f);
    }
    
    std::vector<TensorElement> masks_data;
    // Fill masks with 0
    for(int i=0; i<300 * mask_h * mask_w; ++i) masks_data.push_back(0.0f);
    
    // Swapped order: Boxes, Masks, Labels
    std::vector<std::vector<TensorElement>> infer_results = {boxes_data, masks_data, labels_data};
    
    std::vector<std::vector<int64_t>> infer_shapes = {
        {1, 300, 4},
        {1, 300, static_cast<int64_t>(mask_h), static_cast<int64_t>(mask_w)},
        {1, 300, static_cast<int64_t>(num_classes)}
    };
    
    auto results = postprocessor.postprocess(infer_results, infer_shapes, frame_size);
    
    // With dynamic identification, this should now work
    ASSERT_EQ(results.size(), 1);
    
    const auto& res = results[0];
    
    // Check Class ID
    EXPECT_EQ(res.class_id, 0); // 1 - 1 = 0
    
    // Check BBox
    // Expected if fixed (cxcywh):
    // x1 = 0.4 * 100 = 40
    // y1 = 0.4 * 100 = 40
    // w = 0.2 * 100 = 20
    // h = 0.2 * 100 = 20
    // Rect(40, 40, 20, 20)
    
    // Expected if bug (xyxy):
    // x1 = 0.5 * 100 = 50
    // y1 = 0.5 * 100 = 50
    // x2 = 0.2 * 100 = 20
    // y2 = 0.2 * 100 = 20
    // Rect(50, 50, -30, -30) -> OpenCV might handle this weirdly or make it empty
    
    std::cout << "Result BBox: " << res.bbox << std::endl;
    
    EXPECT_NEAR(res.bbox.x, 40, 2);
    EXPECT_NEAR(res.bbox.y, 40, 2);
    EXPECT_NEAR(res.bbox.width, 20, 2);
    EXPECT_NEAR(res.bbox.height, 20, 2);


}

} // namespace
