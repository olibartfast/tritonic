#pragma once
#include "common.hpp"
#include "YOLO.hpp"





class YOLOSeg : public YOLO{
public:
    YOLOSeg(int input_width, int input_height)
        : YOLO(input_width, input_height) {
    }

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<float>>& infer_results, 
        const std::vector<std::vector<int64_t>>& infer_shapes) override;

    cv::Rect getSegPadSize(const size_t inputW,
    const size_t inputH,
    const cv::Size& inputSize);
            
};

