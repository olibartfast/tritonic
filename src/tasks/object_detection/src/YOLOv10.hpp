#pragma once
#include "common.hpp"
#include "TaskInterface.hpp"

class YOLOv10 : public TaskInterface {
public:
    YOLOv10(const std::vector<std::vector<int64_t>>& input_sizes);

    
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;

    std::vector<uint8_t> preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size) override;

private:
    int input_width_;
    int input_height_;
};
