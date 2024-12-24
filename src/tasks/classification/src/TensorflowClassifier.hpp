#pragma once
#include "TaskInterface.hpp"
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class TensorflowClassifier : public TaskInterface {
public:
    TensorflowClassifier(const TritonModelInfo& model_info);
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;   
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;

private:
    std::vector<uint8_t> preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size);
};


