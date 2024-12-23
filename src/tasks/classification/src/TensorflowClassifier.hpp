#pragma once
#include "TaskInterface.hpp"
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class TensorflowClassifier : public TaskInterface {
public:
    TensorflowClassifier(int input_width, int input_height, int channels);

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;

    std::vector<uint8_t> preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size) override;
};


