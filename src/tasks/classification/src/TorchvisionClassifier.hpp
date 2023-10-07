#pragma once
#include "TaskInterface.hpp"

class TorchvisionClassifier : public TaskInterface {
public:
    TorchvisionClassifier(int input_width, int input_height, int channels)
        : TaskInterface(input_width, input_height, channels) {
    }

    std::vector<Result> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results,
                                            const std::vector<std::vector<int64_t>>& infer_shapes) override {
        // Implement your postprocess logic here

        return std::vector<Result>{};
    }

    std::vector<uint8_t> preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size) override {
        // Implement your preprocess logic here
        return std::vector<uint8_t>{};
    }
};
