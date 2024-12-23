#include "RTDetr.hpp"

RTDetr::RTDetr(int input_width, int input_height) : input_width_(input_width), input_height_(input_height) {}

std::vector<uint8_t> RTDetr::preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size) {
    std::vector<uint8_t> result;
}

std::vector<Result> RTDetr::postprocess(const cv::Size& frame_size, const std::vector<std::vector<float>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) {
    std::vector<Result> results;
    return results;
}