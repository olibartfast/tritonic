#pragma once
#include "common.hpp"
#include "TaskInterface.hpp"
#include <optional>
#include <stdexcept>

class DFine : public TaskInterface {
public:
    DFine(const TritonModelInfo& model_info);
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;                                
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;
    TaskType getTaskType() override { return TaskType::Detection; }                                  

private:
    std::vector<uint8_t> preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size);
    std::vector<uint8_t> preprocess_orig_size(const cv::Mat& img, const std::vector<int64_t>& shape);                                 

    std::optional<size_t> scores_idx_;
    std::optional<size_t> boxes_idx_;
    std::optional<size_t> labels_idx_;
};
