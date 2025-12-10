#pragma once
#include "common.hpp"
#include "TaskInterface.hpp"
#include <optional>
#include <stdexcept>

class RFDetr : public TaskInterface {
public:
    RFDetr(const TritonModelInfo& model_info);
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;    
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;
    TaskType getTaskType() override { return TaskType::Detection; }
    
protected:
    std::vector<uint8_t> preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size);
    std::vector<uint8_t> preprocess_orig_size(const cv::Mat& img, const std::vector<int64_t>& shape);                                 

    inline float sigmoid(float x) const noexcept {
        return 1.0f / (1.0f + std::exp(-x));
    }    
    
    std::optional<size_t> dets_idx_;
    std::optional<size_t> labels_idx_;
};
