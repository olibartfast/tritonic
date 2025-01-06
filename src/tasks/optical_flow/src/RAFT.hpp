#pragma once
#include "common.hpp"
#include "TaskInterface.hpp"
#include <optional>
#include <stdexcept>

class RAFT : public TaskInterface {
public:
    RAFT(const TritonModelInfo& model_info) : TaskInterface(model_info)  {

        for (size_t i = 0; i < model_info.output_names.size(); ++i) {
            if (model_info.output_names[i] == "output") output_idx_ = i;
        }

        if (!output_idx_.has_value()) {
            throw std::runtime_error("Not all required output indices were set in the model info");
        }  
    }

    std::vector<std::vector<uint8_t>>  preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, 
                                    const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override; 

private:
    std::vector<uint8_t> preprocess_image(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size); 
        std::optional<size_t> output_idx_;

};