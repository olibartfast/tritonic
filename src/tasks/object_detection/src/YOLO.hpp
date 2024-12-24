#pragma once
#include "common.hpp"
#include "TaskInterface.hpp"

class YOLO : public TaskInterface {
public:
    YOLO(const TritonModelInfo& model_info);
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override; 
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;

protected:
    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);
    std::tuple<float, int> getBestClassInfo(const std::vector<TensorElement>& data, size_t startIdx, const size_t& numClasses);
    std::vector<uint8_t> preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size);

};
