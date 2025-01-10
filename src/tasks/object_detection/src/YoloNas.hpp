#include "TaskInterface.hpp"
class YoloNas : public TaskInterface
{
public:
    YoloNas(const TritonModelInfo& model_info)
        : TaskInterface(model_info) {
    }
    TaskType getTaskType() override { return TaskType::Detection; }    
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results, 
    const std::vector<std::vector<int64_t>>& infer_shapes) override;
    
private:
    std::vector<uint8_t> preprocess_image(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size);
};
