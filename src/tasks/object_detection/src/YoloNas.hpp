#include "TaskInterface.hpp"
class YoloNas : public TaskInterface
{
public:
    YoloNas(int input_width, int input_height)
        : TaskInterface(input_width, input_height) {
    }

    // Override the preprocess function
    std::vector<uint8_t> preprocess(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size) override;
   
    // Override the postprocess function
    std::vector<Result> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results, 
    const std::vector<std::vector<int64_t>>& infer_shapes) override;
    
private:
    // Add additional member variables specific to YoloNas
};
