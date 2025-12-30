#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "ITriton.hpp"
#include <vision-infra/vision-infra.hpp>
#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/result_types.hpp"

class App {
public:
    App(std::shared_ptr<ITriton> triton, 
        std::shared_ptr<vision_infra::config::InferenceConfig> config,
        std::shared_ptr<vision_infra::core::Logger> logger);
    
    int run();

private:
    std::shared_ptr<ITriton> tritonClient_;
    std::shared_ptr<vision_infra::config::InferenceConfig> config_;
    std::shared_ptr<vision_infra::core::Logger> logger_;
    std::unique_ptr<vision_core::TaskInterface> task_;
    std::vector<std::string> class_names_;

    vision_core::ModelInfo convertToVisionCoreModelInfo(const TritonModelInfo& triton_info);
    
    std::vector<vision_core::Result> processSource(const std::vector<cv::Mat>& source);
    
    void processImages(const std::vector<std::string>& sourceNames);
    
    void processVideo(const std::string& sourceName);
    
    void drawLabel(cv::Mat& image, const std::string& label, float confidence, int x, int y);
    
    std::vector<cv::Scalar> generateRandomColors(int numColors);
    
    bool isImageFile(const std::string& filename);
    bool isVideoFile(const std::string& filename);
    std::vector<std::string> split(const std::string& s, char delimiter);
};
