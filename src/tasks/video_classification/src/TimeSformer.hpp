#pragma once

#include "TaskInterface.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

class TimeSformer : public TaskInterface {
public:
    explicit TimeSformer(const TritonModelInfo& model_info);
    
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& frames) override;
    
    std::vector<Result> postprocess(const cv::Size& frame_size, 
                                   const std::vector<std::vector<TensorElement>>& infer_results,
                                   const std::vector<std::vector<int64_t>>& infer_shapes) override;
    
    TaskType getTaskType() override { return TaskType::VideoClassification; }

private:
    std::vector<uint8_t> preprocess_video_clip(const std::vector<cv::Mat>& frames, 
                                              const std::string& format, 
                                              int img_type1, 
                                              int img_type3,
                                              size_t img_channels, 
                                              const cv::Size& img_size);
    
    void apply_imagenet_normalization(cv::Mat& image);
    std::vector<float> apply_softmax(const std::vector<float>& logits);
    std::vector<cv::Mat> sample_frames(const std::vector<cv::Mat>& frames, int target_frames);
    
    // TimeSformer-specific constants
    static constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};
    static constexpr float CONFIDENCE_THRESHOLD = 0.01f;
    static constexpr size_t TOP_K_PREDICTIONS = 5;
    static constexpr int DEFAULT_NUM_FRAMES = 8; // Default number of frames for TimeSformer
    
    int num_frames_;
    std::vector<std::string> class_names_;
    
    void loadActionLabels();
};
