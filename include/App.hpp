#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Config.hpp"
#include "ConfigManager.hpp"
#include "ITriton.hpp"
#include "Logger.hpp"
#include "neuriplo/tasks/core/batch_types.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

class App {
public:
    App(std::shared_ptr<ITriton> triton, std::shared_ptr<InferenceConfig> config,
        std::shared_ptr<Logger> logger);

    int run();

private:
    std::shared_ptr<ITriton> tritonClient_;
    std::shared_ptr<InferenceConfig> config_;
    std::shared_ptr<Logger> logger_;
    std::unique_ptr<neuriplo_tasks::TaskInterface> task_;
    std::vector<std::string> class_names_;
    std::vector<cv::Scalar> colors_;
    int num_frames_{16};     // Frame buffer size for video classification tasks
    int max_batch_size_{1};  // Engine batch cap from model config
    std::vector<std::vector<int64_t>> base_input_shapes_;  // Per-input shapes at N=1 ([1,C,H,W])

    neuriplo_tasks::ModelInfo convertToNeuriploTasksModelInfo(const TritonModelInfo& triton_info);

    std::vector<neuriplo_tasks::Result> processSource(const std::vector<cv::Mat>& source);

    void processImages(const std::vector<std::string>& sourceNames);

    // Batched image pipeline (neuriplo-tasks v0.5.0 batch utilities).
    bool isBatchableImageTask() const;
    void processImagesBatched(const std::vector<std::string>& sourceNames);
    std::vector<std::vector<uint8_t>> stackBatchBuffers(
        const neuriplo_tasks::BatchPreprocessOutput& pre, int batch_size) const;
    void applyBatchedInputShapes(int batch_size);
    std::vector<neuriplo_tasks::Tensor> toNeuriploTensors(const std::vector<Tensor>& tensors) const;
    std::vector<neuriplo_tasks::Tensor> sliceTensorsAxis0(
        const std::vector<neuriplo_tasks::Tensor>& tensors, int index, int batch_size) const;
    std::vector<std::vector<neuriplo_tasks::Result>> postprocessBatched(
        const std::vector<Tensor>& tensors, const std::vector<cv::Mat>& images, int batch_size);

    void processVideo(const std::string& sourceName);

    void processVideoClassification(const std::string& sourceName);

    void processTextGeneration(const TritonModelInfo& modelInfo);

    void renderPrediction(cv::Mat& frame, const neuriplo_tasks::Result& prediction);

    void drawLabel(cv::Mat& image, const std::string& label, float confidence, int x, int y);

    void drawPose(cv::Mat& image, const neuriplo_tasks::PoseEstimation& pose,
                  float confidence_threshold = 0.5f);

    std::vector<cv::Scalar> generateRandomColors(int numColors);

    bool isImageFile(const std::string& filename);
    bool isVideoFile(const std::string& filename);
    bool isTextGenerationModelType(const std::string& modelType);
    std::vector<std::string> split(const std::string& s, char delimiter);
};
