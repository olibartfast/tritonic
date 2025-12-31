#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <variant>
#include <stdexcept>
#include "TritonModelInfo.hpp"
#include "CommonTypes.hpp"

struct Classification {
    // fields specific to Classification
    float class_id;
    float class_confidence;
};

struct Detection : public Classification{
    // fields specific to Detection
    cv::Rect bbox;
};

struct InstanceSegmentation : public Detection {
    std::vector<uchar> mask_data;  // Store mask data as a vector
    int mask_height;  // Store mask height
    int mask_width;   // Store mask width
};

struct OpticalFlow {
    cv::Mat flow;           // Colored visualization
    cv::Mat raw_flow;       // Raw flow field (CV_32FC2)
    float max_displacement; // Maximum flow magnitude
};

struct VideoClassification : public Classification{
    std::string action_label;   // Human-readable action name
    std::vector<float> frame_scores;  // Confidence scores per frame if needed
};

using Result = std::variant<Classification, Detection, InstanceSegmentation, OpticalFlow, VideoClassification>;


enum class TaskType {
    OpticalFlow,
    Classification,
    Detection,
    InstanceSegmentation,
    VideoClassification
};

class InputDimensionError : public std::runtime_error {
public:
    InputDimensionError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class TaskInterface {
public:
    TaskInterface(const TritonModelInfo& modelInfo)
        : model_info_(modelInfo) {
        std::tie(input_width_, input_height_, input_channels_) = initializeInputDimensions(model_info_);
        
        if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ <= 0) {
            throw InputDimensionError("Invalid input dimensions");
        }
    }

    virtual ~TaskInterface() = default;

    virtual TaskType getTaskType() = 0;

    // Pure virtual functions
    virtual std::vector<Result> postprocess(
        const cv::Size& frame_size, 
        const std::vector<Tensor>& tensors) = 0;
    
    virtual std::vector<std::vector<uint8_t>> preprocess(
        const std::vector<cv::Mat>& imgs) = 0;

    // Utility functions
    std::vector<std::string> readLabelNames(const std::string& fileName) const {
        std::vector<std::string> classes;
        std::ifstream ifs(fileName.c_str());
        std::string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }
        return classes;
    }    

protected:
    TritonModelInfo model_info_;
    int input_width_ = 0;
    int input_height_ = 0;
    int input_channels_ = 0;    

private:
    std::tuple<int, int, int> initializeInputDimensions(const TritonModelInfo& model_info) const {
        for (size_t i = 0; i < model_info.input_shapes.size(); i++) {
            if (model_info.input_shapes[i].size() >= 3) {
                int channels = model_info.input_formats[i] == "FORMAT_NHWC" ? model_info.input_shapes[i][3] : model_info.input_shapes[i][1];
                int height = model_info.input_formats[i] == "FORMAT_NHWC" ? model_info.input_shapes[i][1] : model_info.input_shapes[i][2];
                int width = model_info.input_formats[i] == "FORMAT_NHWC" ? model_info.input_shapes[i][2] : model_info.input_shapes[i][3];
                return std::make_tuple(width, height, channels);
            }
        }
        throw InputDimensionError("No valid input shape found");
    }
};
