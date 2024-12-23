#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <variant>
#include "TritonModelInfo.hpp"

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


using Result = std::variant<Classification, Detection, InstanceSegmentation>;
using TensorElement = std::variant<float, int32_t, int64_t>;

class TaskInterface {
public:
    TaskInterface(const TritonModelInfo& modelInfo)
        : model_info_(modelInfo) {}

    std::vector<std::string> readLabelNames(const std::string& fileName) 
    {
        std::vector<std::string> classes;
        std::ifstream ifs(fileName.c_str());
        std::string line;
        while (getline(ifs, line))
            classes.push_back(line);
        return classes;
    }    

    virtual std::vector<Result> postprocess(
        const cv::Size& frame_size, 
        const std::vector<std::vector<TensorElement>>& infer_results, 
        const std::vector<std::vector<int64_t>>& infer_shapes) = 0;
    
    virtual std::vector<std::vector<uint8_t>> preprocess(
        const std::vector<cv::Mat>& imgs) = 0;

    virtual ~TaskInterface() {}

protected:
    TritonModelInfo model_info_;
};