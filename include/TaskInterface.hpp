#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <variant>

struct Classification {
    // fields specific to Classification
    float class_id;
    float class_confidence;
};

struct Detection : public Classification{
    // fields specific to Detection
    cv::Rect bbox;
};

using Result = std::variant<Classification, Detection>;

class TaskInterface {
public:

    TaskInterface(int input_width = 0, int input_height = 0, int channels = 3)
        : input_width_(input_width), input_height_(input_height), channels_(channels) {}

    std::vector<std::string> readLabelNames(const std::string& fileName) 
    {
        std::vector<std::string> classes;
        std::ifstream ifs(fileName.c_str());
        std::string line;
        while (getline(ifs, line))
            classes.push_back(line);
        return classes;
    }    

    virtual std::vector<Result> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results, 
        const std::vector<std::vector<int64_t>>& infer_shapes) = 0;
    
    virtual std::vector<uint8_t> preprocess(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size) = 0;

    virtual ~TaskInterface() {}

protected:
    int input_width_;
    int input_height_;
    int channels_;    
};