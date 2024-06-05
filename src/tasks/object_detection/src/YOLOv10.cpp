#include "YOLOv10.hpp"

YOLOv10::YOLOv10(int input_width, int input_height) : TaskInterface(input_width, input_height) {
    input_width_ = input_width;
    input_height_ = input_height;
}

std::vector<Result> YOLOv10::postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results, 
const std::vector<std::vector<int64_t>>& infer_shapes) 
{
    
    std::vector<Result> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    const auto confThreshold = 0.5f;
    const auto infer_shape = infer_shapes.front(); 
    auto infer_result = infer_results.front(); 

    std::vector<float> confidences;


    int rows = infer_shape[1]; // 300

    for (int i = 0; i < rows; ++i) 
    {
        if (i*infer_shape[2] + 4 >= infer_result.size()) {
            break;
        }

        float score = infer_result[i*infer_shape[2] + 4];
        if (score >= confThreshold) 
        {
            Detection d;
            float label = infer_result[i*infer_shape[2] + 5];
            d.class_id = static_cast<int>(label);
            d.class_confidence = score;
            float r_w = (frame_size.width * 1.0) / input_width_;
            float r_h = (frame_size.height * 1.0) / input_height_ ;

            float x1 = infer_result[i*infer_shape[2] + 0] * r_w;
            float y1 = infer_result[i*infer_shape[2] + 1] * r_h;
            float x2 = infer_result[i*infer_shape[2] + 2] * r_w;
            float y2 = infer_result[i*infer_shape[2] + 3] * r_h;

            d.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            detections.emplace_back(d);
        }
    }
    return detections; 
}


std::vector<uint8_t> YOLOv10::preprocess(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size) 
{
    // Image channels are in BGR order. Currently model configuration
    // data doesn't provide any information as to the expected channel
    // orderings (like RGB, BGR). We are going to assume that RGB is the
    // most likely ordering and so change the channels to that ordering.
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    sample.convertTo(
        sample, (img_channels == 3) ? img_type3 : img_type1);
    cv::resize(sample, sample, cv::Size(input_width_, input_height_));
    sample.convertTo(sample, CV_32FC3, 1.f / 255.f);


    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample.total() * sample.elemSize();
    size_t pos = 0;
    input_data.resize(img_byte_size);

    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i)
    {
        input_bgr_channels.emplace_back(
            img_size.height, img_size.width, img_type1, &(input_data[pos]));
        pos += input_bgr_channels.back().total() *
            input_bgr_channels.back().elemSize();
    }

    cv::split(sample, input_bgr_channels);

    if (pos != img_byte_size)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
            << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}
