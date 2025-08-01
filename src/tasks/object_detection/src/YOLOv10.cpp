#include "YOLOv10.hpp"
#include "Logger.hpp"

YOLOv10::YOLOv10(const TritonModelInfo& model_info) : TaskInterface(model_info) {

}

std::vector<std::vector<uint8_t>> YOLOv10::preprocess(const std::vector<cv::Mat>& imgs)
{
    if (imgs.empty()) {
        throw std::runtime_error("Input image vector is empty");
    }

    cv::Mat img = imgs.front();
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    std::vector<std::vector<uint8_t>> input_data(model_info_.input_shapes.size());

    for (size_t i = 0; i < model_info_.input_shapes.size(); ++i) {
        const auto& input_name = model_info_.input_names[i];
        const auto& input_shape = model_info_.input_shapes[i];
        const auto& input_format = model_info_.input_formats[i];
        const auto& input_type = model_info_.input_types[i];

        if (input_shape.size() >= 3) {
            // This is likely an image input
            const auto input_size = cv::Size(input_width_, input_height_);
            input_data[i] = preprocess_image(img, input_format, model_info_.type1_, model_info_.type3_, img.channels(), input_size);
        } else {
            // For other types of inputs, you might need to add more cases
            // or use a default handling method
            throw std::runtime_error("Unhandled input");
        }
    }
    return input_data;
}

std::vector<uint8_t> YOLOv10::preprocess_image(
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
        logger.errorf("unexpected total size of channels {}, expecting {}", pos, img_byte_size);
        exit(1);
    }

    return input_data;
}


std::vector<Result> YOLOv10::postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results, 
const std::vector<std::vector<int64_t>>& infer_shapes) 
{
    std::vector<Result> detections;
    const auto confThreshold = 0.5f;
    const auto& infer_shape = infer_shapes.front(); 
    const auto& infer_result = infer_results.front(); 

    auto get_float = [](const TensorElement& elem) {
        return std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, elem);
    };

    int rows = infer_shape[1]; // 300

    for (int i = 0; i < rows; ++i) 
    {
        if (i*infer_shape[2] + 5 >= infer_result.size()) {
            break;
        }

        float score = get_float(infer_result[i*infer_shape[2] + 4]);
        if (score >= confThreshold) 
        {
            Detection d;
            float label = get_float(infer_result[i*infer_shape[2] + 5]);
            d.class_id = static_cast<int>(label);
            d.class_confidence = score;
            float r_w = (frame_size.width * 1.0) / input_width_;
            float r_h = (frame_size.height * 1.0) / input_height_;

            float x1 = get_float(infer_result[i*infer_shape[2] + 0]) * r_w;
            float y1 = get_float(infer_result[i*infer_shape[2] + 1]) * r_h;
            float x2 = get_float(infer_result[i*infer_shape[2] + 2]) * r_w;
            float y2 = get_float(infer_result[i*infer_shape[2] + 3]) * r_h;

            d.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            detections.emplace_back(d);
        }
    }
    return detections; 
}
