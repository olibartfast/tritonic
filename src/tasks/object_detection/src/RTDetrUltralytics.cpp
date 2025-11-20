#include "RTDetrUltralytics.hpp"
#include "Logger.hpp"

std::vector<uint8_t> RTDetrUltralytics::preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size) {
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    sample.convertTo(
        sample, (img_channels == 3) ? img_type3 : img_type1);
    sample.convertTo(sample, CV_32FC3, 1.f / 255.f);
    cv::resize(sample, sample, cv::Size(img_size.width, img_size.height));

    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample.total() * sample.elemSize();
    size_t pos = 0;
    input_data.resize(img_byte_size);

    // (format.compare("FORMAT_NCHW") == 0)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
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


std::vector<std::vector<uint8_t>> RTDetrUltralytics::preprocess(const std::vector<cv::Mat>& imgs)
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
            logger.warnf("Warning: Unhandled input {}. Sending empty data.", input_name);
        }
    }
    return input_data;
}

std::vector<Result> RTDetrUltralytics::postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results, 
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
    int dimensions_scores = infer_shape[2] - 4; // num classes (80)

    // Iterate through detections.
    for (int i = 0; i < rows; ++i) 
    {
        auto start = infer_result.begin() + i * infer_shape[2];
        auto maxSPtr = std::max_element(start + 4, start + 4 + dimensions_scores, 
            [&get_float](const TensorElement& a, const TensorElement& b) {
                return get_float(a) < get_float(b);
            });

        float score = get_float(*maxSPtr);
        if (score >= confThreshold) 
        {
            int label = std::distance(start + 4, maxSPtr);

            float b0 = get_float(*(start));
            float b1 = get_float(*(start + 1));
            float b2 = get_float(*(start + 2));
            float b3 = get_float(*(start + 3));

            float x1 = (b0 - b2 / 2.0f) * frame_size.width;
            float y1 = (b1 - b3 / 2.0f) * frame_size.height;
            float x2 = (b0 + b2 / 2.0f) * frame_size.width;
            float y2 = (b1 + b3 / 2.0f) * frame_size.height;

            Detection d;
            d.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            d.class_confidence = score;
            d.class_id = label;
            detections.emplace_back(d);
        }
    }

    return detections; 
}