#include "RTDetr.hpp"

RTDetr::RTDetr(const TritonModelInfo& model_info) : TaskInterface(model_info) {
    for(size_t i = 0; i < model_info.input_shapes.size(); i++)
    {
        if(model_info.input_shapes[i].size() >= 3)
        { 
            input_channels_ =  model_info.input_formats[i] == "FORMAT_NHWC" ? model_info.input_shapes[i][3] : model_info.input_shapes[i][1];
            input_height_ = model_info.input_formats[i] == "FORMAT_NHWC" ? model_info.input_shapes[i][1] : model_info.input_shapes[i][2];
            input_width_ = model_info.input_shapes[i][2];
        }
    }    
    if (input_channels_ == 0 || input_height_ == 0 || input_width_ == 0)
        throw std::invalid_argument("Not initialized input width/height");

     for (size_t i = 0; i < model_info.output_names.size(); ++i) {
        if (model_info.output_names[i] == "scores") scores_idx_ = i;
        else if (model_info.output_names[i] == "boxes") boxes_idx_ = i;
        else if (model_info.output_names[i] == "labels") labels_idx_ = i;
    }

    // Check if all indices are set
    if (!scores_idx_.has_value() || !boxes_idx_.has_value() || !labels_idx_.has_value()) {
        throw std::runtime_error("Not all required output indices were set in the model info");
    }       
}   


std::vector<uint8_t> RTDetr::preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
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
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
            << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}


std::vector<std::vector<uint8_t>> RTDetr::preprocess(const std::vector<cv::Mat>& imgs)
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
        } else if (input_name == "orig_target_sizes" || input_name == "orig_size") {
            // Handle original image size input
            std::vector<int64_t> orig_sizes = {static_cast<int64_t>(input_width_), static_cast<int64_t>(input_height_)};
            input_data[i] = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(orig_sizes.data()),
                                                 reinterpret_cast<uint8_t*>(orig_sizes.data()) + orig_sizes.size() * sizeof(int64_t));
        } else {
            // For other types of inputs, you might need to add more cases
            // or use a default handling method
            std::cerr << "Warning: Unhandled input " << input_name << ". Sending empty data." << std::endl;
        }
    }
    return input_data;
}


std::vector<Result> RTDetr::postprocess(const cv::Size& frame_size, 
                                       const std::vector<std::vector<TensorElement>>& infer_results,
                                       const std::vector<std::vector<int64_t>>& infer_shapes) {
    


    const float confThreshold = 0.5f, iouThreshold = 0.4f;

    const auto& scores = infer_results[scores_idx_.value()];
    const auto& boxes = infer_results[boxes_idx_.value()];
    const auto& labels = infer_results[labels_idx_.value()];


    int rows = infer_shapes[scores_idx_.value()][1]; // Assuming this is 300

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes_rect;
    classIds.reserve(rows);
    confidences.reserve(rows);
    boxes_rect.reserve(rows);

    auto get_float = [](const TensorElement& elem) {
        return std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, elem);
    };

    float r_w = static_cast<float>(frame_size.width) / input_width_;
    float r_h = static_cast<float>(frame_size.height) / input_height_;

    for (int i = 0; i < rows; ++i) {
        float score = get_float(scores[i]);
        if (score >= confThreshold) {
            int class_id = std::visit([](auto&& label) -> int {
                using T = std::decay_t<decltype(label)>;
                if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
                    return static_cast<int>(label);
                }
                return -1; // Invalid class id
            }, labels[i]);

            if (class_id == -1) {
                std::cerr << "Warning: Invalid class id at index " << i << std::endl;
                continue;
            }

            classIds.push_back(class_id);
            confidences.push_back(score);
            float x1 = std::get<float>(boxes[i*4]);
            float y1 = std::get<float>(boxes[i*4 + 1]);
            float x2 = std::get<float>(boxes[i*4 + 2]);
            float y2 = std::get<float>(boxes[i*4 + 3]);
            
            x2 *= r_w;
            y2 *= r_h;
            x1 *= r_w;
            y1 *= r_h;
            boxes_rect.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_rect, confidences, confThreshold, iouThreshold, indices);
    
    std::vector<Result> detections;
    detections.reserve(indices.size());
   for (int idx : indices)
    {
        Detection d;
        d.bbox = cv::Rect(boxes_rect[idx]);
        d.class_confidence = confidences[idx];
        d.class_id = classIds[idx];
        detections.emplace_back(d);
    }                
    return detections; 
}

