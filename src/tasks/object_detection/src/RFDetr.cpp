#include "RFDetr.hpp"
#include "Logger.hpp"

RFDetr::RFDetr(const TritonModelInfo& model_info) : TaskInterface(model_info) {

     for (size_t i = 0; i < model_info.output_names.size(); ++i) {
         if (model_info.output_names[i] == "dets") dets_idx_ = i;
        else if (model_info.output_names[i] == "labels") labels_idx_ = i;
    }

    // Check if all indices are set
    if (!dets_idx_.has_value()  || !labels_idx_.has_value()) {
        throw std::runtime_error("Not all required output indices were set in the model info");
    }       
}   
std::vector<uint8_t> RFDetr::preprocess_image(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                              size_t img_channels, const cv::Size& img_size) {
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    cv::resize(sample, sample, img_size, 0, 0, cv::INTER_LINEAR);
    sample.convertTo(sample, CV_32FC3, 1.f / 255.f);
    cv::subtract(sample, cv::Scalar(0.485f, 0.456f, 0.406f), sample);
    cv::divide(sample, cv::Scalar(0.229f, 0.224f, 0.225f), sample);


    // Handle data layout (NHWC vs NCHW)
    size_t img_byte_size = sample.total() * sample.elemSize();
    input_data.resize(img_byte_size);

    // Check if data needs to be reordered for NCHW format
    // Assuming format string might be "FORMAT_NCHW" or similar based on Triton examples
    if (format.find("NCHW") != std::string::npos) {
        size_t pos = 0;
        std::vector<cv::Mat> input_channels;
        for (size_t i = 0; i < img_channels; ++i) {
             // Create a Mat view backed by the input_data vector
             input_channels.emplace_back(
                 img_size.height, img_size.width, CV_32F, input_data.data() + pos);
             pos += input_channels.back().total() * input_channels.back().elemSize();
        }
        cv::split(sample, input_channels); // Split channels directly into the buffer

        if (pos != img_byte_size) {
             logger.errorf("RFDetr Preprocess Error: unexpected total size of channels {}, expecting {}", pos, img_byte_size);
             throw std::runtime_error("RFDetr Preprocess Error: Mismatch in byte size during NCHW conversion.");
        }
    } else { // Assume NHWC or contiguous format
        std::memcpy(input_data.data(), sample.data, img_byte_size);
    }


    return input_data;
}



std::vector<std::vector<uint8_t>> RFDetr::preprocess(const std::vector<cv::Mat>& imgs)
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

std::vector<Result> RFDetr::postprocess(const cv::Size& frame_size,
                                        const std::vector<std::vector<TensorElement>>& infer_results,
                                        const std::vector<std::vector<int64_t>>& infer_shapes) {
    const float confThreshold = 0.5f;

    if (!dets_idx_.has_value() || !labels_idx_.has_value()) {
        throw std::runtime_error("Not all required output indices were set in the model info");
    }

    const auto& boxes = infer_results[dets_idx_.value()];
    const auto& labels = infer_results[labels_idx_.value()];
    const auto& shape_boxes = infer_shapes[dets_idx_.value()];
    const auto& shape_labels = infer_shapes[labels_idx_.value()];

    if (shape_boxes.size() < 3 || shape_labels.size() < 3) {
        throw std::runtime_error("Invalid output tensor shapes");
    }

    const size_t num_detections = shape_boxes[1];
    const size_t num_classes = shape_labels[2];

    const float scale_w = static_cast<float>(frame_size.width) / input_width_;
    const float scale_h = static_cast<float>(frame_size.height) / input_height_;

    std::vector<Result> final_detections;

    for (size_t i = 0; i < num_detections; ++i) {
        const size_t det_offset = i * shape_boxes[2];
        const size_t label_offset = i * num_classes;

        float max_score = -1.0f;
        int max_class_idx = -1;

        for (size_t j = 0; j < num_classes; ++j) {
            float logit;
            try {
                logit = std::get<float>(labels[label_offset + j]);
            } catch (const std::bad_variant_access&) {
                throw std::runtime_error("Invalid TensorElement type for labels at index " + std::to_string(label_offset + j));
            }

            const float score = sigmoid(logit);
            if (score > max_score) {
                max_score = score;
                max_class_idx = j;
            }
        }

        max_class_idx -= 1; // Adjust class index if necessary

        if (max_score > confThreshold && max_class_idx >= 0 && static_cast<size_t>(max_class_idx) < num_classes) {
            float x_center, y_center, width, height;
            try {
                x_center = std::get<float>(boxes[det_offset + 0]) * input_width_;
                y_center = std::get<float>(boxes[det_offset + 1]) * input_height_;
                width = std::get<float>(boxes[det_offset + 2]) * input_width_;
                height = std::get<float>(boxes[det_offset + 3]) * input_height_;
            } catch (const std::bad_variant_access&) {
                throw std::runtime_error("Invalid TensorElement type for boxes at index " + std::to_string(det_offset));
            }

            const float x_min = x_center - width / 2.0f;
            const float y_min = y_center - height / 2.0f;
            const float x_max = x_center + width / 2.0f;
            const float y_max = y_center + height / 2.0f;

            cv::Rect bbox(
                static_cast<int>(x_min * scale_w),
                static_cast<int>(y_min * scale_h),
                static_cast<int>((x_max - x_min) * scale_w),
                static_cast<int>((y_max - y_min) * scale_h)
            );

            Detection detection;
            detection.bbox = bbox;
            detection.class_confidence = max_score;
            detection.class_id = max_class_idx;
            final_detections.push_back(detection);
        }
    }

    return final_detections;
}