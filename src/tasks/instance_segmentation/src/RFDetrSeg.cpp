#include "RFDetrSeg.hpp"
#include "Logger.hpp"

RFDetrSeg::RFDetrSeg(const TritonModelInfo& model_info) : RFDetr(model_info) {
    // Find the masks output in addition to the base class outputs (dets, labels)
    // The masks output can be named "masks" or any other name (like a numeric ID)
    for (size_t i = 0; i < model_info.output_names.size(); ++i) {
        const auto& name = model_info.output_names[i];
        if (name == "masks") {
            masks_idx_ = i;
            break;
        } else if (name != "dets" && name != "labels") {
            // Assume any output that's not dets or labels is the masks output
            masks_idx_ = i;
        }
    }

    // Check if masks index is set
    if (!masks_idx_.has_value()) {
        throw std::runtime_error("Masks output not found in model info. Expected 3 outputs: dets, labels, and masks.");
    }
    
    logger.infof("RFDetrSeg initialized with masks output: {}", model_info.output_names[masks_idx_.value()]);
}

std::vector<Result> RFDetrSeg::postprocess(const cv::Size& frame_size,
                                           const std::vector<std::vector<TensorElement>>& infer_results,
                                           const std::vector<std::vector<int64_t>>& infer_shapes) {
    const float confThreshold = 0.5f;
    const float iouThreshold = 0.4f;
    const float mask_threshold = 0.5f;

    if (!dets_idx_.has_value() || !labels_idx_.has_value() || !masks_idx_.has_value()) {
        throw std::runtime_error("Not all required output indices were set in the model info");
    }

    const auto& boxes = infer_results[dets_idx_.value()];
    const auto& labels = infer_results[labels_idx_.value()];
    const auto& masks_raw = infer_results[masks_idx_.value()];
    const auto& shape_boxes = infer_shapes[dets_idx_.value()];
    const auto& shape_labels = infer_shapes[labels_idx_.value()];
    const auto& shape_masks = infer_shapes[masks_idx_.value()];

    if (shape_boxes.size() < 3 || shape_labels.size() < 3 || shape_masks.size() < 4) {
        throw std::runtime_error("Invalid output tensor shapes");
    }

    const size_t num_detections = shape_boxes[1];
    const size_t num_classes = shape_labels[2];
    const size_t mask_h = shape_masks[2];
    const size_t mask_w = shape_masks[3];

    const float scale_w = static_cast<float>(frame_size.width) / input_width_;
    const float scale_h = static_cast<float>(frame_size.height) / input_height_;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels_vec;
    std::vector<cv::Mat> mask_proposals; // Store mask logits for each detection

    // First pass: collect detections above threshold
    for (size_t i = 0; i < num_detections; ++i) {
        const size_t det_offset = i * shape_boxes[2];
        const size_t label_offset = i * num_classes;
        const size_t mask_offset = i * mask_h * mask_w;

        float max_score = -1.0f;
        int max_class_idx = -1;

        // Find best class
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

            bboxes.push_back(bbox);
            scores.push_back(max_score);
            labels_vec.push_back(max_class_idx);

            // Extract mask logits for this detection
            cv::Mat mask_logits(mask_h, mask_w, CV_32F);
            for (size_t r = 0; r < mask_h; ++r) {
                for (size_t c = 0; c < mask_w; ++c) {
                    const size_t idx = mask_offset + r * mask_w + c;
                    try {
                        mask_logits.at<float>(r, c) = std::get<float>(masks_raw[idx]);
                    } catch (const std::bad_variant_access&) {
                        throw std::runtime_error("Invalid TensorElement type for mask at index " + std::to_string(idx));
                    }
                }
            }
            mask_proposals.push_back(mask_logits);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, confThreshold, iouThreshold, indices);

    // Second pass: generate final segmentation results with masks
    std::vector<Result> final_results;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        InstanceSegmentation seg;
        seg.bbox = bboxes[idx];
        seg.class_confidence = scores[idx];
        seg.class_id = labels_vec[idx];

        // Process mask
        cv::Mat mask = mask_proposals[idx];
        
        // Apply sigmoid activation to mask logits
        cv::exp(-mask, mask);
        mask = 1.0 / (1.0 + mask);

        // Resize mask to input size (640x640)
        cv::Mat mask_resized;
        cv::resize(mask, mask_resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);

        // Resize to frame size
        cv::resize(mask_resized, mask, frame_size, 0, 0, cv::INTER_LINEAR);

        // Ensure bbox is within frame boundaries
        cv::Rect safeBbox = seg.bbox & cv::Rect(0, 0, frame_size.width, frame_size.height);
        if (safeBbox.width > 0 && safeBbox.height > 0) {
            // Crop mask to bbox region
            mask = mask(safeBbox);
            
            // Apply threshold
            mask = mask > mask_threshold;

            // Store mask data
            seg.mask_data.assign(mask.data, mask.data + mask.total());
            seg.mask_height = mask.rows;
            seg.mask_width = mask.cols;

            final_results.push_back(seg);
        }
    }

    return final_results;
}
