#include "ViTClassifier.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

ViTClassifier::ViTClassifier(const TritonModelInfo& model_info)
    : TaskInterface(model_info) {
    
    // Validate model inputs/outputs for ViT
    if (model_info.input_names.empty()) {
        throw std::runtime_error("ViT model must have at least one input");
    }
    
    if (model_info.output_names.empty()) {
        throw std::runtime_error("ViT model must have at least one output");
    }
    

    logger.infof("ViT Classifier initialized with input: {}, output: {}", model_info.input_names[0],  model_info.output_names[0]);    
    
    // Validate input shape (should be [batch, channels, height, width] or [batch, height, width, channels])
    if (model_info.input_shapes.empty() || model_info.input_shapes[0].size() != 4) {
        throw std::runtime_error("ViT model input should have 4 dimensions [B, C, H, W] or [B, H, W, C]");
    }
    
    const auto& input_shape = model_info.input_shapes[0];

    logger.infof("Input shape: [{}]", 
        std::to_string(input_shape[0]) + ", " + 
        std::to_string(input_shape[1]) + ", " + 
        std::to_string(input_shape[2]) + ", " + 
        std::to_string(input_shape[3]));    
}

void ViTClassifier::apply_imagenet_normalization(cv::Mat& image) {
    // Convert to float if not already
    if (image.type() != CV_32FC3) {
        image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
    }
    
    // Apply ImageNet normalization: (pixel - mean) / std
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i];
    }
    
    cv::merge(channels, image);
}

std::vector<uint8_t> ViTClassifier::preprocess_image(const cv::Mat& img, 
                                                    const std::string& format,
                                                    int img_type1, 
                                                    int img_type3,
                                                    size_t img_channels, 
                                                    const cv::Size& img_size) {
    
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    
    cv::Mat processed_img;
    
    // Convert BGR to RGB (OpenCV uses BGR by default)
    if (img.channels() == 3) {
        cv::cvtColor(img, processed_img, cv::COLOR_BGR2RGB);
    } else {
        processed_img = img.clone();
    }
    
    // Resize to target size (typically 224x224 or 384x384 for ViT)
    cv::resize(processed_img, processed_img, img_size, 0, 0, cv::INTER_LINEAR);
    
    // Convert to float and apply normalization
    apply_imagenet_normalization(processed_img);
    
    std::vector<uint8_t> input_data;
    
    // Convert to the required format
    if (format == "FORMAT_NCHW" || format == "NCHW") {
        // Channel-first format: [C, H, W]
        std::vector<cv::Mat> channels;
        cv::split(processed_img, channels);
        
        for (const auto& channel : channels) {
            const float* data = channel.ptr<float>();
            const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
            size_t channel_size = channel.total() * sizeof(float);
            input_data.insert(input_data.end(), byte_data, byte_data + channel_size);
        }
    } else {
        // Channel-last format: [H, W, C] (default for NHWC)
        const float* data = processed_img.ptr<float>();
        const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
        size_t total_size = processed_img.total() * processed_img.channels() * sizeof(float);
        input_data.assign(byte_data, byte_data + total_size);
    }
    

    logger.infof("Preprocessed image: {}x{}, format: {}, data size: {} bytes", 
        img_size.width, img_size.height, format, input_data.size());    
    
    return input_data;
}

std::vector<std::vector<uint8_t>> ViTClassifier::preprocess(const std::vector<cv::Mat>& imgs) {
    if (imgs.empty()) {
        throw std::runtime_error("No input images provided");
    }
    
    std::vector<std::vector<uint8_t>> input_data;
    input_data.reserve(imgs.size());
    
    // Get model input configuration
    const std::string& format = model_info_.input_formats.empty() ? "NCHW" : model_info_.input_formats[0];
    const auto& input_shape = model_info_.input_shapes[0];
    
    // Determine input size based on format
    cv::Size img_size;
    size_t img_channels;
    
    if (format == "FORMAT_NCHW" || format == "NCHW") {
        // [B, C, H, W]
        img_channels = input_shape[1];
        img_size = cv::Size(input_shape[3], input_shape[2]); // width, height
    } else {
        // [B, H, W, C] 
        img_channels = input_shape[3];
        img_size = cv::Size(input_shape[2], input_shape[1]); // width, height
    }
    
    logger.infof("Processing {} images, target size: {}x{}", 
        imgs.size(), img_size.width, img_size.height);
    
    for (const auto& img : imgs) {
        input_data.push_back(preprocess_image(img, format, 
                                            model_info_.type1_, 
                                            model_info_.type3_, 
                                            img_channels, 
                                            img_size));
    }
    
    return input_data;
}

std::vector<float> ViTClassifier::apply_softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return {};
    }
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exponentials
    std::vector<float> exp_values;
    exp_values.reserve(logits.size());
    
    float sum = 0.0f;
    for (float logit : logits) {
        float exp_val = std::exp(logit - max_logit);
        exp_values.push_back(exp_val);
        sum += exp_val;
    }
    
    // Normalize to get probabilities
    std::vector<float> probabilities;
    probabilities.reserve(exp_values.size());
    
    for (float exp_val : exp_values) {
        probabilities.push_back(exp_val / sum);
    }
    
    return probabilities;
}

std::vector<Result> ViTClassifier::postprocess(const cv::Size& frame_size,
                                              const std::vector<std::vector<TensorElement>>& infer_results,
                                              const std::vector<std::vector<int64_t>>& infer_shapes) {
    
    logger.infof("ViT postprocess: Starting");
    
    // Check if the input vectors are not empty - same as TensorflowClassifier
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Inference results or shapes are empty.");
    }

    // Get the first set of results and shape - same as TensorflowClassifier
    const auto& results = infer_results.front();
    const auto shape = infer_shapes[0][1];
    
    logger.infof("ViT postprocess: results size={}, shape={}", results.size(), shape);
    
    // Quick validation
    if (shape != static_cast<int64_t>(results.size())) {
        logger.errorf("Shape mismatch: shape={}, results.size()={}", shape, results.size());
        return {};
    }

    logger.infof("ViT postprocess: Size validation passed");
    
    // For ViT logits, find the maximum value and its index
    float max_logit = std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, results[0]);
    int best_class = 0;
    
    for (size_t i = 1; i < results.size(); ++i) {
        float logit = std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, results[i]);
        if (logit > max_logit) {
            max_logit = logit;
            best_class = static_cast<int>(i);
        }
    }
    
    logger.infof("ViT prediction: class {} with logit {}", best_class, max_logit);
    
    // Convert logits to probabilities using softmax to get proper confidence
    std::vector<float> logits;
    logits.reserve(results.size());
    for (const auto& element : results) {
        float value = std::visit([](auto&& arg) -> float {
            return static_cast<float>(arg);
        }, element);
        logits.push_back(value);
    }
    
    std::vector<float> probabilities = apply_softmax(logits);
    float confidence = probabilities[best_class];
    
    logger.infof("ViT confidence after softmax: {}", confidence);
    
    logger.infof("ViT postprocess: Prediction - Class: {}, Logit: {}, Confidence: {}", 
                 best_class, max_logit, confidence);
    
    // Now let's try to create the Result object with the correct approach
    std::vector<Result> classification_results;
    
    try {
        // Create Classification object and add directly to results
        Classification classification;
        classification.class_id = static_cast<float>(best_class);
        classification.class_confidence = confidence;
        classification_results.emplace_back(classification);
        
        logger.infof("ViT classifier: Successfully created Result object for class {} with confidence {}", 
                     best_class, confidence);
    } catch (const std::exception& e) {
        logger.errorf("ViT classifier: Failed to create Result object: {}", e.what());
    }
    
    return classification_results;
}
