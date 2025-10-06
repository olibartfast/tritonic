#include "TimeSformer.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <random>

TimeSformer::TimeSformer(const TritonModelInfo& model_info)
    : TaskInterface(model_info), num_frames_(DEFAULT_NUM_FRAMES) {
    
    auto& logger = Logger::getInstance();
    
    // Validate model inputs/outputs for TimeSformer
    if (model_info.input_names.empty()) {
        throw std::runtime_error("TimeSformer model must have at least one input");
    }
    
    if (model_info.output_names.empty()) {
        throw std::runtime_error("TimeSformer model must have at least one output");
    }
    
    logger.infof("TimeSformer initialized with input: {}, output: {}", 
                 model_info.input_names[0], model_info.output_names[0]);    
    
    // Validate input shape (should be [batch, num_frames, channels, height, width] or [batch, channels, num_frames, height, width])
    if (model_info.input_shapes.empty() || model_info.input_shapes[0].size() != 5) {
        throw std::runtime_error("TimeSformer model input should have 5 dimensions [B, T, C, H, W] or [B, C, T, H, W]");
    }
    
    const auto& input_shape = model_info.input_shapes[0];
    
    // Determine format and extract num_frames
    if (model_info.input_formats[0] == "FORMAT_NHWC") {
        // Format: [B, T, H, W, C] - unusual for video models
        num_frames_ = input_shape[1];
    } else {
        // Typical formats: [B, T, C, H, W] or [B, C, T, H, W]
        // We need to determine which dimension is temporal
        if (input_shape[1] <= 32) { // Assume temporal dimension is small (8, 16, etc.)
            num_frames_ = input_shape[1]; // [B, T, C, H, W]
        } else if (input_shape[2] <= 32) {
            num_frames_ = input_shape[2]; // [B, C, T, H, W]
        } else {
            num_frames_ = DEFAULT_NUM_FRAMES; // Fallback
            logger.warnf("Could not determine temporal dimension from shape, using default: {}", num_frames_);
        }
    }
    
    logger.infof("TimeSformer configuration: num_frames={}, input_shape=[{}, {}, {}, {}, {}]", 
                 num_frames_, input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]);
    
    // Load action labels
    loadActionLabels();
}

void TimeSformer::loadActionLabels() {
    auto& logger = Logger::getInstance();
    
    // Try to load Kinetics-400 labels
    std::vector<std::string> possible_paths = {
        "labels/kinetics400.txt",
        "labels/kinetics.txt", 
        "../labels/kinetics400.txt",
        "../../labels/kinetics400.txt"
    };
    
    for (const auto& path : possible_paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    class_names_.push_back(line);
                }
            }
            file.close();
            logger.infof("Loaded {} action labels from: {}", class_names_.size(), path);
            return;
        }
    }
    
    // If no label file found, create default numeric labels
    for (int i = 0; i < 400; ++i) {
        class_names_.push_back("action_" + std::to_string(i));
    }
    logger.warn("No action label file found, using default numeric labels");
}

std::vector<cv::Mat> TimeSformer::sample_frames(const std::vector<cv::Mat>& frames, int target_frames) {
    std::vector<cv::Mat> sampled_frames;
    
    if (frames.empty()) {
        throw std::runtime_error("No frames provided for sampling");
    }
    
    if (frames.size() <= target_frames) {
        // If we have fewer frames than needed, duplicate the last frame
        sampled_frames = frames;
        while (sampled_frames.size() < target_frames) {
            sampled_frames.push_back(frames.back().clone());
        }
    } else {
        // Sample frames uniformly
        float step = static_cast<float>(frames.size()) / target_frames;
        for (int i = 0; i < target_frames; ++i) {
            int frame_idx = static_cast<int>(i * step);
            frame_idx = std::min(frame_idx, static_cast<int>(frames.size() - 1));
            sampled_frames.push_back(frames[frame_idx].clone());
        }
    }
    
    return sampled_frames;
}

void TimeSformer::apply_imagenet_normalization(cv::Mat& image) {
    // Convert from BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    
    // Apply ImageNet normalization
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i];
    }
    
    cv::merge(channels, image);
}

std::vector<uint8_t> TimeSformer::preprocess_video_clip(const std::vector<cv::Mat>& frames, 
                                                       const std::string& format,
                                                       int img_type1, 
                                                       int img_type3,
                                                       size_t img_channels, 
                                                       const cv::Size& img_size) {
    auto& logger = Logger::getInstance();
    std::vector<uint8_t> input_data;
    
    if (frames.empty()) {
        throw std::runtime_error("No frames provided for preprocessing");
    }
    
    // Sample frames to match model requirements
    std::vector<cv::Mat> sampled_frames = sample_frames(frames, num_frames_);
    
    logger.debugf("TimeSformer preprocessing: {} frames -> {} frames, target size: {}x{}", 
                  frames.size(), sampled_frames.size(), img_size.width, img_size.height);
    
    // Prepare containers for processed frames
    std::vector<cv::Mat> processed_frames;
    processed_frames.reserve(num_frames_);
    
    // Process each frame
    for (auto& frame : sampled_frames) {
        cv::Mat processed_frame;
        frame.copyTo(processed_frame);
        
        // Resize to target size
        cv::resize(processed_frame, processed_frame, img_size);
        
        // Apply ImageNet normalization
        apply_imagenet_normalization(processed_frame);
        
        processed_frames.push_back(processed_frame);
    }
    
    // Calculate total size needed
    size_t frame_byte_size = img_size.width * img_size.height * img_channels * sizeof(float);
    size_t total_byte_size = frame_byte_size * num_frames_;
    input_data.resize(total_byte_size);
    
    // Pack data based on format
    size_t pos = 0;
    
    if (format == "FORMAT_NHWC") {
        // Format: [T, H, W, C] - frame by frame
        for (int t = 0; t < num_frames_; ++t) {
            const cv::Mat& frame = processed_frames[t];
            std::memcpy(&input_data[pos], frame.data, frame_byte_size);
            pos += frame_byte_size;
        }
    } else {
        // Format: [T, C, H, W] or [C, T, H, W] - channel by channel
        for (int t = 0; t < num_frames_; ++t) {
            const cv::Mat& frame = processed_frames[t];
            std::vector<cv::Mat> channels;
            cv::split(frame, channels);
            
            for (size_t c = 0; c < img_channels; ++c) {
                size_t channel_size = img_size.width * img_size.height * sizeof(float);
                std::memcpy(&input_data[pos], channels[c].data, channel_size);
                pos += channel_size;
            }
        }
    }
    
    if (pos != total_byte_size) {
        logger.errorf("Unexpected total size of data {}, expecting {}", pos, total_byte_size);
        throw std::runtime_error("Data packing size mismatch");
    }
    
    return input_data;
}

std::vector<std::vector<uint8_t>> TimeSformer::preprocess(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) {
        throw std::runtime_error("No frames provided for TimeSformer preprocessing");
    }
    
    const auto& input_format = model_info_.input_formats[0];
    const auto input_datatype = model_info_.input_datatypes[0];
    
    int img_type1 = (input_datatype == "TYPE_FP32") ? CV_32FC1 : CV_8UC1;
    int img_type3 = (input_datatype == "TYPE_FP32") ? CV_32FC3 : CV_8UC3;
    
    const cv::Size img_size(input_width_, input_height_);
    
    std::vector<uint8_t> input_data = preprocess_video_clip(frames, input_format, img_type1, img_type3, 
                                                           input_channels_, img_size);
    
    return {input_data};
}

std::vector<float> TimeSformer::apply_softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    
    // Normalize
    for (auto& prob : probabilities) {
        prob /= sum;
    }
    
    return probabilities;
}

std::vector<Result> TimeSformer::postprocess(const cv::Size& frame_size,
                                            const std::vector<std::vector<TensorElement>>& infer_results,
                                            const std::vector<std::vector<int64_t>>& infer_shapes) {
    auto& logger = Logger::getInstance();
    std::vector<Result> classification_results;
    
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Inference results or shapes are empty.");
    }

    const auto& output_result = infer_results[0];
    const auto& output_shape = infer_shapes[0];
    
    logger.debugf("TimeSformer postprocess: output_shape=[{}], result_size={}", 
                  output_shape.size() > 1 ? std::to_string(output_shape[1]) : "unknown", 
                  output_result.size());
    
    // Convert TensorElement to float
    std::vector<float> logits;
    logits.reserve(output_result.size());
    for (const auto& elem : output_result) {
        logits.push_back(std::visit([](auto&& arg) -> float { 
            return static_cast<float>(arg); 
        }, elem));
    }
    
    // Apply softmax to get probabilities
    std::vector<float> probabilities = apply_softmax(logits);
    
    // Find top predictions
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&probabilities](size_t i1, size_t i2) { 
                  return probabilities[i1] > probabilities[i2]; 
              });
    
    // Generate results for top predictions
    size_t num_predictions = std::min(TOP_K_PREDICTIONS, indices.size());
    for (size_t i = 0; i < num_predictions; ++i) {
        size_t class_idx = indices[i];
        float confidence = probabilities[class_idx];
        
        if (confidence >= CONFIDENCE_THRESHOLD) {
            try {
                VideoClassification video_classification;
                video_classification.class_id = static_cast<float>(class_idx);
                video_classification.class_confidence = confidence;
                
                // Set action label
                if (class_idx < class_names_.size()) {
                    video_classification.action_label = class_names_[class_idx];
                } else {
                    video_classification.action_label = "action_" + std::to_string(class_idx);
                }
                
                classification_results.emplace_back(video_classification);
                
                logger.infof("TimeSformer: Action {} ({}): {:.3f}", 
                             class_idx, video_classification.action_label, confidence);
            } catch (const std::exception& e) {
                logger.errorf("TimeSformer: Failed to create Result object: {}", e.what());
            }
        }
    }
    
    if (classification_results.empty()) {
        logger.warn("TimeSformer: No predictions above confidence threshold");
        
        // Add top prediction even if below threshold
        if (!indices.empty()) {
            size_t class_idx = indices[0];
            float confidence = probabilities[class_idx];
            
            VideoClassification video_classification;
            video_classification.class_id = static_cast<float>(class_idx);
            video_classification.class_confidence = confidence;
            
            if (class_idx < class_names_.size()) {
                video_classification.action_label = class_names_[class_idx];
            } else {
                video_classification.action_label = "action_" + std::to_string(class_idx);
            }
            
            classification_results.emplace_back(video_classification);
        }
    }
    
    return classification_results;
}
