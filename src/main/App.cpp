#include "App.hpp"
#include <deque>
#include <filesystem>
#include <random>
#include <sstream>
#include <thread>
#include <chrono>
#include "vision-core/core/task_factory.hpp"


App::App(std::shared_ptr<ITriton> triton, 
         std::shared_ptr<vision_infra::config::InferenceConfig> config,
         std::shared_ptr<vision_infra::core::Logger> logger)
    : tritonClient_(triton), config_(config), logger_(logger) {}

int App::run() {
    try {
        logger_->Info("Starting Triton Client Application");
        logger_->Info("Current path is " + std::string(std::filesystem::current_path()));

        // Connect to Triton
        tritonClient_->createTritonClient();

        // Check if server is live with retries
        logger_->Info("Checking if Triton server is live...");
        int max_retries = 5;
        int retry_count = 0;
        bool server_live = false;
        
        while (retry_count < max_retries) {
            if (tritonClient_->isServerLive()) {
                server_live = true;
                break;
            }
            logger_->Warn("Server not live, retrying in 2 seconds... (" + std::to_string(retry_count + 1) + "/" + std::to_string(max_retries) + ")");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            retry_count++;
        }
        
        if (!server_live) {
            throw std::runtime_error("Triton server is not live after " + std::to_string(max_retries) + " attempts");
        }
        logger_->Info("Triton server is live.");

        // Check if model is in repository
        logger_->Info("Checking if model " + config_->GetModelName() + " is in repository...");
        if (!tritonClient_->isModelInRepository(config_->GetModelName())) {
             throw std::runtime_error("Model " + config_->GetModelName() + " not found in Triton repository.");
        }
        logger_->Info("Model found in repository.");

        logger_->Info("Checking if model " + config_->GetModelName() + " is ready...");
        if (!tritonClient_->isModelReady(config_->GetModelName())) {
            logger_->Info("Model is not ready. Attempting to load...");
            tritonClient_->loadModel(config_->GetModelName());
        } else {
            logger_->Info("Model is ready.");
        }

        logger_->Info("Getting model info for: " + config_->GetModelName());
        TritonModelInfo modelInfo = tritonClient_->getModelInfo(config_->GetModelName(), config_->GetServerAddress(), config_->GetInputSizes());

        // Create task instance
        logger_->Info("Creating task instance for model type: " + config_->GetModelType());
        auto visionCoreModelInfo = convertToVisionCoreModelInfo(modelInfo);
        vision_core::TaskConfig taskConfig;
        taskConfig.confidence_threshold = config_->GetConfidenceThreshold();
        taskConfig.nms_threshold = config_->GetNmsThreshold();
        task_ = vision_core::TaskFactory::createTaskInstance(config_->GetModelType(), visionCoreModelInfo, taskConfig);

        // Extract frame buffer size for video classification from 5D input shape [B, T, C, H, W]
        if (task_->getTaskType() == vision_core::TaskType::VideoClassification &&
            !modelInfo.input_shapes.empty()) {
            const auto& shape = modelInfo.input_shapes[0];
            if (shape.size() == 5) {
                num_frames_ = static_cast<int>(shape[1]);
            }
            logger_->Info("Video classification frame buffer size: " + std::to_string(num_frames_));
        }

        if (!task_) {
            throw std::runtime_error("Failed to create task instance");
        }

        // Load class names
        class_names_ = task_->readLabelNames(config_->GetLabelsFile());
        logger_->Info("Loaded " + std::to_string(class_names_.size()) + " class names from " + config_->GetLabelsFile());

        // Parse source files
        std::vector<std::string> sourceNames = split(config_->GetSource(), ',');
        
        // Categorize source files
        std::vector<std::string> image_list;
        std::vector<std::string> video_list;
        for (const auto& sourceName : sourceNames) {
            if (isImageFile(sourceName)) {
                image_list.push_back(sourceName);
                logger_->Debug("Added image file: " + sourceName);
            } else if (isVideoFile(sourceName)) {
                video_list.push_back(sourceName);
                logger_->Debug("Added video file: " + sourceName);
            } else {
                logger_->Warn("Unknown file type: " + sourceName);
            }
        }

        if (image_list.empty() && video_list.empty()) {
            throw std::runtime_error("No valid image or video files provided");
        }
        
        logger_->Info("Processing " + std::to_string(image_list.size()) + " images and " + std::to_string(video_list.size()) + " videos");
        
        // Process images
        if (!image_list.empty()) {
            processImages(image_list);
        }
        
        // Process videos
        if (!video_list.empty()) {
            logger_->Info("Processing videos");
            for (const auto& sourceName : video_list) {
                if (task_->getTaskType() == vision_core::TaskType::VideoClassification) {
                    processVideoClassification(sourceName);
                } else {
                    processVideo(sourceName);
                }
            }
        }

        logger_->Info("Application completed successfully");
        return 0;

    } catch (const std::exception& e) {
        logger_->Error("Application error: " + std::string(e.what()));
        return 1;
    } catch (...) {
        logger_->Fatal("An unknown error occurred");
        return 1;
    }
}

vision_core::ModelInfo App::convertToVisionCoreModelInfo(const TritonModelInfo& triton_info) {
    vision_core::ModelInfo model_info;
    model_info.input_shapes = triton_info.input_shapes;
    model_info.input_formats = triton_info.input_formats;
    model_info.input_names = triton_info.input_names;
    model_info.output_names = triton_info.output_names;
    model_info.input_types = triton_info.input_types;
    model_info.max_batch_size_ = triton_info.max_batch_size_;
    model_info.batch_size_ = triton_info.batch_size_;
    return model_info;
}

std::vector<vision_core::Result> App::processSource(const std::vector<cv::Mat>& source) {
    const auto input_data = task_->preprocess(source);
    auto tensors = tritonClient_->infer(input_data);
    std::vector<vision_core::Tensor> vision_tensors;
    vision_tensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        vision_tensors.emplace_back(tensor.data, tensor.shape);
    }
    return task_->postprocess(cv::Size(source.front().cols, source.front().rows), vision_tensors);
}

void App::processImages(const std::vector<std::string>& sourceNames) {
    if (task_->getTaskType() == vision_core::TaskType::OpticalFlow) {
        logger_->Info("Processing optical flow for image pairs");
        for(size_t i = 0; i < sourceNames.size() - 1; i++) {
            std::vector<std::string> flowInputs = {sourceNames[i], sourceNames[i+1]};

            std::vector<cv::Mat> images;
            for (const auto& name : flowInputs) {
                cv::Mat img = cv::imread(name);
                if (img.empty()) {
                    logger_->Error("Could not open or read the image: " + name);
                    continue;
                }
                images.push_back(img);
            }

            if (images.size() != 2) continue;

            auto start = std::chrono::steady_clock::now();
            std::vector<vision_core::Result> predictions = processSource(images);
            auto end = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            logger_->Info("Infer time for " + std::to_string(images.size()) + " images: " + std::to_string(diff) + " ms");

            cv::Mat& image = images[0];
            for (const auto& prediction : predictions) {
                if (std::holds_alternative<vision_core::OpticalFlow>(prediction)) {
                    vision_core::OpticalFlow flow = std::get<vision_core::OpticalFlow>(prediction);
                    flow.flow.copyTo(image);
                }
            }

            std::string sourceDir = flowInputs[0].substr(0, flowInputs[0].find_last_of("/\\"));
            std::string outputDir = sourceDir + "/output";
            std::filesystem::create_directories(outputDir);
            std::string processedFrameFilename = outputDir + "/processed_frame_" + config_->GetModelName() + ".jpg";
            logger_->Info("Saving frame to: " + processedFrameFilename);
            cv::imwrite(processedFrameFilename, image);
        }
    } else if (task_->getTaskType() == vision_core::TaskType::VideoClassification) {
        logger_->Info("Processing video classification for image set (" + std::to_string(sourceNames.size()) + " frames)");
        std::vector<cv::Mat> frames;
        for (const auto& name : sourceNames) {
            cv::Mat img = cv::imread(name);
            if (img.empty()) {
                logger_->Warn("Could not open image: " + name);
                continue;
            }
            frames.push_back(img);
        }
        if (frames.empty()) return;

        auto start = std::chrono::steady_clock::now();
        std::vector<vision_core::Result> predictions = processSource(frames);
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger_->Info("Infer time for " + std::to_string(frames.size()) + " frames: " + std::to_string(diff) + " ms");

        for (const auto& prediction : predictions) {
            if (std::holds_alternative<vision_core::VideoClassification>(prediction)) {
                const auto& vc = std::get<vision_core::VideoClassification>(prediction);
                std::string label = vc.action_label.empty()
                    ? (class_names_.empty() ? "Unknown" : class_names_[static_cast<int>(vc.class_id)])
                    : vc.action_label;
                logger_->Info("Action: " + label + " (" + std::to_string(vc.class_confidence).substr(0, 4) + ")");
            }
        }
    } else {
        logger_->Info("Processing individual images");
        for (const auto& sourceName : sourceNames) {
            cv::Mat image = cv::imread(sourceName);
            if (image.empty()) {
                logger_->Error("Could not open or read the image: " + sourceName);
                continue;
            }

            auto start = std::chrono::steady_clock::now();
            std::vector<vision_core::Result> predictions = processSource({image});
            auto end = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            logger_->Info("Infer time for 1 image: " + std::to_string(diff) + " ms");

            std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
            for (const auto& prediction : predictions) {
                if (std::holds_alternative<vision_core::Classification>(prediction)) {
                    const auto& classification = std::get<vision_core::Classification>(prediction);
                    logger_->Info("Image " + sourceName + ": " + class_names_[static_cast<int>(classification.class_id)] + ": " + std::to_string(classification.class_confidence));
                    drawLabel(image, class_names_[static_cast<int>(classification.class_id)], classification.class_confidence, 30, 30);
                }
                else if (std::holds_alternative<vision_core::Detection>(prediction)) {
                    const auto& detection = std::get<vision_core::Detection>(prediction);
                    cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 2);
                    drawLabel(image, class_names_[static_cast<int>(detection.class_id)], detection.class_confidence,
                              detection.bbox.x, detection.bbox.y - 1);
                }
                else if (std::holds_alternative<vision_core::InstanceSegmentation>(prediction)) {
                    const auto& segmentation = std::get<vision_core::InstanceSegmentation>(prediction);

                    cv::rectangle(image, segmentation.bbox, cv::Scalar(255, 0, 0), 2);
                    drawLabel(image, class_names_[static_cast<int>(segmentation.class_id)], segmentation.class_confidence,
                              segmentation.bbox.x, segmentation.bbox.y - 1);

                    cv::Mat mask;
                    if (!segmentation.mask.empty()) {
                        mask = segmentation.mask;
                    } else if (!segmentation.mask_data.empty()) {
                        mask = cv::Mat(segmentation.mask_height, segmentation.mask_width,
                                      CV_8UC1, const_cast<uint8_t*>(segmentation.mask_data.data()));
                    }

                    if (!mask.empty()) {
                        cv::Mat colorMask = cv::Mat::zeros(image.size(), CV_8UC3);
                        cv::Scalar color = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
                        colorMask.setTo(color, mask);
                        cv::addWeighted(image, 1, colorMask, 0.7, 0, image);
                    }
                }
                else if (std::holds_alternative<vision_core::PoseEstimation>(prediction)) {
                    const auto& pose = std::get<vision_core::PoseEstimation>(prediction);
                    drawPose(image, pose);
                }
                else if (std::holds_alternative<vision_core::DepthEstimation>(prediction)) {
                    const auto& depth = std::get<vision_core::DepthEstimation>(prediction);
                    if (!depth.normalized_depth.empty()) {
                        cv::Mat depth_8u;
                        depth.normalized_depth.convertTo(depth_8u, CV_8UC1, 255.0);
                        cv::applyColorMap(depth_8u, image, cv::COLORMAP_INFERNO);
                    }
                    logger_->Info("Depth range: " + std::to_string(depth.min_depth) + " - " + std::to_string(depth.max_depth));
                }
            }

            std::string outputDir = sourceDir + "/output";
            std::filesystem::create_directories(outputDir);
            std::string processedFrameFilename = outputDir + "/processed_frame_" + config_->GetModelName() + ".jpg";
            logger_->Info("Saving frame to: " + processedFrameFilename);
            cv::imwrite(processedFrameFilename, image);
        }
    }
}

void App::processVideo(const std::string& sourceName) {
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    cv::VideoCapture cap(sourceName);
    if (!cap.isOpened()) {
        logger_->Error("Could not open the video: " + sourceName);
        throw std::runtime_error("Could not open the video: " + sourceName);
    }

    cv::VideoWriter outputVideo;
    if (config_->GetWriteFrame()) {
        cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        std::string outputDir = sourceDir + "/output";
        std::filesystem::create_directories(outputDir);
        outputVideo.open(outputDir + "/processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);

        if (!outputVideo.isOpened()) {
            logger_->Error("Could not open the output video for write: " + sourceName);
            return;
        }
    }

    cv::Mat current_frame, previous_frame, visualization_frame;
    std::vector<cv::Scalar> colors = generateRandomColors(class_names_.size());
    
    // Read first frame
    if (!cap.read(current_frame)) {
        logger_->Error("Failed to read first frame");
        throw std::runtime_error("Failed to read first frame");
    }

    while (true) {
        auto start = std::chrono::steady_clock::now();
        std::vector<vision_core::Result> predictions;

        if (task_->getTaskType() == vision_core::TaskType::OpticalFlow) {
            if (!previous_frame.empty()) {
                // Process optical flow between previous and current frame
                std::vector<cv::Mat> frame_pair = {previous_frame, current_frame};
                predictions = processSource(frame_pair);
            }
        } else {
            // Process single frame for other tasks
            predictions = processSource({current_frame});
        }

        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger_->Info("Infer time: " + std::to_string(diff) + " ms");

        if (config_->GetShowFrame() || config_->GetWriteFrame()) {
            // Create visualization frame
            if (task_->getTaskType() == vision_core::TaskType::OpticalFlow) {
                visualization_frame = cv::Mat::zeros(current_frame.size(), current_frame.type());
                for (const auto& prediction : predictions) {
                    if (std::holds_alternative<vision_core::OpticalFlow>(prediction)) {
                        vision_core::OpticalFlow flow = std::get<vision_core::OpticalFlow>(prediction);
                        flow.flow.copyTo(visualization_frame);
                    }
                }
            } else {
                current_frame.copyTo(visualization_frame);
                for (const auto& prediction : predictions) {
                    if (std::holds_alternative<vision_core::Detection>(prediction)) {
                        vision_core::Detection detection = std::get<vision_core::Detection>(prediction);
                        cv::Rect safeBbox = detection.bbox & cv::Rect(0, 0, visualization_frame.cols, visualization_frame.rows);

                        if (safeBbox.width > 0 && safeBbox.height > 0) {
                            cv::rectangle(visualization_frame, safeBbox, colors[static_cast<int>(detection.class_id)], 2);
                            drawLabel(visualization_frame, class_names_[static_cast<int>(detection.class_id)], detection.class_confidence,
                                     safeBbox.x, safeBbox.y - 1);
                        }
                    }
                    else if (std::holds_alternative<vision_core::InstanceSegmentation>(prediction)) {
                        vision_core::InstanceSegmentation segmentation = std::get<vision_core::InstanceSegmentation>(prediction);

                        cv::Rect safeBbox = segmentation.bbox & cv::Rect(0, 0, visualization_frame.cols, visualization_frame.rows);

                        if (safeBbox.width > 0 && safeBbox.height > 0) {
                            cv::rectangle(visualization_frame, safeBbox, colors[static_cast<int>(segmentation.class_id)], 2);
                            drawLabel(visualization_frame, class_names_[static_cast<int>(segmentation.class_id)], segmentation.class_confidence,
                                     safeBbox.x, safeBbox.y - 1);

                            if (!segmentation.mask_data.empty() && segmentation.mask_height > 0 && segmentation.mask_width > 0) {
                                cv::Mat mask(segmentation.mask_height, segmentation.mask_width, CV_8UC1);
                                std::memcpy(mask.data, segmentation.mask_data.data(), segmentation.mask_data.size());
                                cv::resize(mask, mask, safeBbox.size(), 0, 0, cv::INTER_NEAREST);

                                cv::Mat colorMask = cv::Mat::zeros(safeBbox.size(), CV_8UC3);
                                colorMask.setTo(colors[static_cast<int>(segmentation.class_id)], mask);

                                cv::Mat roi = visualization_frame(safeBbox);
                                if (roi.size() == colorMask.size()) {
                                    cv::addWeighted(roi, 1, colorMask, 0.5, 0, roi);
                                }
                            }
                        }
                    }
                    else if (std::holds_alternative<vision_core::PoseEstimation>(prediction)) {
                        const auto& pose = std::get<vision_core::PoseEstimation>(prediction);
                        drawPose(visualization_frame, pose);
                    }
                    else if (std::holds_alternative<vision_core::DepthEstimation>(prediction)) {
                        const auto& depth = std::get<vision_core::DepthEstimation>(prediction);
                        if (!depth.normalized_depth.empty()) {
                            cv::Mat depth_8u;
                            depth.normalized_depth.convertTo(depth_8u, CV_8UC1, 255.0);
                            cv::applyColorMap(depth_8u, visualization_frame, cv::COLORMAP_INFERNO);
                        }
                    }
                }
            }

            // Add FPS counter
            double fps = 1000.0 / static_cast<double>(diff);
            std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 4);
            cv::putText(visualization_frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        if (config_->GetShowFrame()) {
            cv::imshow("video feed", visualization_frame);
            cv::waitKey(1);
        }

        if (config_->GetWriteFrame()) {
            outputVideo.write(visualization_frame);
        }

        // Store current frame as previous frame
        current_frame.copyTo(previous_frame);
        
        // Read next frame
        if (!cap.read(current_frame)) {
            break;
        }
    }
}

void App::processVideoClassification(const std::string& sourceName) {
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    cv::VideoCapture cap(sourceName);
    if (!cap.isOpened()) {
        logger_->Error("Could not open the video: " + sourceName);
        throw std::runtime_error("Could not open the video: " + sourceName);
    }

    cv::VideoWriter outputVideo;
    if (config_->GetWriteFrame()) {
        cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        std::string outputDir = sourceDir + "/output";
        std::filesystem::create_directories(outputDir);
        outputVideo.open(outputDir + "/processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);
        if (!outputVideo.isOpened()) {
            logger_->Warn("Could not open output video for write: " + sourceName);
        }
    }

    std::deque<cv::Mat> frame_buffer;
    cv::Mat frame;
    std::string last_label;
    float last_confidence = 0.0f;

    while (cap.read(frame)) {
        frame_buffer.push_back(frame.clone());
        if ((int)frame_buffer.size() > num_frames_) {
            frame_buffer.pop_front();
        }

        cv::Mat display_frame = frame.clone();

        if ((int)frame_buffer.size() == num_frames_) {
            auto start = std::chrono::steady_clock::now();
            std::vector<cv::Mat> frames(frame_buffer.begin(), frame_buffer.end());
            std::vector<vision_core::Result> predictions = processSource(frames);
            auto end = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            logger_->Info("Infer time: " + std::to_string(diff) + " ms");

            for (const auto& prediction : predictions) {
                if (std::holds_alternative<vision_core::VideoClassification>(prediction)) {
                    const auto& vc = std::get<vision_core::VideoClassification>(prediction);
                    last_label = vc.action_label.empty()
                        ? (class_names_.empty() ? "Unknown" : class_names_[static_cast<int>(vc.class_id)])
                        : vc.action_label;
                    last_confidence = vc.class_confidence;
                }
            }
        }

        if (!last_label.empty()) {
            drawLabel(display_frame, "Action: " + last_label, last_confidence, 10, 60);
        }

        if (config_->GetShowFrame()) {
            cv::imshow("video feed", display_frame);
            if (cv::waitKey(1) == 27) break;
        }
        if (config_->GetWriteFrame() && outputVideo.isOpened()) {
            outputVideo.write(display_frame);
        }
    }
}

void App::drawPose(cv::Mat& image, const vision_core::PoseEstimation& pose, float confidence_threshold) {
    // COCO 17-keypoint skeleton connections
    static const std::vector<std::pair<int,int>> kSkeleton = {
        {0,1},{0,2},{1,3},{2,4},          // head
        {5,6},{5,7},{7,9},{6,8},{8,10},   // arms
        {5,11},{6,12},{11,12},            // torso
        {11,13},{13,15},{12,14},{14,16}   // legs
    };

    for (const auto& [a, b] : kSkeleton) {
        if (a < (int)pose.keypoints.size() && b < (int)pose.keypoints.size()) {
            const auto& kpA = pose.keypoints[a];
            const auto& kpB = pose.keypoints[b];
            if (kpA.confidence >= confidence_threshold && kpB.confidence >= confidence_threshold) {
                cv::line(image,
                    cv::Point(static_cast<int>(kpA.x), static_cast<int>(kpA.y)),
                    cv::Point(static_cast<int>(kpB.x), static_cast<int>(kpB.y)),
                    cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    for (const auto& kp : pose.keypoints) {
        if (kp.confidence >= confidence_threshold) {
            cv::circle(image,
                cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                4, cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }
}

void App::drawLabel(cv::Mat& image, const std::string& label, float confidence, int x, int y) {
    std::string text = label + ": " + std::to_string(confidence).substr(0, 4);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    y = std::max(y, labelSize.height);
    cv::rectangle(image, cv::Point(x, y - labelSize.height), cv::Point(x + labelSize.width, y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

std::vector<cv::Scalar> App::generateRandomColors(int numColors) {
    std::vector<cv::Scalar> colors;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < numColors; ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
    return colors;
}

bool App::isImageFile(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp";
}

bool App::isVideoFile(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    return ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv";
}

std::vector<std::string> App::split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
