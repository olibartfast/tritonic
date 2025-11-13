#include "task_factory.hpp"
#include <filesystem>
#include "utils.hpp"
#include "ITriton.hpp"
#include "Triton.hpp"
#include "Config.hpp"
#include "Logger.hpp"


std::vector<Result> processSource(const std::vector<cv::Mat>& source, 
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<ITriton>&  tritonClient)
{
    const auto input_data = task->preprocess(source);
    auto [infer_results, infer_shapes] = tritonClient->infer(input_data);
    return task->postprocess(cv::Size(source.front().cols, source.front().rows), infer_results, infer_shapes);
}

void ProcessImages(const std::vector<std::string>& sourceNames,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<ITriton>&  tritonClient,
    const std::vector<std::string>& class_names, 
    const std::string& model_name) {
    
    std::vector<cv::Mat> images;
    images.reserve(sourceNames.size());
    
    // Load all images
    for (const auto& sourceName : sourceNames) {
        cv::Mat image = cv::imread(sourceName);
        if (image.empty()) {
            logger.errorf("Could not open or read the image: {}", sourceName);
            continue;
        }
        images.push_back(image);
    }
    
    if (images.empty()) {
        logger.error("No valid images to process");
        throw std::runtime_error("No valid images to process");
    }

    if (task->getTaskType() == TaskType::OpticalFlow) {
        if (images.size() != 2) {
            throw std::runtime_error("Optical flow task requires exactly 2 images");
        }
        } else {
        if (images.size() != 1) {
            throw std::runtime_error("Non-optical flow task requires exactly 1 image");
        }
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<Result> predictions = processSource(images, task, tritonClient);
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    logger.infof("Infer time for {} images: {} ms", images.size(), diff);

    // Process predictions 
    cv::Mat& image = images[0];
    const std::string& sourceName = sourceNames[0];
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    for (const Result& prediction : predictions)  
    {

        if (std::holds_alternative<Classification>(prediction)) {
            Classification classification = std::get<Classification>(prediction);
            logger.infof("Image {}: {}: {}", sourceName, class_names[classification.class_id], classification.class_confidence);
            draw_label(image, class_names[classification.class_id], classification.class_confidence, 30, 30);
        } 
        else if (std::holds_alternative<Detection>(prediction)) {
            Detection detection = std::get<Detection>(prediction);
            cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 2);
            draw_label(image, class_names[detection.class_id], detection.class_confidence, 
                      detection.bbox.x, detection.bbox.y - 1);
        }
        else if (std::holds_alternative<InstanceSegmentation>(prediction)) {
            InstanceSegmentation segmentation = std::get<InstanceSegmentation>(prediction);
            
            cv::rectangle(image, segmentation.bbox, cv::Scalar(255, 0, 0), 2);
            draw_label(image, class_names[segmentation.class_id], segmentation.class_confidence, 
                      segmentation.bbox.x, segmentation.bbox.y - 1);
            
            cv::Mat mask = cv::Mat(segmentation.mask_height, segmentation.mask_width, 
                                 CV_8UC1, segmentation.mask_data.data());
            
            cv::Mat colorMask = cv::Mat::zeros(mask.size(), CV_8UC3);
            cv::Scalar color = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
            colorMask.setTo(color, mask);
            
            cv::Mat roi = image(segmentation.bbox);
            cv::addWeighted(roi, 1, colorMask, 0.7, 0, roi);
        }
        else if (std::holds_alternative<OpticalFlow>(prediction))
        {
            OpticalFlow flow = std::get<OpticalFlow>(predictions[0]);
            flow.flow.copyTo(image);
        }
    }

    std::string outputDir = sourceDir + "/output";
    std::filesystem::create_directories(outputDir);
    std::string processedFrameFilename = outputDir + "/processed_frame_" + model_name + ".jpg";
    logger.infof("Saving frame to: {}", processedFrameFilename);
    bool writeSuccess = cv::imwrite(processedFrameFilename, image);
    if (writeSuccess) {
        logger.infof("Successfully saved frame to: {}", processedFrameFilename);
    } else {
        logger.errorf("Failed to save frame to: {}", processedFrameFilename);
    }
}
void ProcessVideo(const std::string& sourceName,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<ITriton>&  tritonClient, 
    const std::vector<std::string>& class_names) {
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    cv::VideoCapture cap(sourceName);

    if (!cap.isOpened()) {
        logger.errorf("Could not open the video: {}", sourceName);
        throw std::runtime_error("Could not open the video: " + sourceName);
    }

#ifdef WRITE_FRAME
    cv::VideoWriter outputVideo;
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    std::string outputDir = sourceDir + "/output";
    std::filesystem::create_directories(outputDir);
    outputVideo.open(outputDir + "/processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened()) {
        logger.errorf("Could not open the output video for write: {}", sourceName);
        return;
    }
#endif

    cv::Mat current_frame, previous_frame, visualization_frame;
    std::vector<cv::Scalar> colors = generateRandomColors(class_names.size());
    
    // Read first frame
    if (!cap.read(current_frame)) {
        logger.error("Failed to read first frame");
        throw std::runtime_error("Failed to read first frame");
    }

    while (true) {
        auto start = std::chrono::steady_clock::now();
        std::vector<Result> predictions;

        if (task->getTaskType() == TaskType::OpticalFlow) {
            if (!previous_frame.empty()) {
                // Process optical flow between previous and current frame
                std::vector<cv::Mat> frame_pair = {previous_frame, current_frame};
                predictions = processSource(frame_pair, task, tritonClient);
            }
        } else {
            // Process single frame for other tasks
            predictions = processSource({current_frame}, task, tritonClient);
        }

        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger.infof("Infer time: {} ms", diff);

#if defined(SHOW_FRAME) || defined(WRITE_FRAME)
        // Create visualization frame
        if (task->getTaskType() == TaskType::OpticalFlow) {
            visualization_frame = cv::Mat::zeros(current_frame.size(), current_frame.type());
            for (const auto& prediction : predictions) {
                if (std::holds_alternative<OpticalFlow>(prediction)) {
                    OpticalFlow flow = std::get<OpticalFlow>(prediction);
                    flow.flow.copyTo(visualization_frame);
                }
            }
        } else {
            current_frame.copyTo(visualization_frame);
            for (const auto& prediction : predictions) {
                if (std::holds_alternative<Detection>(prediction)) {
                    Detection detection = std::get<Detection>(prediction);
                    // Ensure bounding box is within frame boundaries
                    cv::Rect safeBbox = detection.bbox & cv::Rect(0, 0, visualization_frame.cols, visualization_frame.rows);
                    
                    if (safeBbox.width > 0 && safeBbox.height > 0) {
                        // Draw bounding box with class-specific color
                        cv::rectangle(visualization_frame, safeBbox, colors[detection.class_id], 2);
                        
                    
                        draw_label(visualization_frame, class_names[detection.class_id], detection.class_confidence, 
                                 safeBbox.x, safeBbox.y - 1);
                    }
                }
                else if (std::holds_alternative<InstanceSegmentation>(prediction)) {
                    InstanceSegmentation segmentation = std::get<InstanceSegmentation>(prediction);
                    
                    // Ensure the bounding box is within the frame boundaries
                    cv::Rect safeBbox = segmentation.bbox & cv::Rect(0, 0, visualization_frame.cols, visualization_frame.rows);
                    
                    if (safeBbox.width > 0 && safeBbox.height > 0) {
                        // Draw bounding box
                        cv::rectangle(visualization_frame, safeBbox, colors[segmentation.class_id], 2);
                        
                        // Draw label
                        draw_label(visualization_frame, class_names[segmentation.class_id], segmentation.class_confidence, 
                                 safeBbox.x, safeBbox.y - 1);
                        
                        // Check if mask data is valid before processing
                        if (!segmentation.mask_data.empty() && segmentation.mask_height > 0 && segmentation.mask_width > 0) {
                            // Create mask from stored data - need to clone to ensure data ownership
                            cv::Mat mask(segmentation.mask_height, segmentation.mask_width, CV_8UC1);
                            std::memcpy(mask.data, segmentation.mask_data.data(), segmentation.mask_data.size());
                            
                            // Resize mask to match the safe bounding box size
                            cv::resize(mask, mask, safeBbox.size(), 0, 0, cv::INTER_NEAREST);
                            
                            // Draw mask
                            cv::Mat colorMask = cv::Mat::zeros(safeBbox.size(), CV_8UC3);
                            cv::Scalar color = colors[segmentation.class_id];
                            colorMask.setTo(color, mask);
                            
                            // Get the ROI from the frame
                            cv::Mat roi = visualization_frame(safeBbox);
                            
                            // Ensure colorMask and roi have the same size
                            if (roi.size() == colorMask.size()) {
                                cv::addWeighted(roi, 1, colorMask, 0.5, 0, roi);
                            } else {
                                logger.warn("ROI and color mask size mismatch. Skipping mask overlay.");
                            }
                        } else {
                            logger.warn("Instance segmentation mask data is empty or invalid. Showing bbox only.");
                        }
                    } else {
                        logger.warn("Bounding box is outside the frame. Skipping this instance.");
                    }
                }
            }
        }

        // Add FPS counter to visualization frame
        double fps = 1000.0 / static_cast<double>(diff);
        std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(visualization_frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
#endif

#ifdef SHOW_FRAME
        cv::imshow("video feed", visualization_frame);
        cv::waitKey(1);
#endif

#ifdef WRITE_FRAME
        outputVideo.write(visualization_frame);
#endif

        // Store current frame as previous frame
        current_frame.copyTo(previous_frame);
        
        // Read next frame
        if (!cap.read(current_frame)) {
            break;
        }
    }
}

int main(int argc, const char* argv[]) {
    try {
        // Initialize logging
        logger.setLogLevel(LogLevel::INFO);
        logger.setConsoleOutput(true);
        
        logger.info("Starting Triton Client Application");
        
        // Load configuration
        std::unique_ptr<Config> config;
        
        try {
            // Try to load from command line first
            config = ConfigManager::loadFromCommandLine(argc, argv);
        } catch (const std::invalid_argument& e) {
            logger.error("Command line configuration error: " + std::string(e.what()));
            return 1; // Exit immediately on validation error
        }
        
        // If command line config failed (e.g., help was requested), exit gracefully
        if (!config) {
            return 0; // Exit successfully when help was requested
        }
        
        // If we reach here, we have a valid config from command line
        // No need to try environment variables
        
        // Set up logging based on configuration
        if (!config->log_file.empty()) {
            logger.setLogFile(config->log_file);
        }
        
        if (config->log_level == "debug") logger.setLogLevel(LogLevel::DEBUG);
        else if (config->log_level == "warn") logger.setLogLevel(LogLevel::WARN);
        else if (config->log_level == "error") logger.setLogLevel(LogLevel::ERROR);
        else logger.setLogLevel(LogLevel::INFO);
        
        // Print configuration
        ConfigManager::printConfig(*config);
        
        logger.info("Current path is " + std::string(std::filesystem::current_path()));
        
        // Parse source files
        std::vector<std::string> sourceNames = split(config->source, ',');
        
        // Determine protocol
        ProtocolType protocol = config->protocol == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
        
        // Build URL
        std::string url = config->server_address + ":" + std::to_string(config->port);
        
        logger.infof("Connecting to Triton server at {} using {} protocol", url, config->protocol);
        
        // Create Triton client
        std::unique_ptr<ITriton> tritonClient = std::make_unique<Triton>(url, protocol, config->model_name, config->model_version, config->verbose);
        tritonClient->createTritonClient();

        logger.infof("Getting model info for: {}", config->model_name);
        TritonModelInfo modelInfo = tritonClient->getModelInfo(config->model_name, config->server_address, config->input_sizes);

        // Create task instance
        logger.infof("Creating task instance for model type: {}", config->model_type);
        std::unique_ptr<TaskInterface> task = TaskFactory::createTaskInstance(config->model_type, modelInfo);

        if (!task) {
            throw std::runtime_error("Failed to create task instance");
        }

        // Load class names
        const auto class_names = task->readLabelNames(config->labels_file);
        logger.infof("Loaded {} class names from {}", class_names.size(), config->labels_file);

        // Categorize source files
        std::vector<std::string> image_list;
        std::vector<std::string> video_list;
        for (const auto& sourceName : sourceNames) {
            if (IsImageFile(sourceName)) {
                image_list.push_back(sourceName);
                logger.debug("Added image file: " + sourceName);
            } else if (IsVideoFile(sourceName)) {
                video_list.push_back(sourceName);
                logger.debug("Added video file: " + sourceName);
            } else {
                logger.warn("Unknown file type: " + sourceName);
            }
        }

        if (image_list.empty() && video_list.empty()) {
            throw std::runtime_error("No valid image or video files provided");
        }
        
        logger.infof("Processing {} images and {} videos", image_list.size(), video_list.size());
        
        // Process images
        if (!image_list.empty()) {
            switch(task->getTaskType()) {
                case TaskType::OpticalFlow:
                    logger.info("Processing optical flow for image pairs");
                    for(size_t i = 0; i < image_list.size() - 1; i++) {
                        std::vector<std::string> flowInputs = {image_list[i], image_list[i+1]};
                        ProcessImages(flowInputs, task, tritonClient, class_names, config->model_name);
                    }
                    break;        
                default:
                    logger.info("Processing individual images");
                    for (const auto& sourceName : image_list) {
                        ProcessImages({sourceName}, task, tritonClient, class_names, config->model_name);
                    }
                    break;
            } 
        }
        
        // Process videos
        if (!video_list.empty()) {
            logger.info("Processing videos");
            for (const auto& sourceName : video_list) {
                ProcessVideo(sourceName, task, tritonClient, class_names);
            }
        }

        logger.info("Application completed successfully");

    } catch (const std::exception& e) {
        logger.error("Application error: " + std::string(e.what()));
        return 1;
    } catch (...) {
        logger.fatal("An unknown error occurred");
        return 1;
    }

    return 0;
}
