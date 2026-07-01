#include "App.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include "neuriplo/tasks/core/batch_postprocess.hpp"
#include "neuriplo/tasks/core/batch_preprocess.hpp"
#include "neuriplo/tasks/core/opencv_interop.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"

namespace {
std::string NormalizeModelType(const std::string& modelType) {
    std::string normalized;
    normalized.reserve(modelType.size());
    for (char c : modelType) {
        if (c != '-' && c != '_' && c != ' ') {
            normalized += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return normalized;
}

std::string ReadTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open text input file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::string EscapeJsonString(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += c;
                break;
        }
    }
    return escaped;
}

std::string BuildSamplingParameters(const InferenceConfig& config) {
    std::ostringstream oss;
    oss << "{\"max_tokens\":" << config.GetMaxTokens()
        << ",\"temperature\":" << config.GetTemperature() << ",\"top_p\":" << config.GetTopP();
    if (config.GetRepetitionPenalty() != 1.0f) {
        oss << ",\"repetition_penalty\":" << config.GetRepetitionPenalty();
    }
    if (!config.GetStopWords().empty()) {
        oss << ",\"stop\":[";
        bool first = true;
        std::istringstream stream(config.GetStopWords());
        std::string word;
        while (std::getline(stream, word, ',')) {
            if (!first) {
                oss << ',';
            }
            first = false;
            oss << '"' << EscapeJsonString(word) << '"';
        }
        oss << ']';
    }
    oss << '}';
    return oss.str();
}
}  // namespace

App::App(std::shared_ptr<ITriton> triton, std::shared_ptr<InferenceConfig> config,
         std::shared_ptr<Logger> logger)
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
            logger_->Warn("Server not live, retrying in 2 seconds... (" +
                          std::to_string(retry_count + 1) + "/" + std::to_string(max_retries) +
                          ")");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            retry_count++;
        }

        if (!server_live) {
            throw std::runtime_error("Triton server is not live after " +
                                     std::to_string(max_retries) + " attempts");
        }
        logger_->Info("Triton server is live.");

        // Check if model is in repository
        logger_->Info("Checking if model " + config_->GetModelName() + " is in repository...");
        if (!tritonClient_->isModelInRepository(config_->GetModelName())) {
            throw std::runtime_error("Model " + config_->GetModelName() +
                                     " not found in Triton repository.");
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
        TritonModelInfo modelInfo = tritonClient_->getModelInfo(
            config_->GetModelName(), config_->GetServerAddress(), config_->GetInputSizes());

        // Cache batch metadata for the batched image pipeline (neuriplo-tasks v0.5.0).
        max_batch_size_ = modelInfo.max_batch_size_;
        base_input_shapes_ = modelInfo.input_shapes;

        if (isTextGenerationModelType(config_->GetModelType())) {
            logger_->Info("Text-generation model detected: " + config_->GetModelType());
            processTextGeneration(modelInfo);
            logger_->Info("Application completed successfully");
            return 0;
        }

        // Create task instance
        logger_->Info("Creating task instance for model type: " + config_->GetModelType());
        auto neuriploTasksModelInfo = convertToNeuriploTasksModelInfo(modelInfo);
        neuriplo_tasks::TaskConfig taskConfig;
        taskConfig.confidence_threshold = config_->GetConfidenceThreshold();
        taskConfig.nms_threshold = config_->GetNmsThreshold();
        task_ = neuriplo_tasks::TaskFactory::createTaskInstance(config_->GetModelType(),
                                                                neuriploTasksModelInfo, taskConfig);

        if (!task_) {
            throw std::runtime_error("Failed to create task instance");
        }

        // Extract frame buffer size for video classification from 5D input shape [B, T, C, H, W]
        if (task_->getTaskType() == neuriplo_tasks::TaskType::VideoClassification &&
            !modelInfo.input_shapes.empty()) {
            const auto& shape = modelInfo.input_shapes[0];
            if (shape.size() == 5) {
                num_frames_ = static_cast<int>(shape[1]);
            }
            logger_->Info("Video classification frame buffer size: " + std::to_string(num_frames_));
        }

        // Load class names
        class_names_ = task_->readLabelNames(config_->GetLabelsFile());
        logger_->Info("Loaded " + std::to_string(class_names_.size()) + " class names from " +
                      config_->GetLabelsFile());
        colors_ = generateRandomColors(class_names_.size());

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

        logger_->Info("Processing " + std::to_string(image_list.size()) + " images and " +
                      std::to_string(video_list.size()) + " videos");

        // Process images
        if (!image_list.empty()) {
            processImages(image_list);
        }

        // Process videos
        if (!video_list.empty()) {
            logger_->Info("Processing videos");
            for (const auto& sourceName : video_list) {
                if (task_->getTaskType() == neuriplo_tasks::TaskType::VideoClassification) {
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

neuriplo_tasks::ModelInfo App::convertToNeuriploTasksModelInfo(const TritonModelInfo& triton_info) {
    neuriplo_tasks::ModelInfo model_info;
    model_info.input_shapes = triton_info.input_shapes;
    model_info.input_formats = triton_info.input_formats;
    model_info.input_names = triton_info.input_names;
    model_info.output_names = triton_info.output_names;
    model_info.output_shapes = triton_info.output_shapes;
    model_info.input_types = triton_info.input_types;
    model_info.max_batch_size_ = triton_info.max_batch_size_;
    model_info.batch_size_ = triton_info.batch_size_;
    return model_info;
}

std::vector<neuriplo_tasks::Result> App::processSource(const std::vector<cv::Mat>& source) {
    const auto input_data = task_->preprocess(source);
    auto tensors = tritonClient_->infer(input_data);
    auto vision_tensors = toNeuriploTensors(tensors);
    return task_->postprocess(cv::Size(source.front().cols, source.front().rows), vision_tensors);
}

std::vector<neuriplo_tasks::Tensor> App::toNeuriploTensors(
    const std::vector<Tensor>& tensors) const {
    std::vector<neuriplo_tasks::Tensor> vision_tensors;
    vision_tensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        std::vector<neuriplo_tasks::TensorElement> data;
        data.reserve(tensor.data.size());
        for (const auto& element : tensor.data) {
            if (std::holds_alternative<float>(element)) {
                data.emplace_back(std::get<float>(element));
            } else if (std::holds_alternative<int32_t>(element)) {
                data.emplace_back(std::get<int32_t>(element));
            } else if (std::holds_alternative<int64_t>(element)) {
                data.emplace_back(std::get<int64_t>(element));
            } else if (std::holds_alternative<uint8_t>(element)) {
                data.emplace_back(std::get<uint8_t>(element));
            } else {
                throw std::runtime_error(
                    "String tensors are only supported by the text-generation path.");
            }
        }
        vision_tensors.emplace_back(std::move(data), tensor.shape);
    }
    return vision_tensors;
}

bool App::isBatchableImageTask() const {
    if (!task_ || max_batch_size_ <= 1) {
        return false;
    }
    // Only independent-image families benefit from batched inference (see
    // neuriplo-tasks docs/batch_support_matrix.md). Temporal/multi-input tasks
    // (optical flow, video classification, gaussian splatting) are excluded.
    const auto type = task_->getTaskType();
    return type == neuriplo_tasks::TaskType::Classification ||
           type == neuriplo_tasks::TaskType::Detection ||
           type == neuriplo_tasks::TaskType::InstanceSegmentation ||
           type == neuriplo_tasks::TaskType::PoseEstimation ||
           type == neuriplo_tasks::TaskType::DepthEstimation ||
           type == neuriplo_tasks::TaskType::OpenVocabDetection;
}

std::vector<std::vector<uint8_t>> App::stackBatchBuffers(
    const neuriplo_tasks::BatchPreprocessOutput& pre, int batch_size) const {
    const auto num_inputs = task_->getModelInfo().input_names.size();

    // Pattern A (classification, YOLO, RF-DETR, depth, pose tensor): preprocess emits
    // one buffer per image. Concatenate them into a single Triton input buffer.
    if (static_cast<int>(pre.buffers.size()) == batch_size && num_inputs == 1) {
        std::vector<uint8_t> stacked;
        for (const auto& buf : pre.buffers) {
            stacked.insert(stacked.end(), buf.begin(), buf.end());
        }
        return {std::move(stacked)};
    }

    // Pattern B (RT-DETR, EdgeCrafter, open-vocab): preprocess already concatenated
    // per-image buffers into one buffer per model input node.
    if (pre.buffers.size() == num_inputs) {
        return pre.buffers;
    }

    // Unexpected layout — fall back to concatenating everything per input index.
    return pre.buffers;
}

void App::applyBatchedInputShapes(int batch_size) {
    if (base_input_shapes_.empty()) {
        return;
    }
    std::vector<std::vector<int64_t>> shapes = base_input_shapes_;
    for (auto& shape : shapes) {
        if (!shape.empty()) {
            shape[0] = batch_size;
        }
    }
    tritonClient_->setInputShapes(shapes);
}

std::vector<neuriplo_tasks::Tensor> App::sliceTensorsAxis0(
    const std::vector<neuriplo_tasks::Tensor>& tensors, int index, int batch_size) const {
    std::vector<neuriplo_tasks::Tensor> out;
    out.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        if (batch_size > 1 && !tensor.shape.empty() && tensor.shape[0] == batch_size) {
            const auto total = static_cast<int64_t>(tensor.data.size());
            const int64_t stride = total / batch_size;
            const int64_t begin = static_cast<int64_t>(index) * stride;
            auto sub = std::vector<neuriplo_tasks::TensorElement>(
                tensor.data.begin() + begin, tensor.data.begin() + begin + stride);
            auto shape = tensor.shape;
            shape[0] = 1;
            out.emplace_back(std::move(sub), std::move(shape));
        } else {
            // Tensor not batched on axis 0 — pass through unchanged.
            out.push_back(tensor);
        }
    }
    return out;
}

std::vector<std::vector<neuriplo_tasks::Result>> App::postprocessBatched(
    const std::vector<Tensor>& tensors, const std::vector<cv::Mat>& images, int batch_size) {
    auto vision_tensors = toNeuriploTensors(tensors);
    std::vector<std::vector<neuriplo_tasks::Result>> per_image(images.size());

    // Strict 1:1 families (classification): batchPostprocess returns exactly one result
    // per batch index. Map result[k] onto image[k].
    if (task_->getTaskType() == neuriplo_tasks::TaskType::Classification && !images.empty()) {
        auto post = neuriplo_tasks::batchPostprocess(*task_, images.front().size(), vision_tensors,
                                                     batch_size);
        if (neuriplo_tasks::postprocessResultsMatchBatchSize(post) &&
            static_cast<int>(post.results.size()) == static_cast<int>(images.size())) {
            for (size_t k = 0; k < images.size(); ++k) {
                per_image[k].push_back(post.results[k]);
            }
            return per_image;
        }
        // Mismatch → fall through to per-image slicing.
    }

    // Variable-count / spatial families (detection, segmentation, pose, depth, open-vocab):
    // slice the batched output tensors per image and postprocess each at its own frame size,
    // preserving exact per-image coordinate mapping.
    for (size_t k = 0; k < images.size(); ++k) {
        auto slice = sliceTensorsAxis0(vision_tensors, static_cast<int>(k), batch_size);
        per_image[k] = task_->postprocess(cv::Size(images[k].cols, images[k].rows), slice);
    }
    return per_image;
}

void App::processImagesBatched(const std::vector<std::string>& sourceNames) {
    const int requested = std::max(1, config_->GetBatchSize());
    const int cap = std::max(1, std::min(requested, max_batch_size_));
    logger_->Info("Processing " + std::to_string(sourceNames.size()) +
                  " images in batches of up to " + std::to_string(cap));

    for (size_t start = 0; start < sourceNames.size(); start += static_cast<size_t>(cap)) {
        const size_t end = std::min(start + static_cast<size_t>(cap), sourceNames.size());

        std::vector<cv::Mat> images;
        std::vector<std::string> chunk_names;
        images.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            cv::Mat img = cv::imread(sourceNames[i]);
            if (img.empty()) {
                logger_->Error("Could not open or read the image: " + sourceNames[i]);
                continue;
            }
            images.push_back(img);
            chunk_names.push_back(sourceNames[i]);
        }
        if (images.empty()) {
            continue;
        }
        const int batch_size = static_cast<int>(images.size());

        auto infer_start = std::chrono::steady_clock::now();

        neuriplo_tasks::BatchRequest request{images};
        auto pre = neuriplo_tasks::batchPreprocess(*task_, request);
        auto stacked = stackBatchBuffers(pre, batch_size);
        applyBatchedInputShapes(batch_size);
        auto tensors = tritonClient_->infer(stacked);
        auto per_image = postprocessBatched(tensors, images, pre.batch_size);

        auto infer_end = std::chrono::steady_clock::now();
        const auto diff =
            std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
        logger_->Info("Infer time for " + std::to_string(batch_size) +
                      " images: " + std::to_string(diff) + " ms");

        for (size_t k = 0; k < images.size(); ++k) {
            cv::Mat& image = images[k];
            for (const auto& prediction : per_image[k]) {
                renderPrediction(image, prediction);
            }

            std::string sourceDir = chunk_names[k].substr(0, chunk_names[k].find_last_of("/\\"));
            std::string outputDir = sourceDir + "/output";
            std::filesystem::create_directories(outputDir);
            std::string processedFrameFilename =
                outputDir + "/processed_frame_" + config_->GetModelName() + ".jpg";
            logger_->Info("Saving frame to: " + processedFrameFilename);
            cv::imwrite(processedFrameFilename, image);
        }
    }

    // Restore N=1 shapes so the subsequent video path uses single-image inference.
    applyBatchedInputShapes(1);
}

bool App::isTextGenerationModelType(const std::string& modelType) {
    const std::string normalized = NormalizeModelType(modelType);
    return normalized == "vllm" || normalized == "llm" || normalized == "llama" ||
           normalized == "mistral" || normalized == "qwen" || normalized == "phi" ||
           normalized == "gemma" || normalized == "chatglm" || normalized == "textgeneration";
}

void App::processTextGeneration(const TritonModelInfo& modelInfo) {
    std::string prompt = config_->GetTextPrompt();
    if (prompt.empty() && !config_->GetTextInput().empty()) {
        prompt = ReadTextFile(config_->GetTextInput());
    }
    if (prompt.empty()) {
        prompt = config_->GetSource();
    }
    if (prompt.empty()) {
        throw std::runtime_error(
            "No text input provided. Use --text_prompt, --text_input, or --source.");
    }

    const std::string samplingParameters = BuildSamplingParameters(*config_);
    logger_->Debug("Sampling parameters: " + samplingParameters);

    std::vector<std::vector<std::string>> stringInputs;
    stringInputs.reserve(modelInfo.input_names.size());

    for (size_t i = 0; i < modelInfo.input_names.size(); ++i) {
        const std::string& name = modelInfo.input_names[i];
        if (name == "text_input" || name == "prompt") {
            stringInputs.push_back({prompt});
        } else if (name == "sampling_parameters") {
            stringInputs.push_back({samplingParameters});
        } else if (name == "stream") {
            stringInputs.push_back({"false"});
        } else if (name == "exclude_input_in_output") {
            stringInputs.push_back({"true"});
        } else if (name == "image" && config_->GetEnableMultimodal()) {
            throw std::runtime_error(
                "Multimodal image input is not supported in tritonic yet. "
                "This requires upstream neuriplo-tasks task contracts.");
        } else {
            stringInputs.push_back({""});
        }
    }

    auto start = std::chrono::steady_clock::now();
    const auto results = tritonClient_->inferText(stringInputs);
    auto end = std::chrono::steady_clock::now();
    const auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    logger_->Info("Text generation completed in " + std::to_string(diff) + " ms");

    for (const auto& tensor : results) {
        for (const auto& element : tensor.data) {
            if (std::holds_alternative<std::string>(element)) {
                const std::string& text = std::get<std::string>(element);
                logger_->Info("Generated text: " + text);
                std::cout << text << std::endl;
            }
        }
    }
}

void App::processImages(const std::vector<std::string>& sourceNames) {
    if (task_->getTaskType() == neuriplo_tasks::TaskType::OpticalFlow) {
        logger_->Info("Processing optical flow for image pairs");
        for (size_t i = 0; i < sourceNames.size() - 1; i++) {
            std::vector<std::string> flowInputs = {sourceNames[i], sourceNames[i + 1]};

            std::vector<cv::Mat> images;
            for (const auto& name : flowInputs) {
                cv::Mat img = cv::imread(name);
                if (img.empty()) {
                    logger_->Error("Could not open or read the image: " + name);
                    continue;
                }
                images.push_back(img);
            }

            if (images.size() != 2)
                continue;

            auto start = std::chrono::steady_clock::now();
            std::vector<neuriplo_tasks::Result> predictions = processSource(images);
            auto end = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            logger_->Info("Infer time for " + std::to_string(images.size()) +
                          " images: " + std::to_string(diff) + " ms");

            cv::Mat& image = images[0];
            for (const auto& prediction : predictions) {
                if (std::holds_alternative<neuriplo_tasks::OpticalFlow>(prediction)) {
                    neuriplo_tasks::OpticalFlow flow =
                        std::get<neuriplo_tasks::OpticalFlow>(prediction);
                    neuriplo_tasks::toCvMat(flow.flow).copyTo(image);
                }
            }

            std::string sourceDir = flowInputs[0].substr(0, flowInputs[0].find_last_of("/\\"));
            std::string outputDir = sourceDir + "/output";
            std::filesystem::create_directories(outputDir);
            std::string processedFrameFilename =
                outputDir + "/processed_frame_" + config_->GetModelName() + ".jpg";
            logger_->Info("Saving frame to: " + processedFrameFilename);
            cv::imwrite(processedFrameFilename, image);
        }
    } else if (task_->getTaskType() == neuriplo_tasks::TaskType::VideoClassification) {
        logger_->Info("Processing video classification for image set (" +
                      std::to_string(sourceNames.size()) + " frames)");
        std::vector<cv::Mat> frames;
        for (const auto& name : sourceNames) {
            cv::Mat img = cv::imread(name);
            if (img.empty()) {
                logger_->Warn("Could not open image: " + name);
                continue;
            }
            frames.push_back(img);
        }
        if (frames.empty())
            return;

        auto start = std::chrono::steady_clock::now();
        std::vector<neuriplo_tasks::Result> predictions = processSource(frames);
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger_->Info("Infer time for " + std::to_string(frames.size()) +
                      " frames: " + std::to_string(diff) + " ms");

        for (const auto& prediction : predictions) {
            if (std::holds_alternative<neuriplo_tasks::VideoClassification>(prediction)) {
                const auto& vc = std::get<neuriplo_tasks::VideoClassification>(prediction);
                std::string label =
                    vc.action_label.empty()
                        ? (class_names_.empty() ? "Unknown"
                                                : class_names_[static_cast<int>(vc.class_id)])
                        : vc.action_label;
                logger_->Info("Action: " + label + " (" +
                              std::to_string(vc.class_confidence).substr(0, 4) + ")");
            }
        }
    } else {
        logger_->Info("Processing individual images");
        if (isBatchableImageTask()) {
            processImagesBatched(sourceNames);
            return;
        }
        for (const auto& sourceName : sourceNames) {
            cv::Mat image = cv::imread(sourceName);
            if (image.empty()) {
                logger_->Error("Could not open or read the image: " + sourceName);
                continue;
            }

            auto start = std::chrono::steady_clock::now();
            std::vector<neuriplo_tasks::Result> predictions = processSource({image});
            auto end = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            logger_->Info("Infer time for 1 image: " + std::to_string(diff) + " ms");

            std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
            for (const auto& prediction : predictions) {
                if (std::holds_alternative<neuriplo_tasks::Classification>(prediction)) {
                    const auto& c = std::get<neuriplo_tasks::Classification>(prediction);
                    logger_->Info("Image " + sourceName + ": " +
                                  class_names_[static_cast<int>(c.class_id)] + ": " +
                                  std::to_string(c.class_confidence));
                }
                renderPrediction(image, prediction);
            }

            std::string outputDir = sourceDir + "/output";
            std::filesystem::create_directories(outputDir);
            std::string processedFrameFilename =
                outputDir + "/processed_frame_" + config_->GetModelName() + ".jpg";
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
        cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                              (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
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

    // Read first frame
    if (!cap.read(current_frame)) {
        logger_->Error("Failed to read first frame");
        throw std::runtime_error("Failed to read first frame");
    }

    while (true) {
        auto start = std::chrono::steady_clock::now();
        std::vector<neuriplo_tasks::Result> predictions;

        if (task_->getTaskType() == neuriplo_tasks::TaskType::OpticalFlow) {
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
            if (task_->getTaskType() == neuriplo_tasks::TaskType::OpticalFlow) {
                visualization_frame = cv::Mat::zeros(current_frame.size(), current_frame.type());
                for (const auto& prediction : predictions) {
                    if (std::holds_alternative<neuriplo_tasks::OpticalFlow>(prediction)) {
                        neuriplo_tasks::OpticalFlow flow =
                            std::get<neuriplo_tasks::OpticalFlow>(prediction);
                        neuriplo_tasks::toCvMat(flow.flow).copyTo(visualization_frame);
                    }
                }
            } else {
                current_frame.copyTo(visualization_frame);
                for (const auto& prediction : predictions)
                    renderPrediction(visualization_frame, prediction);
            }

            // Add FPS counter
            double fps = 1000.0 / static_cast<double>(diff);
            std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 4);
            cv::putText(visualization_frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        1, cv::Scalar(0, 255, 0), 2);
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
        cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                              (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
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
            std::vector<neuriplo_tasks::Result> predictions = processSource(frames);
            auto end = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            logger_->Info("Infer time: " + std::to_string(diff) + " ms");

            for (const auto& prediction : predictions) {
                if (std::holds_alternative<neuriplo_tasks::VideoClassification>(prediction)) {
                    const auto& vc = std::get<neuriplo_tasks::VideoClassification>(prediction);
                    last_label =
                        vc.action_label.empty()
                            ? (class_names_.empty() ? "Unknown"
                                                    : class_names_[static_cast<int>(vc.class_id)])
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
            if (cv::waitKey(1) == 27)
                break;
        }
        if (config_->GetWriteFrame() && outputVideo.isOpened()) {
            outputVideo.write(display_frame);
        }
    }
}

void App::renderPrediction(cv::Mat& frame, const neuriplo_tasks::Result& prediction) {
    if (std::holds_alternative<neuriplo_tasks::Classification>(prediction)) {
        const auto& c = std::get<neuriplo_tasks::Classification>(prediction);
        drawLabel(frame, class_names_[static_cast<int>(c.class_id)], c.class_confidence, 30, 30);
    } else if (std::holds_alternative<neuriplo_tasks::Detection>(prediction)) {
        const auto& det = std::get<neuriplo_tasks::Detection>(prediction);
        cv::Rect safeBbox =
            neuriplo_tasks::toCvRect(det.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeBbox.width > 0 && safeBbox.height > 0) {
            cv::rectangle(frame, safeBbox, colors_[static_cast<int>(det.class_id)], 2);
            drawLabel(frame, class_names_[static_cast<int>(det.class_id)], det.class_confidence,
                      safeBbox.x, safeBbox.y - 1);
        }
    } else if (std::holds_alternative<neuriplo_tasks::InstanceSegmentation>(prediction)) {
        const auto& seg = std::get<neuriplo_tasks::InstanceSegmentation>(prediction);
        cv::Rect safeBbox =
            neuriplo_tasks::toCvRect(seg.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeBbox.width > 0 && safeBbox.height > 0) {
            cv::rectangle(frame, safeBbox, colors_[static_cast<int>(seg.class_id)], 2);
            drawLabel(frame, class_names_[static_cast<int>(seg.class_id)], seg.class_confidence,
                      safeBbox.x, safeBbox.y - 1);

            cv::Mat mask;
            if (!seg.mask.empty()) {
                mask = neuriplo_tasks::toCvMat(seg.mask);
            } else if (!seg.mask_data.empty()) {
                mask = cv::Mat(seg.mask_height, seg.mask_width, CV_8UC1,
                               const_cast<uint8_t*>(seg.mask_data.data()));
            }
            if (!mask.empty()) {
                cv::Mat resized_mask;
                cv::resize(mask, resized_mask, safeBbox.size(), 0, 0, cv::INTER_NEAREST);
                cv::Mat colorMask = cv::Mat::zeros(safeBbox.size(), CV_8UC3);
                colorMask.setTo(colors_[static_cast<int>(seg.class_id)], resized_mask);
                cv::Mat roi = frame(safeBbox);
                if (roi.size() == colorMask.size())
                    cv::addWeighted(roi, 1, colorMask, 0.5, 0, roi);
            }
        }
    } else if (std::holds_alternative<neuriplo_tasks::PoseEstimation>(prediction)) {
        drawPose(frame, std::get<neuriplo_tasks::PoseEstimation>(prediction));
    } else if (std::holds_alternative<neuriplo_tasks::DepthEstimation>(prediction)) {
        const auto& depth = std::get<neuriplo_tasks::DepthEstimation>(prediction);
        if (!depth.normalized_depth.empty()) {
            cv::Mat depth_8u;
            neuriplo_tasks::toCvMat(depth.normalized_depth).convertTo(depth_8u, CV_8UC1, 255.0);
            cv::applyColorMap(depth_8u, frame, cv::COLORMAP_INFERNO);
        }
    } else if (std::holds_alternative<neuriplo_tasks::OpenVocabDetection>(prediction)) {
        const auto& det = std::get<neuriplo_tasks::OpenVocabDetection>(prediction);
        cv::Rect safeBbox =
            neuriplo_tasks::toCvRect(det.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeBbox.width > 0 && safeBbox.height > 0) {
            cv::rectangle(frame, safeBbox, cv::Scalar(0, 255, 0), 2);
            drawLabel(frame, det.label, det.score, safeBbox.x, safeBbox.y - 1);
        }
    }
}

void App::drawPose(cv::Mat& image, const neuriplo_tasks::PoseEstimation& pose,
                   float confidence_threshold) {
    // COCO 17-keypoint skeleton connections
    static const std::vector<std::pair<int, int>> kSkeleton = {
        {0, 1},   {0, 2},   {1, 3},   {2, 4},            // head
        {5, 6},   {5, 7},   {7, 9},   {6, 8},  {8, 10},  // arms
        {5, 11},  {6, 12},  {11, 12},                    // torso
        {11, 13}, {13, 15}, {12, 14}, {14, 16}           // legs
    };

    for (const auto& [a, b] : kSkeleton) {
        if (a < (int)pose.keypoints.size() && b < (int)pose.keypoints.size()) {
            const auto& kpA = pose.keypoints[a];
            const auto& kpB = pose.keypoints[b];
            if (kpA.confidence >= confidence_threshold && kpB.confidence >= confidence_threshold) {
                cv::line(image, cv::Point(static_cast<int>(kpA.x), static_cast<int>(kpA.y)),
                         cv::Point(static_cast<int>(kpB.x), static_cast<int>(kpB.y)),
                         cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    for (const auto& kp : pose.keypoints) {
        if (kp.confidence >= confidence_threshold) {
            cv::circle(image, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 4,
                       cv::Scalar(0, 0, 255), cv::FILLED);
        }
    }
}

void App::drawLabel(cv::Mat& image, const std::string& label, float confidence, int x, int y) {
    std::string text = label + ": " + std::to_string(confidence).substr(0, 4);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    y = std::max(y, labelSize.height);
    cv::rectangle(image, cv::Point(x, y - labelSize.height),
                  cv::Point(x + labelSize.width, y + baseLine), cv::Scalar(255, 255, 255),
                  cv::FILLED);
    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0),
                1);
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
