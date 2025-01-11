#include "task_factory.hpp"
#include "utils.hpp"
#include "Triton.hpp"


std::vector<Result> processSource(const std::vector<cv::Mat>& source, 
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient)
{
    const auto input_data = task->preprocess(source);
    auto [infer_results, infer_shapes] = tritonClient->infer(input_data);
    return task->postprocess(cv::Size(source.front().cols, source.front().rows), infer_results, infer_shapes);
}

void ProcessImages(const std::vector<std::string>& sourceNames,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient,
    const std::vector<std::string>& class_names, 
    const std::string& model_name) {
    
    std::vector<cv::Mat> images;
    images.reserve(sourceNames.size());
    
    // Load all images
    for (const auto& sourceName : sourceNames) {
        cv::Mat image = cv::imread(sourceName);
        if (image.empty()) {
            std::cerr << "Could not open or read the image: " << sourceName << std::endl;
            continue;
        }
        images.push_back(image);
    }
    
    if (images.empty()) {
        std::cerr << "No valid images to process" << std::endl;
        return;
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
    std::cout << "Infer time for " << images.size() << " images: " << diff << " ms" << std::endl;

    // Process predictions 
    cv::Mat& image = images[0];
    const std::string& sourceName = sourceNames[0];
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    for (const Result& prediction : predictions)  
    {

        if (std::holds_alternative<Classification>(prediction)) {
            Classification classification = std::get<Classification>(prediction);
            std::cout << "Image " << sourceName << ": " << class_names[classification.class_id] 
                     << ": " << classification.class_confidence << std::endl;
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

    std::string processedFrameFilename = sourceDir + "/processed_frame_" + model_name + ".jpg";
    std::cout << "Saving frame to: " << processedFrameFilename << std::endl;
    cv::imwrite(processedFrameFilename, image);
}
void ProcessVideo(const std::string& sourceName,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient, 
    const std::vector<std::string>& class_names) {
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    cv::VideoCapture cap(sourceName);

    if (!cap.isOpened()) {
        std::cout << "Could not open the video: " << sourceName << std::endl;
        return;
    }

#ifdef WRITE_FRAME
    cv::VideoWriter outputVideo;
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    outputVideo.open(sourceDir + "/processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened()) {
        std::cout << "Could not open the output video for write: " << sourceName << std::endl;
        return;
    }
#endif

    cv::Mat current_frame, previous_frame, visualization_frame;
    std::vector<cv::Scalar> colors = generateRandomColors(class_names.size());
    
    // Read first frame
    if (!cap.read(current_frame)) {
        std::cout << "Failed to read first frame" << std::endl;
        return;
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
        std::cout << "Infer time: " << diff << " ms" << std::endl;

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
                        
                        // Draw label with confidence
                        std::string label = class_names[detection.class_id] + ": " + 
                                          std::to_string(detection.class_confidence).substr(0, 4);
                        draw_label(visualization_frame, label, detection.class_confidence, 
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
                        
                        // Create mask from stored data
                        cv::Mat mask = cv::Mat(segmentation.mask_height, segmentation.mask_width, 
                                             CV_8UC1, segmentation.mask_data.data());
                        
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
                            std::cerr << "ROI and color mask size mismatch. Skipping mask overlay." << std::endl;
                        }
                    } else {
                        std::cerr << "Bounding box is outside the frame. Skipping this instance." << std::endl;
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

static const std::string keys =
    "{ help h   | | Print help message. }"
    "{ model_type mt | yolo11 | yolo version used i.e yolov5 -> yolov10, yolo11, yolonas, yoloseg, dfine, torchvision-classifier}"
    "{ model m | yolo11x_onnx | model name of folder in triton }"
    "{ source s | data/dog.jpg | comma separated list of source images o videos}"
    "{ serverAddress  ip  | localhost | server address ip, default localhost}"
    "{ port  p  | 8001 | Port number(Grpc 8001, Http 8000)}"
    "{ verbose vb | false | Verbose mode, true or false}"
    "{ protocol pt | grpc | Protocol type, grpc or http}"
    "{ labelsFile l | ../coco.names | path to coco labels names}"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes for each model input. Format: CHW;CHW;... (e.g., '3,224,224' for single input or '3,224,224;3,224,224' for two inputs, '3,640,640;2' for rtdetr/dfine models) }";
    
int main(int argc, const char* argv[]) {

    try {
            cv::CommandLineParser parser(argc, argv, keys);
            if (parser.has("help")) 
            {
                parser.printMessage();
                return 0;
            }

            std::cout << "Current path is " << std::filesystem::current_path() << '\n';
            std::string serverAddress = parser.get<std::string>("serverAddress");
            std::string port = parser.get<std::string>("port");
            bool verbose = parser.get<bool>("verbose");

            std::vector<std::string> sourceNames = split(parser.get<std::string>("source"), ',');
            ProtocolType protocol = parser.get<std::string>("protocol") == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
            const size_t batch_size = parser.get<size_t>("batch");

            std::string modelName = parser.get<std::string>("model");
            std::string modelVersion = "";
            std::string modelType = parser.get<std::string>("model_type");


            std::string url(serverAddress + ":" + port);
            std::string labelsFile = parser.get<std::string>("labelsFile");

            std::vector<std::vector<int64_t>> input_sizes;
            if(parser.has("input_sizes")) {
                std::cout << "Parsing input sizes..." << parser.get<std::string>("input_sizes") << std::endl;
                input_sizes = parseInputSizes(parser.get<std::string>("input_sizes"));
                // Output the parsed sizes
                std::cout << "Parsed input sizes:\n";
                for (const auto& size : input_sizes) {
                    std::cout << "(";
                    for (size_t i = 0; i < size.size(); ++i) {
                        std::cout << size[i];
                        if (i < size.size() - 1) {
                            std::cout << ",";
                        }
                    }
                    std::cout << ")\n";
                }               
            }
            else {
                std::cout << "No input sizes provided. Will use default model configuration." << std::endl;
            }

            std::cout << "Chosen Parameters:" << std::endl;
            std::cout << "model_type (mt): " << parser.get<std::string>("model_type") << std::endl;
            std::cout << "model (m): " << parser.get<std::string>("model") << std::endl;
            std::cout << "source (s): " << parser.get<std::string>("source") << std::endl; 
            std::cout << "serverAddress (ip): " << parser.get<std::string>("serverAddress") << std::endl;
            std::cout << "verbose (vb): " << parser.get<bool>("verbose") << std::endl;
            std::cout << "protocol (pt): " << parser.get<std::string>("protocol") << std::endl;
            std::cout << "labelsFile (l): " << parser.get<std::string>("labelsFile") << std::endl;
            std::cout << "batch (b): " << parser.get<size_t>("batch") << std::endl;

    

            // Create Triton client
            std::unique_ptr<Triton> tritonClient = std::make_unique<Triton>(url, protocol, modelName);
            tritonClient->createTritonClient();

            TritonModelInfo modelInfo = tritonClient->getModelInfo(modelName, serverAddress, input_sizes);


            
            // Use the new TaskFactory with input_sizes
            std::unique_ptr<TaskInterface> task = TaskFactory::createTaskInstance(modelType, modelInfo);

            if (!task) {
                throw std::runtime_error("Failed to create task instance");
            }

            const auto class_names = task->readLabelNames(labelsFile);

            std::vector<std::string> image_list;
            std::vector<std::string> video_list;
            for (const auto& sourceName : sourceNames) {
                if (IsImageFile(sourceName)) {
                    image_list.push_back(sourceName);
                } else if (IsVideoFile(sourceName)) {
                    video_list.push_back(sourceName);
                }
                else
                {
                    std::cerr << "Unknown file type: " << sourceName << std::endl;
                }
            }

            if (image_list.empty() && video_list.empty()) {
                throw std::runtime_error("No valid image or video files provided");
            }
            
            if(image_list.size() > 0){
                switch(task->getTaskType())
                {
                    case TaskType::OpticalFlow:
                        {
                            for(size_t i = 0; i < image_list.size() - 1 ; i++) {
                            std::vector<std::string> flowInputs = {image_list[i], image_list[i+1]};
                            ProcessImages(flowInputs, task, tritonClient, class_names, modelName);
                            }
                        }
                        break;        
                    default:
                        {
                            for (const auto& sourceName : image_list) 
                                ProcessImages({sourceName}, task, tritonClient, class_names, modelName);
                        }
                        break;
                } 

            }
            if(video_list.size() > 0){
                for (const auto& sourceName : video_list) 
                    ProcessVideo(sourceName, task, tritonClient, class_names);
            }


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
