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


// Define a function to perform inference on an image
void ProcessImage(const std::string& sourceName,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient,
    const std::vector<std::string>& class_names) {
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    cv::Mat image = cv::imread(sourceName);
    if (image.empty()) {
        std::cerr << "Could not open or read the image: " << sourceName << std::endl;
        return;
    }    
    std::vector<cv::Mat> images = {image};


    auto start = std::chrono::steady_clock::now();
    // Call your processSource function here
    std::vector<Result> predictions = processSource(images, task,  tritonClient);
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Infer time: " << diff << " ms" << std::endl;

    for (const Result& prediction : predictions) 
    {
        if (std::holds_alternative<Classification>(prediction)) {
            Classification classification = std::get<Classification>(prediction);
            std::cout << class_names[classification.class_id] << ": " << classification.class_confidence << std::endl; 
            draw_label(image, class_names[classification.class_id] , classification.class_confidence, 30, 30); 
        } 
        else if (std::holds_alternative<Detection>(prediction)) 
        {
            Detection detection = std::get<Detection>(prediction);
            cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 2);
            draw_label(image,  class_names[detection.class_id], detection.class_confidence, detection.bbox.x, detection.bbox.y - 1);
        }

        else if (std::holds_alternative<InstanceSegmentation>(prediction)) 
        {
            InstanceSegmentation segmentation = std::get<InstanceSegmentation>(prediction);
            
            // Draw bounding box
            cv::rectangle(image, segmentation.bbox, cv::Scalar(255, 0, 0), 2);
            
            // Draw label
            draw_label(image, class_names[segmentation.class_id], segmentation.class_confidence, segmentation.bbox.x, segmentation.bbox.y - 1);
            
            // Create mask from stored data
            cv::Mat mask = cv::Mat(segmentation.mask_height, segmentation.mask_width, CV_8UC1, segmentation.mask_data.data());
            
            // Draw mask
            cv::Mat colorMask = cv::Mat::zeros(mask.size(), CV_8UC3);
            cv::Scalar color = cv::Scalar(rand() & 255, rand() & 255, rand() & 255);
            colorMask.setTo(color, mask);
            
            cv::Mat roi = image(segmentation.bbox);
            cv::addWeighted(roi, 1, colorMask, 0.7, 0, roi);
        }
    }    

    std::string processedFrameFilename = sourceDir + "/processed_frame.jpg";
    std::cout << "Saving frame to: " << processedFrameFilename << std::endl;
    cv::imwrite(processedFrameFilename, image);
}

// Define a function to perform inference on a video
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

    cv::Mat frame;
    std::vector<cv::Scalar> colors = generateRandomColors(class_names.size()); 
    while (cap.read(frame)) {
        auto start = std::chrono::steady_clock::now();
        // Call your processSource function here
        std::vector<Result> predictions = processSource(frame, task, tritonClient);
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Infer time: " << diff << " ms" << std::endl;


#if defined(SHOW_FRAME) || defined(WRITE_FRAME)
        double fps = 1000.0 / static_cast<double>(diff);
        std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (const auto& prediction : predictions) 
        {
            if (std::holds_alternative<InstanceSegmentation>(prediction)) 
            {
                InstanceSegmentation segmentation = std::get<InstanceSegmentation>(prediction);
                
                // Ensure the bounding box is within the frame boundaries
                cv::Rect safeBbox = segmentation.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
                
                if (safeBbox.width > 0 && safeBbox.height > 0) {
                    // Draw bounding box
                    cv::rectangle(frame, safeBbox, cv::Scalar(255, 0, 0), 2);
                    
                    // Draw label
                    draw_label(frame, class_names[segmentation.class_id], segmentation.class_confidence, safeBbox.x, safeBbox.y - 1);
                    
                    // Create mask from stored data
                    cv::Mat mask = cv::Mat(segmentation.mask_height, segmentation.mask_width, CV_8UC1, segmentation.mask_data.data());
                    
                    // Resize mask to match the safe bounding box size
                    cv::resize(mask, mask, safeBbox.size(), 0, 0, cv::INTER_NEAREST);
                    
                    // Draw mask
                    cv::Mat colorMask = cv::Mat::zeros(safeBbox.size(), CV_8UC3);
                    cv::Scalar color = colors[segmentation.class_id];
                    colorMask.setTo(color, mask);
                    
                    // Get the ROI from the frame
                    cv::Mat roi = frame(safeBbox);
                    
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
#endif

#ifdef SHOW_FRAME
        cv::imshow("video feed", frame);
        cv::waitKey(1);
#endif

#ifdef WRITE_FRAME
        outputVideo.write(frame);
#endif
    }
}


static const std::string keys =
    "{ help h   | | Print help message. }"
    "{ model_type mt | yolo11 | yolo version used i.e yolov5 -> yolov10, yolo11, yolonas, yoloseg, dfine, torchvision-classifier}"
    "{ model m | yolo11x_onnx | model name of folder in triton }"
    "{ task_type tt | | detection, classification, instance_segmentation}"
    "{ source s | data/dog.jpg | path to video or image}"
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
        if (parser.has("help")) {
            parser.printMessage();
            return 0;
        }

        std::cout << "Current path is " << std::filesystem::current_path() << '\n';
        std::string serverAddress = parser.get<std::string>("serverAddress");
        std::string port = parser.get<std::string>("port");
        bool verbose = parser.get<bool>("verbose");
        std::string sourceName = parser.get<std::string>("source");
        ProtocolType protocol = parser.get<std::string>("protocol") == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
        const size_t batch_size = parser.get<size_t>("batch");

        std::string modelName = parser.get<std::string>("model");
        std::string modelVersion = "";
        std::string modelType = parser.get<std::string>("model_type");

        if (!parser.has("task_type")) {
            throw std::runtime_error("Task type (classification or detection) is required");
        }

        std::string taskType = parser.get<std::string>("task_type");
        std::string url(serverAddress + ":" + port);
        std::string labelsFile = parser.get<std::string>("labelsFile");

        if (!std::filesystem::exists(labelsFile)) {
            throw std::runtime_error("Labels file " + labelsFile + " does not exist");
        }

        if (!std::filesystem::exists(sourceName)) {
            throw std::runtime_error("Source file " + sourceName + " does not exist");
        }

        std::vector<std::vector<int64_t>> input_sizes;
        if (parser.has("input_sizes")) {
            std::string sizes_str = parser.get<std::string>("input_sizes");
            std::istringstream sizes_stream(sizes_str);
            std::string size;
            while (std::getline(sizes_stream, size, ';')) {
                std::vector<int64_t> dims;
                std::istringstream dim_stream(size);
                std::string dim;
                while (std::getline(dim_stream, dim, ',')) {
                    dims.push_back(std::stoll(dim));
                }
                input_sizes.push_back(dims);
            }

            // Print parsed input information
            std::cout << "Parsed input sizes:" << std::endl;
            for (size_t i = 0; i < input_sizes.size(); ++i) {
                std::cout << "Input " << i << " - Shape: ";
                for (const auto& dim : input_sizes[i]) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "No input sizes provided. Will use default model configuration." << std::endl;
        }

        std::cout << "Chosen Parameters:" << std::endl;
        std::cout << "task_type (tt): " << parser.get<std::string>("task_type") << std::endl;
        std::cout << "model_type (mt): " << parser.get<std::string>("model_type") << std::endl;
        std::cout << "model (m): " << parser.get<std::string>("model") << std::endl;
        std::cout << "source (s): " << parser.get<std::string>("source") << std::endl;  // Changed from 'video' to 'source'
        std::cout << "serverAddress (ip): " << parser.get<std::string>("serverAddress") << std::endl;
        std::cout << "verbose (vb): " << parser.get<bool>("verbose") << std::endl;
        std::cout << "protocol (pt): " << parser.get<std::string>("protocol") << std::endl;
        std::cout << "labelsFile (l): " << parser.get<std::string>("labelsFile") << std::endl;
        std::cout << "batch (b): " << parser.get<size_t>("batch") << std::endl;

        if(!std::filesystem::exists(labelsFile))
        {
            std::cerr << "Labels file " << labelsFile << " does not exist" << std::endl;
            std::exit(1);
        }

        if(!std::filesystem::exists(sourceName))
        {
            std::cerr << "Source file " << sourceName << " does not exist" << std::endl;
            std::exit(1);
        }

   

        // Create Triton client
        std::unique_ptr<Triton> tritonClient = std::make_unique<Triton>(url, protocol, modelName);
        tritonClient->createTritonClient();

        TritonModelInfo modelInfo = tritonClient->getModelInfo(modelName, serverAddress, input_sizes);


        
        // Use the new TaskFactory with input_sizes
        std::unique_ptr<TaskInterface> task = TaskFactory::createTaskInstance(modelType, modelInfo);

        if (!task) {
            throw std::runtime_error("Failed to create task instance");
        }

        // Validate that the created task matches the specified task type
        if ((taskType == "detection" && !dynamic_cast<Detection*>(task.get()) && !dynamic_cast<DFine*>(task.get())) ||
            (taskType == "classification" && !dynamic_cast<Classification*>(task.get())) ||
            (taskType == "instance_segmentation" && !dynamic_cast<InstanceSegmentation*>(task.get()))) {
            throw std::runtime_error("Created task does not match specified task type: " + taskType);
        }

        const auto class_names = task->readLabelNames(labelsFile);
        std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));

        if (IsImageFile(sourceName)) {
            ProcessImage(sourceName, task, tritonClient, class_names);
        } else {
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
