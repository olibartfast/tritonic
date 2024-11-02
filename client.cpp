#include "YOLO.hpp"
#include "YoloNas.hpp"
#include "YOLOv10.hpp"
#include "Triton.hpp"
#include "TorchvisionClassifier.hpp"
#include "TensorflowClassifier.hpp"
#include "YOLOSeg.hpp"

std::string ToLower(const std::string& str) {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return lowerStr;
}

// Function to check if the file has an image extension
bool IsImageFile(const std::string& fileName) {
    static const std::set<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"};
    
    // Extract the file extension
    std::string extension = fileName.substr(fileName.find_last_of("."));
    
    // Convert the extension to lowercase and check if it's in the set of image extensions
    return imageExtensions.find(ToLower(extension)) != imageExtensions.end();
}

std::vector<cv::Scalar> generateRandomColors(size_t size) {
    std::vector<cv::Scalar> colors;
    colors.reserve(size); // Reserve space to avoid reallocations

    // Create a random number generator
    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> dist(0, 255); // Range for BGR values

    // Generate random BGR colors
    for (size_t i = 0; i < size; ++i) {
        cv::Scalar color(dist(gen), dist(gen), dist(gen)); // Random B, G, R values
        colors.push_back(color);
    }

    return colors;
}

void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top)
{
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_DUPLEX; // Change font type to what you think is better for you
    const int THICKNESS = 2;
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

    // Display the label and confidence at the top of the bounding box.
    int baseLine;
    std::string confidenceStr = std::to_string(confidence).substr(0, 4);
    std::string display_text = label + ": " + confidenceStr;
    cv::Size label_size = cv::getTextSize(display_text, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);

    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), cv::FILLED);

    // Put the label and confidence on the black rectangle.
    cv::putText(input_image, display_text, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}



std::unique_ptr<TaskInterface> createClassifierInstance(const std::string& modelType, const int input_width, const int input_height, const int channels)
{
    if (modelType == "torchvision-classifier")
    {
        return std::make_unique<TorchvisionClassifier>(input_width, input_height, channels);
    }
    else if (modelType == "tensorflow-classifier")
    {
        return std::make_unique<TensorflowClassifier>(input_width, input_height, channels);
    }
    else
    {
        return nullptr;
    }
}


std::unique_ptr<TaskInterface> createDetectorInstance(const std::string& modelType, const int input_width, const int input_height)
{
    if (modelType.find("yolov10") != std::string::npos )
    {
        return std::make_unique<YOLOv10>(input_width, input_height);
    }    
    else if (modelType.find("yolonas") != std::string::npos )
    {
        return std::make_unique<YoloNas>(input_width, input_height);
    }     
    else if (modelType.find("yolo") != std::string::npos )
    {
        return std::make_unique<YOLO>(input_width, input_height);
    }       
    else
    {
        return nullptr;
    }
}


std::unique_ptr<TaskInterface> createSegmentationInstance(const std::string& modelType, const int input_width, const int input_height)
{
    if (modelType == "yoloseg")
    {
        return std::make_unique<YOLOSeg>(input_width, input_height);
    }
    else
    {
        return nullptr;
    }
}


std::vector<Result> processSource(const cv::Mat& source, 
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient, 
    const TritonModelInfo& modelInfo)
{
 
    std::vector<uint8_t> input_data = task->preprocess(source, modelInfo.input_format_, modelInfo.type1_, modelInfo.type3_,
        modelInfo.input_c_, cv::Size(modelInfo.input_w_, modelInfo.input_h_));

    auto [infer_results, infer_shapes] = tritonClient->infer(input_data);
    return task->postprocess(cv::Size(source.cols, source.rows), infer_results, infer_shapes);
}


// Define a function to perform inference on an image
void ProcessImage(const std::string& sourceName,
    const std::unique_ptr<TaskInterface>& task, 
    const std::unique_ptr<Triton>&  tritonClient,
    const TritonModelInfo& modelInfo, 
    const std::vector<std::string>& class_names) {
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    cv::Mat image = cv::imread(sourceName);

    if (image.empty()) {
        std::cerr << "Could not open or read the image: " << sourceName << std::endl;
        return;
    }

    auto start = std::chrono::steady_clock::now();
    // Call your processSource function here
    std::vector<Result> predictions = processSource(image, task,  tritonClient, modelInfo);
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
    TritonModelInfo& modelInfo,  
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
        std::vector<Result> predictions = processSource(frame, task, tritonClient, modelInfo);
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
    "{ model_type mt | yolo11 | yolo version used i.e yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, yolonas, yoloseg, torchvision-classifier}"
    "{ model m | yolo11x_onnx | model name of folder in triton }"
    "{ task_type tt | | detection, classification, instance_segmentation}"
    "{ source s | data/dog.jpg | path to video or image}"
    "{ serverAddress  ip  | localhost | server address ip, default localhost}"
    "{ port  p  | 8001 | Port number(Grpc 8001, Http 8000)}"
    "{ verbose vb | false | Verbose mode, true or false}"
    "{ protocol pt | grpc | Protocol type, grpc or http}"
    "{ labelsFile l | ../coco.names | path to coco labels names}"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes (channels width height)}";
    
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

        std::istringstream input_sizes_stream(parser.get<std::string>("input_sizes"));
        size_t input_size_c, input_size_w, input_size_h;
        input_sizes_stream >> input_size_c >> input_size_w >> input_size_h;

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

        std::vector<int64_t> input_sizes;
        if (parser.has("input_sizes")) {
            std::cout << "input_size_c: " << input_size_c << std::endl;
            std::cout << "input_size_w: " << input_size_w << std::endl;
            std::cout << "input_size_h: " << input_size_h << std::endl;
            input_sizes.push_back(batch_size);
            input_sizes.push_back(input_size_c);
            input_sizes.push_back(input_size_w);
            input_sizes.push_back(input_size_h);
        }

   
    

        // Create Triton client
        std::unique_ptr<Triton> tritonClient = std::make_unique<Triton>(url, protocol, modelName);
        tritonClient->createTritonClient();

        TritonModelInfo modelInfo = tritonClient->getModelInfo(modelName, serverAddress, {batch_size, input_size_c, input_size_w, input_size_h});

        std::unique_ptr<TaskInterface> task;
        if (taskType == "detection") {
            task = createDetectorInstance(modelType, modelInfo.input_w_, modelInfo.input_h_);
        } else if (taskType == "classification") {
            task = createClassifierInstance(modelType, modelInfo.input_w_, modelInfo.input_h_, modelInfo.input_c_);
        } else if (taskType == "instance_segmentation") {
            task = createSegmentationInstance(modelType, modelInfo.input_w_, modelInfo.input_h_);
        } else {
            throw std::invalid_argument("Invalid task type specified: " + taskType);
        }

        if (!task) {
            throw std::runtime_error("Invalid model type specified: " + modelType);
        }

        const auto class_names = task->readLabelNames(labelsFile);
        std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));

        if (IsImageFile(sourceName)) {
            ProcessImage(sourceName, task, tritonClient, modelInfo, class_names);
        } else {
            ProcessVideo(sourceName, task, tritonClient, modelInfo, class_names);
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
