#include "Yolo.hpp"
#include "YoloNas.hpp"
#include "YOLOv7_NMS.hpp"
#include "Triton.hpp"
#include "TorchvisionClassifier.hpp"

std::unique_ptr<TaskInterface> createClassifierInstance(const std::string& modelType, const int input_width, const int input_height, const int channels)
{
    if (modelType == "torchvision-classifier")
    {
        return std::make_unique<TorchvisionClassifier>(input_width, input_height, channels);
    }
    else
    {
        return nullptr;
    }
}


std::unique_ptr<TaskInterface> createDetectorInstance(const std::string& modelType, const int input_width, const int input_height)
{
    if (modelType == "yolov7nms")
    {
        //return std::make_unique<YOLOv7_NMS>(input_width, input_height);
        std::cout << "Work in progress..." << std::endl;
        return nullptr;
    }
    else if (modelType.find("yolov") != std::string::npos )
    {
        return std::make_unique<Yolo>(input_width, input_height);
    }
    else if (modelType.find("yolonas") != std::string::npos )
    {
        return std::make_unique<YoloNas>(input_width, input_height);
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

        } 
        else if (std::holds_alternative<Detection>(prediction)) 
        {
            Detection detection = std::get<Detection>(prediction);
            cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 2);
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
    while (cap.read(frame)) {
        auto start = std::chrono::steady_clock::now();
        // Call your processSource function here
        std::vector<Result> predictions = processSource(frame, task, tritonClient, modelInfo);
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Infer time: " << diff << " ms" << std::endl;

#if defined(SHOW_FRAME) || defined(WRITE_FRAME)
        double fps = 1000.0 / static_cast<double>(diff);
        std::string fpsText = "FPS: " + std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (auto&& prediction : predictions) 
        {
            if (std::holds_alternative<Detection>(prediction)) 
            {
                Detection detection = std::get<Detection>(prediction);
                cv::rectangle(frame, detection.bbox, cv::Scalar(255, 0, 0), 2);
                cv::rectangle(frame, detection.bbox, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, class_names[detection.class_id],
                cv::Point(detection.bbox.x, detection.bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
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
    "{ model_type mt | yolov7 | yolo version used i.e yolov5, yolov6, yolov7, yolov8}"
    "{ model m | yolov7-tiny_onnx | model name of folder in triton }"
    "{ task_type tt | | detection, classification}"
    "{ source s | data/dog.jpg | path to video or image}"
    "{ serverAddress  ip  | localhost | server address ip, default localhost}"
    "{ port  p  | 8001 | Port number(Grpc 8001, Http 8000)}"
    "{ verbose vb | false | Verbose mode, true or false}"
    "{ protocol pt | grpc | Protocol type, grpc or http}"
    "{ labelsFile l | ../coco.names | path to coco labels names}"
    "{ batch b | 1 | Batch size}"
    "{ input_sizes is | | Input sizes (channels width height)}";

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    std::string serverAddress = parser.get<std::string>("serverAddress");
    std::string port = parser.get<std::string>("port");
    bool verbose = parser.get<bool>("verbose");
    std::string sourceName = parser.get<std::string>("source");  // Changed from 'video' to 'source'
    ProtocolType protocol = parser.get<std::string>("protocol") == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
    const size_t batch_size = parser.get<size_t>("batch");

    std::string modelName = parser.get<std::string>("model");
    std::string modelVersion = "";
    std::string modelType = parser.get<std::string>("model_type");

    if(!parser.has("task_type"))
    {
        std::cerr << "Task type (classification or detection) is required " << std::endl;
        std::exit(1);
    }

    std::string taskType = parser.get<std::string>("task_type");
    std::string url(serverAddress + ":" + port);
    std::string labelsFile = parser.get<std::string>("labelsFile");

    std::string input_sizes_str = parser.get<std::string>("input_sizes");
    std::istringstream input_sizes_stream(input_sizes_str);
    
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

    TritonModelInfo modelInfo = tritonClient->getModelInfo(modelName, serverAddress, input_sizes);
    std::unique_ptr<TaskInterface> task;
    if (taskType == "detection") {
        task = createDetectorInstance(modelType, modelInfo.input_w_, modelInfo.input_h_);
        if(task == nullptr)
        {
            std::cerr << "Invalid model type specified: " +  modelType << std::endl;
        }
    } 
    else if (taskType == "classification") {
 
        task = createClassifierInstance(modelType, modelInfo.input_w_, modelInfo.input_h_, modelInfo.input_c_);
        if(task == nullptr)
        {
            std::cerr << "Invalid model type specified: " +  modelType << std::endl;
        }
    } 
    else {
        std::cerr << "Invalid task type specified: " +  taskType << std::endl;
        return 1;
    }

    const auto class_names = task->readLabelNames(labelsFile);



    // Get the directory of the source file
    std::string sourceDir = sourceName.substr(0, sourceName.find_last_of("/\\"));
    if (sourceName.find(".jpg") != std::string::npos || sourceName.find(".png") != std::string::npos) {
         ProcessImage(sourceName, task, tritonClient, modelInfo, class_names);
    } else {
         ProcessVideo(sourceName, task, tritonClient, modelInfo, class_names);
    }

    return 0;
}
