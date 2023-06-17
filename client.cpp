#include "Yolo.hpp"
#include "YoloNas.hpp"
#include "Triton.hpp"

std::unique_ptr<YoloInterface> createYoloInstance(const std::string& modelType, const int input_width, const int input_height)
{
    if (modelType == "yolonas")
    {
        return std::make_unique<YoloNas>(input_width, input_height);
    }
    else if (modelType.find("yolov") != std::string::npos )
    {
        return std::make_unique<Yolo>(input_width, input_height);
    }
    else
    {
        return nullptr;
    }
}

// Function to parse input size string and validate the format
std::pair<int, int> parseInputSize(const std::string& input_size_str) {
    std::istringstream iss(input_size_str);
    std::vector<std::string> input_size_values;
    std::string token;
    while (std::getline(iss, token, ',')) {
        input_size_values.push_back(token);
    }

    if (input_size_values.size() != 2) {
        throw std::runtime_error("Invalid input size format. Please provide width and height values separated by a comma.");
    }

    const int input_width = std::stoi(input_size_values[0]);
    const int input_height = std::stoi(input_size_values[1]);

    return std::make_pair(input_width, input_height);
}


static const std::string keys =
    "{ help h   | | Print help message. }"
    "{ model_type t | yolov7 | yolo version used i.e yolov5, yolov6 or yolov7}"
    "{ model m | yolov7-tiny_onnx | model name of folder in triton }"
    "{ video v | video.mp4 | video name}"
    "{ serverAddress  s  | localhost:8000 | Path to server address}"
    "{ verbose vb | false | Verbose mode, true or false}"
    "{ protocol p | http | Protocol type, grpc or http}"
    "{ labelsFile l | ../coco.names | path to  coco labels names}"
    "{ batch b | 1 | Batch size}"
    "{ inputSize i | 640,640 | Input size of the model}";

int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    std::string serverAddress = parser.get<std::string>("serverAddress");
    bool verbose = parser.get<bool>("verbose");
    std::string videoName = parser.get<std::string>("video");
    Triton::ProtocolType protocol = parser.get<std::string>("protocol") == "grpc" ? Triton::ProtocolType::GRPC : Triton::ProtocolType::HTTP;
    const size_t batch_size = parser.get<size_t>("batch");

    std::string modelName = parser.get<std::string>("model");
    std::string modelVersion = "";
    std::string modelType = parser.get<std::string>("model_type");
    std::string url(serverAddress);
    std::string labelsFile = parser.get<std::string>("labelsFile");
    auto [input_width, input_height] = parseInputSize( parser.get<std::string>("inputSize"));

    std::cout << "Chosen Parameters:" << std::endl;
    std::cout << "model_type (t): " << parser.get<std::string>("model_type") << std::endl;
    std::cout << "model (m): " << parser.get<std::string>("model") << std::endl;
    std::cout << "video (v): " << parser.get<std::string>("video") << std::endl;
    std::cout << "serverAddress (s): " << parser.get<std::string>("serverAddress") << std::endl;
    std::cout << "verbose (vb): " << parser.get<bool>("verbose") << std::endl;
    std::cout << "protocol (p): " << parser.get<std::string>("protocol") << std::endl;
    std::cout << "labelsFile (l): " << parser.get<std::string>("labelsFile") << std::endl;
    std::cout << "batch (b): " << parser.get<size_t>("batch") << std::endl;
    std::cout << "inputSize (i): " << parser.get<std::string>("inputSize") << std::endl;

    // Create Triton client
    Triton::TritonClient tritonClient;
    Triton::createTritonClient(tritonClient, url, verbose, protocol);

    Triton::TritonModelInfo yoloModelInfo = Triton::setModel(batch_size, input_width, input_height, modelType);
    std::unique_ptr<YoloInterface>  yolo = createYoloInstance(modelType, input_width, input_height);
    const auto coco_names = yolo->readLabelNames(labelsFile);

    std::vector<tc::InferInput*> inputs = { nullptr };
    std::vector<const tc::InferRequestedOutput*> outputs = Triton::createInferRequestedOutput(yoloModelInfo.output_names_);
    tc::InferOptions options = Triton::createInferOptions(modelName, modelVersion);

    cv::Mat frame;
    std::vector<uint8_t> input_data;
    cv::VideoCapture cap(videoName);

#ifdef WRITE_FRAME
    cv::VideoWriter outputVideo;
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),    // Acquire input size
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    outputVideo.open("processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened()) {
        std::cout << "Could not open the output video for write: " << videoName << std::endl;
        return -1;
    }
#endif

    tc::Error err;
    while (cap.read(frame)) {
        auto start = std::chrono::steady_clock::now();
        input_data = yolo->preprocess(frame, yoloModelInfo.input_format_, yoloModelInfo.type1_, yoloModelInfo.type3_,
            yoloModelInfo.input_c_, cv::Size(yoloModelInfo.input_w_, yoloModelInfo.input_h_));

        if (inputs[0] != nullptr) {
            err = inputs[0]->Reset();
            if (!err.IsOk()) {
                std::cerr << "failed resetting input: " << err << std::endl;
                exit(1);
            }
        }
        else {
            err = tc::InferInput::Create(
                &inputs[0], yoloModelInfo.input_name_, yoloModelInfo.shape_, yoloModelInfo.input_datatype_);
            if (!err.IsOk()) {
                std::cerr << "unable to get input: " << err << std::endl;
                exit(1);
            }
        }

        err = inputs[0]->AppendRaw(input_data);
        if (!err.IsOk()) {
            std::cerr << "failed setting input: " << err << std::endl;
            exit(1);
        }

        tc::InferResult* result;
        std::unique_ptr<tc::InferResult> result_ptr;
        if (protocol == Triton::ProtocolType::HTTP) {
            err = tritonClient.httpClient->Infer(
                &result, options, inputs, outputs);
        }
        else {
            err = tritonClient.grpcClient->Infer(
                &result, options, inputs, outputs);
        }
        if (!err.IsOk()) {
            std::cerr << "failed sending synchronous infer request: " << err
                << std::endl;
            exit(1);
        }

        auto [infer_results, infer_shapes] = Triton::getInferResults(result, batch_size, yoloModelInfo.output_names_, yoloModelInfo.max_batch_size_ != 0);
        result_ptr.reset(result);
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Infer time: " << diff << " ms" << std::endl;

        std::vector<Yolo::Detection> detections = yolo->postprocess(cv::Size(frame.cols, frame.rows),
            infer_results, infer_shapes);

#if defined(SHOW_FRAME) || defined(WRITE_FRAME)
        double fps = 1000.0 / static_cast<double>(diff);
        std::string fpsText = "FPS: " + std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (auto&& detection : detections) {
            cv::rectangle(frame, detection.bbox, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, coco_names[detection.class_id],
                cv::Point(detection.bbox.x, detection.bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
#endif
#ifdef SHOW_FRAME
        cv::imshow("video feed ", frame);
        cv::waitKey(1);
#endif
#ifdef WRITE_FRAME
        outputVideo.write(frame);
#endif
    }

    return 0;
}
