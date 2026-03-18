#include "ConfigManager.hpp"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>

ConfigManager::ConfigManager() = default;
ConfigManager::~ConfigManager() = default;

std::vector<std::vector<int64_t>> ConfigManager::ParseInputSizes(const std::string& input) {
    std::vector<std::vector<int64_t>> sizes;
    std::istringstream outer(input);
    std::string group;
    while (std::getline(outer, group, ';')) {
        std::vector<int64_t> dims;
        std::istringstream inner(group);
        std::string tok;
        while (std::getline(inner, tok, ',')) {
            // trim whitespace
            const char* ws = " \t\r\n";
            size_t s = tok.find_first_not_of(ws);
            size_t e = tok.find_last_not_of(ws);
            if (s != std::string::npos)
                dims.push_back(std::stoll(tok.substr(s, e - s + 1)));
        }
        if (!dims.empty())
            sizes.push_back(dims);
    }
    return sizes;
}

std::unique_ptr<InferenceConfig> ConfigManager::LoadFromCommandLine(int argc, const char* argv[]) {
    auto config = std::make_unique<InferenceConfig>();

    const cv::String keys =
        "{help h usage ? |      | show this help message}"
        "{source s       |      | path to input image/video file}"
        "{model_type mt  |      | type of model (yolov5, yolov8, etc.)}"
        "{model m        |      | model name on inference server}"
        "{labelsFile lf  |      | path to labels file}"
        "{protocol p     |http  | protocol to use (http or grpc)}"
        "{serverAddress sa |localhost | inference server address}"
        "{port pt        |8000  | inference server port}"
        "{input_sizes is |      | input sizes for dynamic axes (format: 'c,h,w;c,h,w')}"
        "{batch_size bs  |1     | batch size}"
        "{show_frame sf  |false | show processed frames}"
        "{write_frame wf |true  | write processed frames to disk}"
        "{confidence_threshold ct |0.5 | confidence threshold}"
        "{nms_threshold nt |0.4 | NMS threshold}"
        "{verbose v      |false | verbose output}"
        "{shared_memory_type smt |none | shared memory type (none, system, cuda)}"
        "{cuda_device_id cdi |0  | CUDA device ID for CUDA shared memory}"
        "{log_level ll   |info  | log level (debug, info, warn, error)}"
        "{log_file lf2   |      | log file path (use --log_file=path)}"
        "{enable_multimodal em |false | enable multimodal model support}"
        "{text_input ti  |      | path to text input file}"
        "{audio_input ai |      | path to audio input file}"
        "{text_prompt tp |      | text prompt for multimodal model}"
        "{modality_combination mc |concat | how to combine modalities}"
        "{text_weight tw |1.0   | weight for text modality}"
        "{image_weight iw |1.0  | weight for image modality}"
        "{audio_weight aw |1.0  | weight for audio modality}";

    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help")) {
        parser.printMessage();
        return nullptr;
    }

    config->SetSource(parser.get<cv::String>("source"));
    config->SetModelType(parser.get<cv::String>("model_type"));
    config->SetModelName(parser.get<cv::String>("model"));
    config->SetLabelsFile(parser.get<cv::String>("labelsFile"));
    config->SetProtocol(parser.get<cv::String>("protocol"));
    config->SetServerAddress(parser.get<cv::String>("serverAddress"));
    config->SetPort(parser.get<int>("port"));
    config->SetBatchSize(parser.get<int>("batch_size"));
    config->SetShowFrame(parser.get<bool>("show_frame"));
    config->SetWriteFrame(parser.get<bool>("write_frame"));
    config->SetConfidenceThreshold(parser.get<float>("confidence_threshold"));
    config->SetNmsThreshold(parser.get<float>("nms_threshold"));
    config->SetVerbose(parser.get<bool>("verbose"));
    config->SetSharedMemoryType(parser.get<cv::String>("shared_memory_type"));
    config->SetCudaDeviceId(parser.get<int>("cuda_device_id"));
    config->SetLogLevel(parser.get<cv::String>("log_level"));
    config->SetLogFile(parser.get<cv::String>("log_file"));
    config->SetEnableMultimodal(parser.get<bool>("enable_multimodal"));
    config->SetTextInput(parser.get<cv::String>("text_input"));
    config->SetAudioInput(parser.get<cv::String>("audio_input"));
    config->SetTextPrompt(parser.get<cv::String>("text_prompt"));
    config->SetModalityCombination(parser.get<cv::String>("modality_combination"));
    config->SetTextWeight(parser.get<float>("text_weight"));
    config->SetImageWeight(parser.get<float>("image_weight"));
    config->SetAudioWeight(parser.get<float>("audio_weight"));

    if (parser.has("input_sizes")) {
        std::string s = parser.get<cv::String>("input_sizes");
        config->SetInputSizes(ParseInputSizes(s));
    }

    return config;
}
