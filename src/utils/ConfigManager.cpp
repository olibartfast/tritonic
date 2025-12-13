#include "Config.hpp"
#include "utils.hpp"
#include "Logger.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <stdexcept>

std::unique_ptr<Config> ConfigManager::loadFromCommandLine(int argc, const char* argv[]) {
    auto config = std::make_unique<Config>();
    
    const cv::String keys =
        "{help h usage ? |      | show this help message}"
        "{source s       |      | path to input image/video file}"
        "{model_type mt  |      | type of model (yolov5, yolov8, etc.)}"
        "{model m        |      | model name on triton server}"
        "{labelsFile lf  |      | path to labels file}"
        "{protocol p     |http  | protocol to use (http or grpc)}"
        "{serverAddress sa |localhost | triton server address}"
        "{port pt        |8000  | triton server port}"
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
        "{log_file lf    |      | log file path}";

    cv::CommandLineParser parser(argc, argv, keys);
    
    if (parser.has("help")) {
        parser.printMessage();
        return nullptr;
    }

    // Parse command line arguments
    config->source = parser.get<cv::String>("source");
    config->model_type = parser.get<cv::String>("model_type");
    config->model_name = parser.get<cv::String>("model");
    config->labels_file = parser.get<cv::String>("labelsFile");
    config->protocol = parser.get<cv::String>("protocol");
    config->server_address = parser.get<cv::String>("serverAddress");
    config->port = parser.get<int>("port");
    config->batch_size = parser.get<int>("batch_size");
    config->show_frame = parser.get<bool>("show_frame");
    config->write_frame = parser.get<bool>("write_frame");
    config->confidence_threshold = parser.get<float>("confidence_threshold");
    config->nms_threshold = parser.get<float>("nms_threshold");
    config->verbose = parser.get<bool>("verbose");
    config->shared_memory_type = parser.get<cv::String>("shared_memory_type");
    config->cuda_device_id = parser.get<int>("cuda_device_id");
    config->log_level = parser.get<cv::String>("log_level");
    config->log_file = parser.get<cv::String>("log_file");

    // Parse input sizes if provided
    if (parser.has("input_sizes")) {
        std::string input_sizes_str = parser.get<cv::String>("input_sizes");
        config->input_sizes = parseInputSizes(input_sizes_str);
    }

    // Validate configuration
    if (!config->isValid()) {
        throw std::invalid_argument("Configuration validation failed: " + config->getValidationErrors());
    }

    return config;
}

std::unique_ptr<Config> ConfigManager::loadFromFile(const std::string& filename) {
    auto config = std::make_unique<Config>();
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        logger.errorf("Could not open config file: {}", filename);
        return nullptr;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Parse key-value pairs
        if (key == "server_address") config->server_address = value;
        else if (key == "port") config->port = std::stoi(value);
        else if (key == "protocol") config->protocol = value;
        else if (key == "model_name") config->model_name = value;
        else if (key == "model_type") config->model_type = value;
        else if (key == "source") config->source = value;
        else if (key == "labels_file") config->labels_file = value;
        else if (key == "batch_size") config->batch_size = std::stoi(value);
        else if (key == "show_frame") config->show_frame = (value == "true");
        else if (key == "write_frame") config->write_frame = (value == "true");
        else if (key == "confidence_threshold") config->confidence_threshold = std::stof(value);
        else if (key == "nms_threshold") config->nms_threshold = std::stof(value);
        else if (key == "verbose") config->verbose = (value == "true");
        else if (key == "shared_memory_type") config->shared_memory_type = value;
        else if (key == "cuda_device_id") config->cuda_device_id = std::stoi(value);
        else if (key == "log_level") config->log_level = value;
        else if (key == "log_file") config->log_file = value;
        else if (key == "input_sizes") config->input_sizes = parseInputSizes(value);
    }

    // Validate configuration
    if (!config->isValid()) {
        throw std::invalid_argument("Configuration validation failed: " + config->getValidationErrors());
    }

    return config;
}

std::unique_ptr<Config> ConfigManager::loadFromEnvironment() {
    auto config = std::make_unique<Config>();
    
    config->server_address = getEnvVar("TRITON_SERVER_ADDRESS", "localhost");
    config->port = std::stoi(getEnvVar("TRITON_SERVER_PORT", "8000"));
    config->protocol = getEnvVar("TRITON_PROTOCOL", "http");
    config->model_name = getEnvVar("TRITON_MODEL_NAME", "");
    config->model_type = getEnvVar("TRITON_MODEL_TYPE", "");
    config->source = getEnvVar("TRITON_SOURCE", "");
    config->labels_file = getEnvVar("TRITON_LABELS_FILE", "");
    config->batch_size = std::stoi(getEnvVar("TRITON_BATCH_SIZE", "1"));
    config->show_frame = (getEnvVar("TRITON_SHOW_FRAME", "false") == "true");
    config->write_frame = (getEnvVar("TRITON_WRITE_FRAME", "true") == "true");
    config->confidence_threshold = std::stof(getEnvVar("TRITON_CONFIDENCE_THRESHOLD", "0.5"));
    config->nms_threshold = std::stof(getEnvVar("TRITON_NMS_THRESHOLD", "0.4"));
    config->verbose = (getEnvVar("TRITON_VERBOSE", "false") == "true");
    config->shared_memory_type = getEnvVar("TRITON_SHARED_MEMORY_TYPE", "none");
    config->cuda_device_id = std::stoi(getEnvVar("TRITON_CUDA_DEVICE_ID", "0"));
    config->log_level = getEnvVar("TRITON_LOG_LEVEL", "info");
    config->log_file = getEnvVar("TRITON_LOG_FILE", "");
    
    std::string input_sizes_env = getEnvVar("TRITON_INPUT_SIZES", "");
    if (!input_sizes_env.empty()) {
        config->input_sizes = parseInputSizes(input_sizes_env);
    }

    // Validate configuration
    if (!config->isValid()) {
        throw std::invalid_argument("Configuration validation failed: " + config->getValidationErrors());
    }

    return config;
}

std::unique_ptr<Config> ConfigManager::createDefault() {
    return std::make_unique<Config>();
}

void ConfigManager::saveToFile(const Config& config, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    file << "# Triton Client Configuration\n";
    file << "server_address=" << config.server_address << "\n";
    file << "port=" << config.port << "\n";
    file << "protocol=" << config.protocol << "\n";
    file << "model_name=" << config.model_name << "\n";
    file << "model_type=" << config.model_type << "\n";
    file << "source=" << config.source << "\n";
    file << "labels_file=" << config.labels_file << "\n";
    file << "batch_size=" << config.batch_size << "\n";
    file << "show_frame=" << (config.show_frame ? "true" : "false") << "\n";
    file << "write_frame=" << (config.write_frame ? "true" : "false") << "\n";
    file << "confidence_threshold=" << config.confidence_threshold << "\n";
    file << "nms_threshold=" << config.nms_threshold << "\n";
    file << "verbose=" << (config.verbose ? "true" : "false") << "\n";
    file << "shared_memory_type=" << config.shared_memory_type << "\n";
    file << "cuda_device_id=" << config.cuda_device_id << "\n";
    file << "log_level=" << config.log_level << "\n";
    file << "log_file=" << config.log_file << "\n";
}

void ConfigManager::printConfig(const Config& config) {
    logger.info("Configuration:");
    logger.infof("  Server: {}:{} ({})", config.server_address, config.port, config.protocol);
    logger.infof("  Model: {} ({})", config.model_name, config.model_type);
    logger.infof("  Source: {}", config.source);
    logger.infof("  Labels: {}", config.labels_file);
    logger.infof("  Batch Size: {}", config.batch_size);
    logger.infof("  Show Frame: {}", config.show_frame ? "true" : "false");
    logger.infof("  Write Frame: {}", config.write_frame ? "true" : "false");
    logger.infof("  Confidence Threshold: {}", config.confidence_threshold);
    logger.infof("  NMS Threshold: {}", config.nms_threshold);
    logger.infof("  Verbose: {}", config.verbose ? "true" : "false");
    logger.infof("  Shared Memory Type: {}", config.shared_memory_type);
    if (config.shared_memory_type == "cuda") {
        logger.infof("  CUDA Device ID: {}", config.cuda_device_id);
    }
    logger.infof("  Log Level: {}", config.log_level);
    if (!config.log_file.empty()) {
        logger.infof("  Log File: {}", config.log_file);
    }
}

std::vector<std::vector<int64_t>> ConfigManager::parseInputSizes(const std::string& input) {
    std::vector<std::vector<int64_t>> sizes;
    std::vector<std::string> inputs = split(input, ';'); // Split by ';'

    for (const auto& input_str : inputs) {
        std::vector<std::string> dims = split(input_str, ','); // Split by ','
        std::vector<int64_t> dimensions;
        for (const auto& dim : dims) {
            dimensions.push_back(std::stoi(dim)); // Convert to int
        }
        sizes.push_back(dimensions);
    }

    return sizes;
}

std::string ConfigManager::getEnvVar(const std::string& name, const std::string& default_value) {
    const char* value = std::getenv(name.c_str());
    return value ? value : default_value;
} 