#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>

struct Config {
    // Server configuration
    std::string server_address = "localhost";
    int port = 8000;
    std::string protocol = "http";  // "http" or "grpc"
    bool verbose = false;
    
    // Model configuration
    std::string model_name;
    std::string model_version = "";
    std::string model_type;
    std::vector<std::vector<int64_t>> input_sizes;
    
    // Input/Output configuration
    std::string source;
    std::string labels_file;
    int batch_size = 1;
    
    // Processing configuration
    bool show_frame = false;
    bool write_frame = true;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    
    // Performance configuration
    int num_threads = 1;
    bool enable_async = false;
    
    // Shared memory configuration
    std::string shared_memory_type = "none";  // "none", "system", or "cuda"
    int cuda_device_id = 0;
    
    // Logging configuration
    std::string log_level = "info";
    std::string log_file = "";
    
    // Validation methods
    bool isValid() const {
        return !server_address.empty() && 
               !model_name.empty() && 
               !model_type.empty() && 
               !source.empty() &&
               port > 0 && port <= 65535 &&
               !isModelNameAPath();
    }
    
    bool isModelNameAPath() const {
        return model_name.find('/') != std::string::npos || 
               model_name.find('\\') != std::string::npos;
    }
    
    std::string getValidationErrors() const {
        std::vector<std::string> errors;
        
        if (server_address.empty()) errors.push_back("Server address is required");
        if (model_name.empty()) errors.push_back("Model name is required");
        if (isModelNameAPath()) errors.push_back("Model name must not contain path separators (/ or \\). Use only the model repository name.");
        if (model_type.empty()) errors.push_back("Model type is required");
        if (source.empty()) errors.push_back("Source is required");
        if (port <= 0 || port > 65535) errors.push_back("Port must be between 1 and 65535");
        if (protocol != "http" && protocol != "grpc") errors.push_back("Protocol must be 'http' or 'grpc'");
        
        std::string result;
        for (const auto& error : errors) {
            if (!result.empty()) result += "; ";
            result += error;
        }
        return result;
    }
};

class ConfigManager {
public:
    static std::unique_ptr<Config> loadFromCommandLine(int argc, const char* argv[]);
    static std::unique_ptr<Config> loadFromFile(const std::string& filename);
    static std::unique_ptr<Config> loadFromEnvironment();
    static std::unique_ptr<Config> createDefault();
    
    static void saveToFile(const Config& config, const std::string& filename);
    static void printConfig(const Config& config);
    
private:
    static std::vector<std::vector<int64_t>> parseInputSizes(const std::string& input);
    static std::string getEnvVar(const std::string& name, const std::string& default_value = "");
}; 