#pragma once

#include <vision-infra/config/ConfigManager.hpp>
#include <vision-infra/config/Config.hpp>
#include <memory>

// Type alias to use vision-infra's InferenceConfig
using Config = vision_infra::config::InferenceConfig;

/**
 * Adapter class for vision-infra ConfigManager with convenience methods
 */
class ConfigManager {
public:
    static std::unique_ptr<Config> loadFromCommandLine(int argc, const char* argv[]);
    static std::unique_ptr<Config> loadFromFile(const std::string& filename);
    static std::unique_ptr<Config> loadFromEnvironment();
    static std::unique_ptr<Config> createDefault();
    
    static void printConfig(const Config& config);

private:
    static std::unique_ptr<vision_infra::config::ConfigManager> manager_;
    static vision_infra::config::ConfigManager* getManager();
}; 