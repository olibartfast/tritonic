#include "Config.hpp"
#include "Logger.hpp"

// Static member definition
std::unique_ptr<vision_infra::config::ConfigManager> ConfigManager::manager_;

vision_infra::config::ConfigManager* ConfigManager::getManager() {
    if (!manager_) {
        manager_ = vision_infra::config::ConfigManager::Create();
    }
    return manager_.get();
}

std::unique_ptr<Config> ConfigManager::loadFromCommandLine(int argc, const char* argv[]) {
    return getManager()->LoadFromCommandLine(argc, argv);
}

std::unique_ptr<Config> ConfigManager::loadFromFile(const std::string& filename) {
    return getManager()->LoadFromFile(filename);
}

std::unique_ptr<Config> ConfigManager::loadFromEnvironment() {
    return getManager()->LoadFromEnvironment();
}

std::unique_ptr<Config> ConfigManager::createDefault() {
    return getManager()->CreateDefault();
}

void ConfigManager::printConfig(const Config& config) {
    getManager()->PrintConfig(config);
} 