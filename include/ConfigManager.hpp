#pragma once

#include <memory>
#include <string>
#include <vector>
#include "Config.hpp"

class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager();

    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    ConfigManager(ConfigManager&&) noexcept = default;
    ConfigManager& operator=(ConfigManager&&) noexcept = default;

    std::unique_ptr<InferenceConfig> LoadFromCommandLine(int argc, const char* argv[]);

private:
    static std::vector<std::vector<int64_t>> ParseInputSizes(const std::string& input);
};
