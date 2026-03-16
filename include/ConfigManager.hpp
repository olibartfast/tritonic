#pragma once

#include "Config.hpp"
#include <memory>
#include <vector>
#include <string>

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

