#pragma once

#include "Config.hpp"
#include <memory>
#include <vector>
#include <string>

/**
 * Manager class for creating and loading configuration objects
 */
class ConfigManager {
public:
    /**
     * Load configuration from command line arguments
     */
    static std::unique_ptr<Config> loadFromCommandLine(int argc, const char* argv[]);
    
    /**
     * Create configuration from argument vector (for testing)
     */
    static std::unique_ptr<Config> createFromArguments(const std::vector<std::string>& args);
    
    /**
     * Load configuration from environment variables
     */
    static std::unique_ptr<Config> loadFromEnvironment();
    
    /**
     * Create default configuration
     */
    static std::unique_ptr<Config> createDefault();
    
    /**
     * Load configuration from file
     */
    static std::unique_ptr<Config> loadFromFile(const std::string& filename);
};
