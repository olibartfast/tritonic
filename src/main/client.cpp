#include "App.hpp"
#include "Config.hpp"
#include "Logger.hpp"
#include "Triton.hpp"
#include <iostream>

int main(int argc, const char* argv[]) {
    try {
        // Initialize logging
        logger.setLogLevel(LogLevel::INFO);
        logger.setConsoleOutput(true);
        
        // Load configuration
        std::unique_ptr<Config> config;
        try {
            config = ConfigManager::loadFromCommandLine(argc, argv);
        } catch (const std::invalid_argument& e) {
            logger.error("Command line configuration error: " + std::string(e.what()));
            return 1;
        }
        
        if (!config) {
            return 0; // Help requested
        }
        
        // Configure logger
        if (!config->log_file.empty()) {
            logger.setLogFile(config->log_file);
        }
        
        if (config->log_level == "debug") logger.setLogLevel(LogLevel::DEBUG);
        else if (config->log_level == "warn") logger.setLogLevel(LogLevel::WARN);
        else if (config->log_level == "error") logger.setLogLevel(LogLevel::ERROR);
        else logger.setLogLevel(LogLevel::INFO);
        
        ConfigManager::printConfig(*config);
        
        // Create dependencies
        std::string url = config->server_address + ":" + std::to_string(config->port);
        ProtocolType protocol = config->protocol == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
        
        auto triton = std::make_shared<Triton>(url, protocol, config->model_name, config->model_version, config->verbose);
        
        // Create and run App
        // We need to wrap the global logger in a shared_ptr wrapper that doesn't delete it
        // Or better, we should have made Logger a shared_ptr from the start.
        // For now, let's create a proxy or just pass the address if we change App to take raw pointer?
        // No, App takes shared_ptr. Let's create a custom deleter.
        std::shared_ptr<ILogger> loggerPtr(&logger, [](ILogger*){});
        
        // We need to convert unique_ptr<Config> to shared_ptr<Config>
        std::shared_ptr<Config> configPtr = std::move(config);
        
        App app(triton, configPtr, loggerPtr);
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
