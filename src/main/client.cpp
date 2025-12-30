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
        if (!config->GetLogFile().empty()) {
            logger.setLogFile(config->GetLogFile());
        }
        
        if (config->GetLogLevel() == "debug") logger.setLogLevel(LogLevel::DEBUG);
        else if (config->GetLogLevel() == "warn") logger.setLogLevel(LogLevel::WARN);
        else if (config->GetLogLevel() == "error") logger.setLogLevel(LogLevel::ERROR);
        else logger.setLogLevel(LogLevel::INFO);
        
        ConfigManager::printConfig(*config);
        
        // Create dependencies
        std::string url = config->GetServerAddress() + ":" + std::to_string(config->GetPort());
        ProtocolType protocol = config->GetProtocol() == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
        
        auto triton = std::make_shared<Triton>(url, protocol, config->GetModelName(), config->GetModelVersion(), config->GetVerbose());
        
        // Create and run App
        std::shared_ptr<ILogger> loggerPtr(&logger, [](ILogger*){});
        std::shared_ptr<Config> configPtr = std::move(config);
        
        App app(triton, configPtr, loggerPtr);
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
