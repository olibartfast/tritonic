#include "App.hpp"
#include <vision-infra/config/Config.hpp>
#include <vision-infra/core/Logger.hpp>
#include <vision-infra/config/ConfigManager.hpp>
#include "Triton.hpp"
#include <iostream>

int main(int argc, const char* argv[]) {
    try {
        // Initialize logging
        auto logger = vision_infra::Logger::getInstance();
        logger->setLogLevel(vision_infra::LogLevel::INFO);
        logger->setConsoleOutput(true);
        
        // Load configuration
        std::unique_ptr<vision_infra::Config> config;
        try {
            config = vision_infra::ConfigManager::loadFromCommandLine(argc, argv);
        } catch (const std::invalid_argument& e) {
            logger->error("Command line configuration error: " + std::string(e.what()));
            return 1;
        }
        
        if (!config) {
            return 0; // Help requested
        }
        
        // Configure logger
        if (!config->GetLogFile().empty()) {
            logger->setLogFile(config->GetLogFile());
        }
        
        if (config->GetLogLevel() == "debug") logger->setLogLevel(vision_infra::LogLevel::DEBUG);
        else if (config->GetLogLevel() == "warn") logger->setLogLevel(vision_infra::LogLevel::WARN);
        else if (config->GetLogLevel() == "error") logger->setLogLevel(vision_infra::LogLevel::ERROR);
        else logger->setLogLevel(vision_infra::LogLevel::INFO);
        
        vision_infra::ConfigManager::printConfig(*config);
        
        // Create dependencies
        std::string url = config->GetServerAddress() + ":" + std::to_string(config->GetPort());
        ProtocolType protocol = config->GetProtocol() == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
        
        auto triton = std::make_shared<Triton>(url, protocol, config->GetModelName(), config->GetModelVersion(), config->GetVerbose());
        
        // Create and run App
        std::shared_ptr<vision_infra::Config> configPtr = std::move(config);
        
        App app(triton, configPtr, logger);
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
