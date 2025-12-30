#include "App.hpp"
#include <vision-infra/vision-infra.hpp>
#include "Triton.hpp"
#include <iostream>

int main(int argc, const char* argv[]) {
    try {
        // Initialize logging
        auto logger = std::dynamic_pointer_cast<vision_infra::core::Logger>(
            vision_infra::core::LoggerManager::GetLogger("tritonic"));
        logger->SetLevel(vision_infra::core::LogLevel::INFO);
        logger->EnableConsoleOutput(true);
        
        // Load configuration
        std::unique_ptr<vision_infra::config::InferenceConfig> config;
        try {
            vision_infra::config::ConfigManager configManager;
            config = configManager.LoadFromCommandLine(argc, argv);
        } catch (const std::invalid_argument& e) {
            logger->Error("Command line configuration error: " + std::string(e.what()));
            return 1;
        }
        
        if (!config) {
            return 0; // Help requested
        }
        
        // Configure logger
        if (!config->GetLogFile().empty()) {
            logger->SetOutputFile(config->GetLogFile());
        }
        
        if (config->GetLogLevel() == "debug") logger->SetLevel(vision_infra::core::LogLevel::DEBUG);
        else if (config->GetLogLevel() == "warn") logger->SetLevel(vision_infra::core::LogLevel::WARN);
        else if (config->GetLogLevel() == "error") logger->SetLevel(vision_infra::core::LogLevel::ERROR);
        else logger->SetLevel(vision_infra::core::LogLevel::INFO);
        
        // Configuration loaded successfully
        
        // Create dependencies
        std::string url = config->GetServerAddress() + ":" + std::to_string(config->GetPort());
        ProtocolType protocol = config->GetProtocol() == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;
        
        auto triton = std::make_shared<Triton>(url, protocol, config->GetModelName(), config->GetModelVersion(), config->GetVerbose());
        
        // Create and run App
        std::shared_ptr<vision_infra::config::InferenceConfig> configPtr = std::move(config);
        
        App app(triton, configPtr, logger);
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
