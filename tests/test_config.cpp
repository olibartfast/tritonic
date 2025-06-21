#include "Config.hpp"
#include <iostream>
#include <cassert>

void testConfigValidation() {
    std::cout << "Testing configuration validation..." << std::endl;
    
    // Test valid configuration
    Config valid_config;
    valid_config.server_address = "localhost";
    valid_config.port = 8000;
    valid_config.model_name = "test_model";
    valid_config.model_type = "yolov8";
    valid_config.source = "test.jpg";
    
    assert(valid_config.isValid());
    assert(valid_config.getValidationErrors().empty());
    
    // Test invalid configuration
    Config invalid_config;
    assert(!invalid_config.isValid());
    assert(!invalid_config.getValidationErrors().empty());
    
    std::cout << "Configuration validation tests passed!" << std::endl;
}

void testConfigManager() {
    std::cout << "Testing ConfigManager..." << std::endl;
    
    // Test default configuration
    auto default_config = ConfigManager::createDefault();
    assert(default_config != nullptr);
    
    // Test environment variable loading
    auto env_config = ConfigManager::loadFromEnvironment();
    assert(env_config != nullptr);
    
    std::cout << "ConfigManager tests passed!" << std::endl;
}

void testLogger() {
    std::cout << "Testing Logger..." << std::endl;
    
    auto& logger = Logger::getInstance();
    
    // Test log level setting
    logger.setLogLevel(LogLevel::DEBUG);
    logger.setConsoleOutput(true);
    
    // Test logging
    logger.debug("Debug message");
    logger.info("Info message");
    logger.warn("Warning message");
    logger.error("Error message");
    
    // Test formatted logging
    logger.infof("Formatted message: {} with value: {}", "test", 42);
    
    std::cout << "Logger tests passed!" << std::endl;
}

int main() {
    try {
        testConfigValidation();
        testConfigManager();
        testLogger();
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 