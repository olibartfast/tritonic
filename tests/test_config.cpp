#include <gtest/gtest.h>
#include "Config.hpp"

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
    }
    
    void TearDown() override {
        // Clean up test data
    }
};

TEST_F(ConfigTest, ValidConfigurationPassesValidation) {
    Config config;
    config.server_address = "localhost";
    config.port = 8000;
    config.model_name = "test_model";
    config.model_type = "yolov8";
    config.source = "test.jpg";
    
    EXPECT_TRUE(config.isValid());
    EXPECT_TRUE(config.getValidationErrors().empty());
}

TEST_F(ConfigTest, InvalidConfigurationFailsValidation) {
    Config config; // All fields empty/default
    
    EXPECT_FALSE(config.isValid());
    EXPECT_FALSE(config.getValidationErrors().empty());
}

TEST_F(ConfigTest, ModelNameWithPathIsInvalid) {
    Config config;
    config.server_address = "localhost";
    config.port = 8000;
    config.model_name = "path/to/model";  // Contains path separator
    config.model_type = "yolov8";
    config.source = "test.jpg";
    
    EXPECT_FALSE(config.isValid());
    
    auto errors = config.getValidationErrors();
    bool foundPathError = false;
    for (const auto& error : errors) {
        if (error.find("path") != std::string::npos) {
            foundPathError = true;
            break;
        }
    }
    EXPECT_TRUE(foundPathError);
}

TEST_F(ConfigTest, ModelNameWithBackslashIsInvalid) {
    Config config;
    config.server_address = "localhost";
    config.port = 8000;
    config.model_name = "path\\to\\model";  // Contains backslash
    config.model_type = "yolov8";
    config.source = "test.jpg";
    
    EXPECT_FALSE(config.isValid());
    
    auto errors = config.getValidationErrors();
    bool foundPathError = false;
    for (const auto& error : errors) {
        if (error.find("path") != std::string::npos) {
            foundPathError = true;
            break;
        }
    }
    EXPECT_TRUE(foundPathError);
}

TEST_F(ConfigTest, InvalidPortFailsValidation) {
    Config config;
    config.server_address = "localhost";
    config.port = 0;  // Invalid port
    config.model_name = "test_model";
    config.model_type = "yolov8";
    config.source = "test.jpg";
    
    EXPECT_FALSE(config.isValid());
}

TEST_F(ConfigTest, EmptyModelNameFailsValidation) {
    Config config;
    config.server_address = "localhost";
    config.port = 8000;
    config.model_name = "";  // Empty model name
    config.model_type = "yolov8";
    config.source = "test.jpg";
    
    EXPECT_FALSE(config.isValid());
}

TEST_F(ConfigTest, EmptySourceFailsValidation) {
    Config config;
    config.server_address = "localhost";
    config.port = 8000;
    config.model_name = "test_model";
    config.model_type = "yolov8";
    config.source = "";  // Empty source
    
    EXPECT_FALSE(config.isValid());
} 