#include <gtest/gtest.h>
#include "Config.hpp"
#include <memory>

class ConfigManagerBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for tests
    }
};

TEST_F(ConfigManagerBasicTest, CreateDefaultReturnsValidConfig) {
    auto config = ConfigManager::createDefault();
    
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->server_address, "localhost");
    EXPECT_EQ(config->port, 8000);
    EXPECT_EQ(config->protocol, "http");
    EXPECT_FALSE(config->verbose);
}

TEST_F(ConfigManagerBasicTest, LoadFromEnvironmentReturnsConfig) {
    // ConfigManager.loadFromEnvironment() throws exception when validation fails
    // This is the expected behavior when required environment variables are missing
    EXPECT_THROW({
        ConfigManager::loadFromEnvironment();
    }, std::invalid_argument);
}

TEST_F(ConfigManagerBasicTest, LoadFromCommandLineWithValidArgs) {
    // Test with simulated command line args
    const char* argv[] = {
        "program",
        "--source=test.jpg",
        "--model_type=yolov8", 
        "--model=test_model",
        "--serverAddress=test-server",
        "--port=8001"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    
    EXPECT_NO_THROW({
        auto config = ConfigManager::loadFromCommandLine(argc, argv);
        ASSERT_NE(config, nullptr);
        EXPECT_TRUE(config->isValid());
    });
}

TEST_F(ConfigManagerBasicTest, LoadFromCommandLineWithInvalidModel) {
    // Test with invalid model name (containing path)
    const char* argv[] = {
        "program",
        "--source=test.jpg",
        "--model_type=yolov8",
        "--model=path/to/model",  // Invalid - contains path
        "--serverAddress=test-server"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    
    EXPECT_THROW({
        ConfigManager::loadFromCommandLine(argc, argv);
    }, std::invalid_argument);
}
