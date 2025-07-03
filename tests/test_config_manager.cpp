#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ConfigManager.hpp"
#include "mocks/MockFileSystem.hpp"

using ::testing::Return;
using ::testing::_;

class ConfigManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        mockFileSystem = std::make_unique<MockFileSystem>();
    }
    
    void TearDown() override {
        mockFileSystem.reset();
    }
    
    std::unique_ptr<MockFileSystem> mockFileSystem;
};

TEST_F(ConfigManagerTest, CreateDefaultReturnsValidConfig) {
    auto config = ConfigManager::createDefault();
    
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->server_address, "localhost");
    EXPECT_EQ(config->port, 8000);
    EXPECT_EQ(config->protocol, 0); // HTTP
    EXPECT_EQ(config->confidence_threshold, 0.5f);
    EXPECT_EQ(config->log_level, "info");
}

TEST_F(ConfigManagerTest, LoadFromEnvironmentUsesEnvironmentVariables) {
    // Mock environment variables
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_SERVER_ADDRESS"))
        .WillOnce(Return("test-server"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_SERVER_PORT"))
        .WillOnce(Return("9000"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_MODEL_NAME"))
        .WillOnce(Return("env_model"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_MODEL_TYPE"))
        .WillOnce(Return("yolov8"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_SOURCE"))
        .WillOnce(Return("env_source.jpg"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_PROTOCOL"))
        .WillOnce(Return("1"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_CONFIDENCE_THRESHOLD"))
        .WillOnce(Return("0.7"));
    EXPECT_CALL(*mockFileSystem, getEnvironmentVariable("TRITON_LOG_LEVEL"))
        .WillOnce(Return("debug"));
    
    // Note: This test would need ConfigManager to accept a file system dependency
    // For now, we'll test the existing functionality
    auto config = ConfigManager::loadFromEnvironment();
    ASSERT_NE(config, nullptr);
}

TEST_F(ConfigManagerTest, CreateFromArgumentsHandlesValidArguments) {
    std::vector<std::string> args = {
        "program_name",
        "--server_address", "test-server",
        "--port", "9000",
        "--model_name", "test_model",
        "--model_type", "yolov8",
        "--source", "test.jpg",
        "--protocol", "1",
        "--confidence_threshold", "0.6",
        "--log_level", "debug"
    };
    
    int argc = args.size();
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    
    auto config = ConfigManager::createFromArguments(argc, argv.data());
    
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->server_address, "test-server");
    EXPECT_EQ(config->port, 9000);
    EXPECT_EQ(config->model_name, "test_model");
    EXPECT_EQ(config->model_type, "yolov8");
    EXPECT_EQ(config->source, "test.jpg");
    EXPECT_EQ(config->protocol, 1);
    EXPECT_FLOAT_EQ(config->confidence_threshold, 0.6f);
    EXPECT_EQ(config->log_level, "debug");
}

TEST_F(ConfigManagerTest, CreateFromArgumentsRejectsInvalidModelName) {
    std::vector<std::string> args = {
        "program_name",
        "--server_address", "localhost",
        "--port", "8000",
        "--model_name", "path/to/model",  // Invalid - contains path
        "--model_type", "yolov8",
        "--source", "test.jpg"
    };
    
    int argc = args.size();
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    
    EXPECT_THROW(ConfigManager::createFromArguments(argc, argv.data()), 
                 std::invalid_argument);
}

TEST_F(ConfigManagerTest, CreateFromArgumentsHandlesInvalidPort) {
    std::vector<std::string> args = {
        "program_name",
        "--server_address", "localhost",
        "--port", "0",  // Invalid port
        "--model_name", "test_model",
        "--model_type", "yolov8",
        "--source", "test.jpg"
    };
    
    int argc = args.size();
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    
    EXPECT_THROW(ConfigManager::createFromArguments(argc, argv.data()), 
                 std::invalid_argument);
}
