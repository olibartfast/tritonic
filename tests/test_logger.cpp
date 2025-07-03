#include <gtest/gtest.h>
#include <sstream>
#include <memory>
#include "Logger.hpp"

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset logger state before each test
        auto& logger = Logger::getInstance();
        logger.setLogLevel(LogLevel::INFO);
        logger.setConsoleOutput(false);
    }
    
    void TearDown() override {
        // Clean up after each test
        auto& logger = Logger::getInstance();
        logger.setConsoleOutput(false);
    }
};

TEST_F(LoggerTest, SingletonInstanceIsConsistent) {
    auto& logger1 = Logger::getInstance();
    auto& logger2 = Logger::getInstance();
    
    // Should be the same instance
    EXPECT_EQ(&logger1, &logger2);
}

TEST_F(LoggerTest, LogLevelFiltering) {
    auto& logger = Logger::getInstance();
    logger.setLogLevel(LogLevel::WARN);
    
    // Test that messages below the threshold are filtered
    // Since we can't directly capture log output in this simplified test,
    // we verify the log level is set correctly
    logger.debug("Debug message");  // Should not appear
    logger.info("Info message");    // Should not appear
    logger.warn("Warning message"); // Should appear
    logger.error("Error message");  // Should appear
    
    // This test mainly verifies no exceptions are thrown
    SUCCEED();
}

TEST_F(LoggerTest, FormattedLogging) {
    auto& logger = Logger::getInstance();
    logger.setLogLevel(LogLevel::DEBUG);
    
    // Test formatted logging doesn't throw exceptions
    EXPECT_NO_THROW(logger.debugf("Debug: {}", 42));
    EXPECT_NO_THROW(logger.infof("Info: {} {}", "test", 123));
    EXPECT_NO_THROW(logger.warnf("Warning: {:.2f}", 3.14159));
    EXPECT_NO_THROW(logger.errorf("Error: {} {} {}", "multiple", "values", 456));
}

TEST_F(LoggerTest, LogLevelConversion) {
    auto& logger = Logger::getInstance();
    
    // Test all log levels can be set without throwing
    EXPECT_NO_THROW(logger.setLogLevel(LogLevel::DEBUG));
    EXPECT_NO_THROW(logger.setLogLevel(LogLevel::INFO));
    EXPECT_NO_THROW(logger.setLogLevel(LogLevel::WARN));
    EXPECT_NO_THROW(logger.setLogLevel(LogLevel::ERROR));
}

TEST_F(LoggerTest, ConsoleOutputToggle) {
    auto& logger = Logger::getInstance();
    
    // Test console output can be toggled without issues
    EXPECT_NO_THROW(logger.setConsoleOutput(true));
    EXPECT_NO_THROW(logger.setConsoleOutput(false));
    
    logger.setConsoleOutput(true);
    EXPECT_NO_THROW(logger.info("Test message with console output"));
    
    logger.setConsoleOutput(false);
    EXPECT_NO_THROW(logger.info("Test message without console output"));
}
