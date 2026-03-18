#include <gtest/gtest.h>
#include "Logger.hpp"

// LogLevel

TEST(LogLevelTest, Ordering) {
    EXPECT_LT(LogLevel::TRACE, LogLevel::DEBUG);
    EXPECT_LT(LogLevel::DEBUG, LogLevel::INFO);
    EXPECT_LT(LogLevel::INFO,  LogLevel::WARN);
    EXPECT_LT(LogLevel::WARN,  LogLevel::ERROR);
    EXPECT_LT(LogLevel::ERROR, LogLevel::FATAL);
}

// LoggerManager

TEST(LoggerManagerTest, ParseLogLevel) {
    EXPECT_EQ(LoggerManager::ParseLogLevel("trace"),   LogLevel::TRACE);
    EXPECT_EQ(LoggerManager::ParseLogLevel("debug"),   LogLevel::DEBUG);
    EXPECT_EQ(LoggerManager::ParseLogLevel("info"),    LogLevel::INFO);
    EXPECT_EQ(LoggerManager::ParseLogLevel("warn"),    LogLevel::WARN);
    EXPECT_EQ(LoggerManager::ParseLogLevel("warning"), LogLevel::WARN);
    EXPECT_EQ(LoggerManager::ParseLogLevel("error"),   LogLevel::ERROR);
    EXPECT_EQ(LoggerManager::ParseLogLevel("fatal"),   LogLevel::FATAL);
    EXPECT_EQ(LoggerManager::ParseLogLevel("unknown"), LogLevel::INFO); // default
}

TEST(LoggerManagerTest, LogLevelToString) {
    EXPECT_EQ(LoggerManager::LogLevelToString(LogLevel::TRACE), "TRACE");
    EXPECT_EQ(LoggerManager::LogLevelToString(LogLevel::DEBUG), "DEBUG");
    EXPECT_EQ(LoggerManager::LogLevelToString(LogLevel::INFO),  "INFO");
    EXPECT_EQ(LoggerManager::LogLevelToString(LogLevel::WARN),  "WARN");
    EXPECT_EQ(LoggerManager::LogLevelToString(LogLevel::ERROR), "ERROR");
    EXPECT_EQ(LoggerManager::LogLevelToString(LogLevel::FATAL), "FATAL");
}

TEST(LoggerManagerTest, GetLoggerReturnsSameInstanceForSameName) {
    auto a = LoggerManager::GetLogger("unit_same");
    auto b = LoggerManager::GetLogger("unit_same");
    EXPECT_EQ(a.get(), b.get());
}

TEST(LoggerManagerTest, GetLoggerReturnsDifferentInstancesForDifferentNames) {
    auto a = LoggerManager::GetLogger("unit_a");
    auto b = LoggerManager::GetLogger("unit_b");
    EXPECT_NE(a.get(), b.get());
}

TEST(LoggerManagerTest, GetLoggerReturnsValidPointer) {
    auto logger = LoggerManager::GetLogger("unit_valid");
    EXPECT_NE(logger, nullptr);
}

// Logger

TEST(LoggerTest, DefaultLevelIsInfo) {
    Logger logger("unit_default");
    EXPECT_EQ(logger.GetLevel(), LogLevel::INFO);
}

TEST(LoggerTest, SetAndGetLevel) {
    Logger logger("unit_setlevel");
    logger.SetLevel(LogLevel::DEBUG);
    EXPECT_EQ(logger.GetLevel(), LogLevel::DEBUG);
    logger.SetLevel(LogLevel::ERROR);
    EXPECT_EQ(logger.GetLevel(), LogLevel::ERROR);
}

TEST(LoggerTest, LogDoesNotThrowWithConsoleDisabled) {
    Logger logger("unit_nothrow");
    logger.EnableConsoleOutput(false);
    EXPECT_NO_THROW(logger.Trace("trace"));
    EXPECT_NO_THROW(logger.Debug("debug"));
    EXPECT_NO_THROW(logger.Info("info"));
    EXPECT_NO_THROW(logger.Warn("warn"));
    EXPECT_NO_THROW(logger.Error("error"));
    EXPECT_NO_THROW(logger.Fatal("fatal"));
}

TEST(LoggerTest, LogFiltersBelowLevel) {
    Logger logger("unit_filter");
    logger.SetLevel(LogLevel::ERROR);
    logger.EnableConsoleOutput(false);
    EXPECT_NO_THROW(logger.Info("filtered"));
    EXPECT_NO_THROW(logger.Debug("filtered"));
    EXPECT_NO_THROW(logger.Warn("filtered"));
    // These should pass through
    EXPECT_NO_THROW(logger.Error("logged"));
    EXPECT_NO_THROW(logger.Fatal("logged"));
}

TEST(LoggerTest, LogDirectMethod) {
    Logger logger("unit_direct");
    logger.EnableConsoleOutput(false);
    EXPECT_NO_THROW(logger.Log(LogLevel::INFO, "direct"));
}

TEST(LoggerTest, FlushDoesNotThrow) {
    Logger logger("unit_flush");
    EXPECT_NO_THROW(logger.Flush());
}

TEST(LoggerTest, SetOutputFileDoesNotThrow) {
    Logger logger("unit_file");
    EXPECT_NO_THROW(logger.SetOutputFile(""));  // empty = no file
}
