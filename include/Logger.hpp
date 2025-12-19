#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <fmt/core.h>
#include "ILogger.hpp"

class Logger : public ILogger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogLevel(LogLevel level) override {
        std::lock_guard<std::mutex> lock(mutex_);
        currentLevel_ = level;
    }

    void setLogFile(const std::string& filename) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fileStream_.is_open()) {
            fileStream_.close();
        }
        fileStream_.open(filename, std::ios::app);
    }

    void setConsoleOutput(bool enable) override {
        std::lock_guard<std::mutex> lock(mutex_);
        consoleOutput_ = enable;
    }

    void log(LogLevel level, const std::string& message) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level < currentLevel_) return;

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count()
           << " [" << getLevelString(level) << "] " << message;

        std::string logMessage = ss.str();

        if (consoleOutput_) {
            std::cout << logMessage << std::endl;
        }

        if (fileStream_.is_open()) {
            fileStream_ << logMessage << std::endl;
        }
    }

    void debug(const std::string& message) override { log(LogLevel::DEBUG, message); }
    void info(const std::string& message) override { log(LogLevel::INFO, message); }
    void warn(const std::string& message) override { log(LogLevel::WARN, message); }
    void error(const std::string& message) override { log(LogLevel::ERROR, message); }
    void fatal(const std::string& message) override { log(LogLevel::FATAL, message); }

private:
    Logger() : currentLevel_(LogLevel::INFO), consoleOutput_(true) {}
    ~Logger() {
        if (fileStream_.is_open()) {
            fileStream_.close();
        }
    }
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }

    LogLevel currentLevel_;
    std::ofstream fileStream_;
    bool consoleOutput_;
    std::mutex mutex_;
};

// Global logger instance for backward compatibility (if needed)
// But we should prefer dependency injection.
// For now, we can keep the global instance but encourage using the interface.
extern Logger& logger;