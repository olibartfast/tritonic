#pragma once

#include <string>
#include <memory>
#include <fmt/core.h>

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class ILogger {
public:
    virtual ~ILogger() = default;

    virtual void setLogLevel(LogLevel level) = 0;
    virtual void setLogFile(const std::string& filename) = 0;
    virtual void setConsoleOutput(bool enable) = 0;

    virtual void log(LogLevel level, const std::string& message) = 0;
    
    // Helper methods for convenience
    virtual void debug(const std::string& message) = 0;
    virtual void info(const std::string& message) = 0;
    virtual void warn(const std::string& message) = 0;
    virtual void error(const std::string& message) = 0;
    virtual void fatal(const std::string& message) = 0;

    template<typename... Args>
    void debugf(const char* format, const Args&... args) {
        debug(fmt::format(fmt::runtime(format), args...));
    }

    template<typename... Args>
    void infof(const char* format, const Args&... args) {
        info(fmt::format(fmt::runtime(format), args...));
    }

    template<typename... Args>
    void warnf(const char* format, const Args&... args) {
        warn(fmt::format(fmt::runtime(format), args...));
    }

    template<typename... Args>
    void errorf(const char* format, const Args&... args) {
        error(fmt::format(fmt::runtime(format), args...));
    }

    template<typename... Args>
    void fatalf(const char* format, const Args&... args) {
        fatal(fmt::format(fmt::runtime(format), args...));
    }
};
