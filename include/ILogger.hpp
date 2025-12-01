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

    // Variadic template methods need to be in the header or implemented via a non-template interface
    // For simplicity in the interface, we can rely on the user formatting the string before calling log,
    // or we can keep the template methods here if they forward to a virtual method.
    // However, since we want to mock this, it's better to keep the virtual interface simple.
    // We can provide non-virtual template helpers in the base class or as free functions if needed.
    // But looking at existing Logger usage, it uses `infof`.
    
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
