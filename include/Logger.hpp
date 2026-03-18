#pragma once

#include <memory>
#include <sstream>
#include <string>

enum class LogLevel { TRACE, DEBUG, INFO, WARN, ERROR, FATAL };

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void Log(LogLevel level, const std::string& message) = 0;
    virtual void SetLevel(LogLevel level) = 0;
    virtual LogLevel GetLevel() const = 0;
    virtual void Flush() = 0;
};

class Logger : public ILogger {
public:
    explicit Logger(const std::string& name = "");
    ~Logger() override;

    void Log(LogLevel level, const std::string& message) override;
    void SetLevel(LogLevel level) override;
    LogLevel GetLevel() const override;
    void Flush() override;

    void SetOutputFile(const std::string& filename);
    void EnableConsoleOutput(bool enable = true);
    void EnableTimestamp(bool enable = true);
    void SetPattern(const std::string& pattern);

    void Trace(const std::string& message);
    void Debug(const std::string& message);
    void Info(const std::string& message);
    void Warn(const std::string& message);
    void Error(const std::string& message);
    void Fatal(const std::string& message);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

class LoggerManager {
public:
    static std::shared_ptr<ILogger> GetLogger(const std::string& name = "default");
    static void SetDefaultLogger(std::shared_ptr<ILogger> logger);
    static void SetGlobalLevel(LogLevel level);
    static LogLevel ParseLogLevel(const std::string& level);
    static std::string LogLevelToString(LogLevel level);
};
