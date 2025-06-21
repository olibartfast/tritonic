#pragma once
#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <sstream>

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogLevel(LogLevel level);
    void setLogFile(const std::string& filename);
    void setConsoleOutput(bool enable);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    void fatal(const std::string& message);
    
    template<typename... Args>
    void debugf(const std::string& format, Args... args) {
        log(LogLevel::DEBUG, formatMessage(format, args...));
    }
    
    template<typename... Args>
    void infof(const std::string& format, Args... args) {
        log(LogLevel::INFO, formatMessage(format, args...));
    }
    
    template<typename... Args>
    void warnf(const std::string& format, Args... args) {
        log(LogLevel::WARN, formatMessage(format, args...));
    }
    
    template<typename... Args>
    void errorf(const std::string& format, Args... args) {
        log(LogLevel::ERROR, formatMessage(format, args...));
    }
    
    template<typename... Args>
    void fatalf(const std::string& format, Args... args) {
        log(LogLevel::FATAL, formatMessage(format, args...));
    }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void log(LogLevel level, const std::string& message);
    std::string getCurrentTimestamp();
    std::string levelToString(LogLevel level);
    std::string formatMessage(const std::string& format);
    
    template<typename T, typename... Args>
    std::string formatMessage(const std::string& format, T value, Args... args) {
        size_t pos = format.find("{}");
        if (pos == std::string::npos) {
            return format;
        }
        
        std::ostringstream oss;
        oss << value;
        std::string result = format.substr(0, pos) + oss.str() + format.substr(pos + 2);
        return formatMessage(result, args...);
    }
    
    LogLevel current_level_ = LogLevel::INFO;
    std::string log_file_;
    std::ofstream file_stream_;
    bool console_output_ = true;
    std::mutex log_mutex_;
}; 