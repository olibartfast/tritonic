#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * Interface for file system operations.
 * This allows for dependency injection and mocking in tests.
 */
class IFileSystem {
public:
    virtual ~IFileSystem() = default;
    
    /**
     * Read lines from a text file
     */
    virtual std::vector<std::string> readLines(const std::string& filename) const = 0;
    
    /**
     * Check if a file exists
     */
    virtual bool fileExists(const std::string& filename) const = 0;
    
    /**
     * Read an image file
     */
    virtual cv::Mat readImage(const std::string& filename) const = 0;
    
    /**
     * Write an image file
     */
    virtual bool writeImage(const std::string& filename, const cv::Mat& image) const = 0;
    
    /**
     * Read environment variable
     */
    virtual std::string getEnvironmentVariable(const std::string& name) const = 0;
};

/**
 * Real implementation of the file system interface
 */
class FileSystem : public IFileSystem {
public:
    std::vector<std::string> readLines(const std::string& filename) const override;
    bool fileExists(const std::string& filename) const override;
    cv::Mat readImage(const std::string& filename) const override;
    bool writeImage(const std::string& filename, const cv::Mat& image) const override;
    std::string getEnvironmentVariable(const std::string& name) const override;
};
