#pragma once

#include <vision-infra/core/FileSystem.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

/**
 * Interface for file system operations with CV-specific extensions.
 * Uses composition to leverage vision-infra's FileSystem functionality.
 */
class IFileSystem {
public:
    virtual ~IFileSystem() = default;
    
    // Direct access to vision-infra FileSystem for all standard operations
    virtual vision_infra::core::FileSystem& fs() = 0;
    virtual const vision_infra::core::FileSystem& fs() const = 0;
    
    // CV-specific operations
    virtual std::vector<std::string> readLines(const std::string& filename) const = 0;
    virtual cv::Mat readImage(const std::string& filename) const = 0;
    virtual bool writeImage(const std::string& filename, const cv::Mat& image) const = 0;
    virtual std::string getEnvironmentVariable(const std::string& name) const = 0;
    
    // Convenience wrapper for common operation
    bool fileExists(const std::string& filename) const {
        return fs().Exists(filename);
    }
};

/**
 * Implementation using vision-infra FileSystem + CV-specific extensions
 */
class FileSystem : public IFileSystem {
private:
    mutable vision_infra::core::FileSystem fs_;
    
public:
    // Direct access to vision-infra FileSystem
    vision_infra::core::FileSystem& fs() override { return fs_; }
    const vision_infra::core::FileSystem& fs() const override { return fs_; }
    
    // CV-specific operations
    std::vector<std::string> readLines(const std::string& filename) const override;
    cv::Mat readImage(const std::string& filename) const override;
    bool writeImage(const std::string& filename, const cv::Mat& image) const override;
    std::string getEnvironmentVariable(const std::string& name) const override;
};
