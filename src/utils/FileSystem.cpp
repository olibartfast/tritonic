#include "IFileSystem.hpp"
#include <fstream>
#include <filesystem>
#include <cstdlib>

std::vector<std::string> FileSystem::readLines(const std::string& filename) const {
    std::vector<std::string> lines;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    
    return lines;
}

// fileExists is now a non-virtual wrapper around Exists()

cv::Mat FileSystem::readImage(const std::string& filename) const {
    return cv::imread(filename);
}

bool FileSystem::writeImage(const std::string& filename, const cv::Mat& image) const {
    return cv::imwrite(filename, image);
}

std::string FileSystem::getEnvironmentVariable(const std::string& name) const {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : std::string();
}
