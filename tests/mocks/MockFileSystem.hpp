#pragma once

#include <gmock/gmock.h>
#include "IFileSystem.hpp"

/**
 * Mock implementation of IFileSystem for unit testing
 */
class MockFileSystem : public IFileSystem {
public:
    MOCK_METHOD(std::vector<std::string>, readLines, (const std::string& filename), (const, override));
    MOCK_METHOD(bool, fileExists, (const std::string& filename), (const, override));
    MOCK_METHOD(cv::Mat, readImage, (const std::string& filename), (const, override));
    MOCK_METHOD(bool, writeImage, (const std::string& filename, const cv::Mat& image), (const, override));
    MOCK_METHOD(std::string, getEnvironmentVariable, (const std::string& name), (const, override));
};
