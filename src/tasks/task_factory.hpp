#pragma once

#include <memory>
#include "TaskInterface.hpp"
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

class TaskFactory {
private:
    using TaskCreator = std::function<std::unique_ptr<TaskInterface>(const TritonModelInfo& modelInfo)>;
    
    static std::map<std::string, TaskCreator> taskCreators;

    static bool icontains(const std::string& str, const std::string& substr) {
        return std::search(
            str.begin(), str.end(),
            substr.begin(), substr.end(),
            [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
        ) != str.end();
    }



    static void validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes) {
        if (input_sizes.empty()) {
            throw std::invalid_argument("Input sizes vector is empty");
        }
        for (const auto& size : input_sizes) {
            if (size.empty()) {
                throw std::invalid_argument("An input size vector is empty");
            }
            if (std::any_of(size.begin(), size.end(), [](int64_t s) { return s < 0; })) {
                throw std::invalid_argument("Negative input size detected");
            }
        }
    }

public:
    static std::unique_ptr<TaskInterface> createTaskInstance(const std::string& modelType, const TritonModelInfo& modelInfo);
};