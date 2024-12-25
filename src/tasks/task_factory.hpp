#pragma once

#include <memory>
#include "TaskInterface.hpp"
#include <functional>
#include <map>
#include <string>
#include <vector>

class TaskFactory {
private:
    using TaskCreator = std::function<std::unique_ptr<TaskInterface>(const TritonModelInfo& modelInfo)>;
    
    static std::map<std::string, TaskCreator> taskCreators;

    static void validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes);

public:
    static std::unique_ptr<TaskInterface> createTaskInstance(const std::string& modelType, const TritonModelInfo& modelInfo);
};