#pragma once

#include <memory>
#include "TaskInterface.hpp"
#include "TorchvisionClassifier.hpp"
#include "TensorflowClassifier.hpp"
#include "YOLOv10.hpp"
#include "YoloNas.hpp"
#include "YOLO.hpp"
#include "YOLOSeg.hpp"

class TaskFactory {
public:
    static std::unique_ptr<TaskInterface> createTaskInstance(const std::string& modelType, int input_width, int input_height, int channels = 3);
};