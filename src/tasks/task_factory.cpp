#include "task_factory.hpp"

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(const std::string& modelType, int input_width, int input_height, int channels) {
    if (modelType == "torchvision-classifier") {
        return std::make_unique<TorchvisionClassifier>(input_width, input_height, channels);
    } else if (modelType == "tensorflow-classifier") {
        return std::make_unique<TensorflowClassifier>(input_width, input_height, channels);
    } else if (modelType.find("yolov10") != std::string::npos) {
        return std::make_unique<YOLOv10>(input_width, input_height);
    } else if (modelType.find("yolonas") != std::string::npos) {
        return std::make_unique<YoloNas>(input_width, input_height);
    } else if (modelType == "yoloseg") {
        return std::make_unique<YOLOSeg>(input_width, input_height);
    }else if (modelType.find("yolo") != std::string::npos) {
        return std::make_unique<YOLO>(input_width, input_height); 
    }
    else {
        return nullptr;
    }
}