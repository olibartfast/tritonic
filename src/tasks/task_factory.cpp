#include "task_factory.hpp"
#include "DFine.hpp"
#include "RTDetr.hpp"
#include "YOLO.hpp"
#include "YoloNas.hpp"
#include "YOLOv10.hpp"
// #include "TorchvisionClassifier.hpp"
// #include "TensorflowClassifier.hpp"
// #include "YOLOSeg.hpp"

// Define the map of task creators
std::map<std::string, TaskFactory::TaskCreator> TaskFactory::taskCreators = {
    // {"torchvision-classifier", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<TorchvisionClassifier>(sizes); }},
    // {"tensorflow-classifier", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<TensorflowClassifier>(sizes); }},
    // {"yoloseg", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<YOLOSeg>(sizes); }},
    {"yolonas", [](const TritonModelInfo& modelInfo) { return std::make_unique<YoloNas>(modelInfo); }},
    {"yolov5", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov6", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov7", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov8", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov9", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov10", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLOv10>(modelInfo); }},
    {"rtdetr", [](const TritonModelInfo& modelInfo) { return std::make_unique<RTDetr>(modelInfo); }},
    {"dfine", [](const TritonModelInfo& modelInfo) { return std::make_unique<DFine>(modelInfo); }}
};

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(const std::string& modelType, const TritonModelInfo& modelInfo) {
    try 
    {
        const auto& input_sizes = modelInfo.input_shapes;
        validateInputSizes(input_sizes);

        for (const auto& [key, creator] : taskCreators) {
            if (icontains(modelType, key)) {
                return creator(modelInfo);
            }
        }

        throw std::runtime_error("Unrecognized model type: " + modelType);
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating task instance: " << e.what() << std::endl;
        return nullptr;
    }
}