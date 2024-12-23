#include "task_factory.hpp"
#include "DFine.hpp"
#include "RTDetr.hpp"

// Define the map of task creators
std::map<std::string, TaskFactory::TaskCreator> TaskFactory::taskCreators = {
    // {"torchvision-classifier", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<TorchvisionClassifier>(sizes); }},
    // {"tensorflow-classifier", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<TensorflowClassifier>(sizes); }},
    // {"yolov10", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<YOLOv10>(sizes); }},
    // {"yolonas", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<YoloNas>(sizes); }},
    // {"yoloseg", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<YOLOSeg>(sizes); }},
    // {"yolo", [](const std::vector<std::vector<int64_t>>& sizes) { return std::make_unique<YOLO>(sizes); }},
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