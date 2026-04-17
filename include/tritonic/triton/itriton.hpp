#pragma once

#include <memory>
#include <string>
#include <vector>
#include "tritonic/core/types.hpp"
#include "tritonic/triton/model_info.hpp"

namespace tritonic::triton {

/**
 * Interface for Triton inference client operations.
 * Abstracting this enables mocking in unit tests without a live server.
 */
class ITriton {
public:
    virtual ~ITriton() = default;

    virtual ModelInfo getModelInfo(
        const std::string& modelName, const std::string& url,
        const std::vector<std::vector<int64_t>>& input_sizes = {}) = 0;

    virtual std::vector<core::Tensor> infer(
        const std::vector<std::vector<uint8_t>>& input_data) = 0;

    virtual std::vector<core::Tensor> inferText(
        const std::vector<std::vector<std::string>>& string_inputs) = 0;

    virtual void setInputShapes(const std::vector<std::vector<int64_t>>& shapes) = 0;
    virtual void setInputShape(const std::vector<int64_t>& shape) = 0;
    virtual void printModelInfo(const ModelInfo& model_info) const = 0;
    virtual void createTritonClient() = 0;
    virtual void registerInputSharedMemory() = 0;
    virtual void registerOutputSharedMemory() = 0;
    virtual void unregisterSharedMemory() = 0;

    virtual std::vector<core::Tensor> inferWithSharedMemory(
        const std::vector<std::vector<uint8_t>>& input_data) = 0;

    virtual bool isServerLive() = 0;
    virtual bool isModelInRepository(const std::string& modelName) = 0;
    virtual bool isModelReady(const std::string& modelName,
                              const std::string& modelVersion = "") = 0;
    virtual void loadModel(const std::string& modelName) = 0;
    virtual void unloadModel(const std::string& modelName) = 0;
};

}  // namespace tritonic::triton
