#pragma once

#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include "TritonModelInfo.hpp"

using TensorElement = std::variant<float, int32_t, int64_t>;

/**
 * Interface for Triton inference client operations.
 * This allows for dependency injection and mocking in tests.
 */
class ITritonClient {
public:
    virtual ~ITritonClient() = default;
    
    /**
     * Get model information from the inference server
     */
    virtual TritonModelInfo getModelInfo(const std::string& modelName, 
                                       const std::string& url, 
                                       const std::vector<std::vector<int64_t>>& input_sizes = {}) = 0;
    
    /**
     * Perform inference with the given input data
     */
    virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
            infer(const std::vector<std::vector<uint8_t>>& input_data) = 0;
    
    /**
     * Set input shapes for dynamic shape models
     */
    virtual void setInputShapes(const std::vector<std::vector<int64_t>>& shapes) = 0;
    
    /**
     * Print model information for debugging
     */
    virtual void printModelInfo(const TritonModelInfo& model_info) const = 0;
};

/**
 * Factory interface for creating Triton clients
 * This enables dependency injection of client creation
 */
class ITritonClientFactory {
public:
    virtual ~ITritonClientFactory() = default;
    
    virtual std::unique_ptr<ITritonClient> createClient(
        const std::string& url,
        int protocol,  // 0=HTTP, 1=GRPC
        const std::string& modelName,
        const std::string& modelVersion = "",
        bool verbose = false) = 0;
};
