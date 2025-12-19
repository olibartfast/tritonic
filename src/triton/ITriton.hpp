#pragma once

#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include "TritonModelInfo.hpp"

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * Interface for Triton inference client operations.
 * This defines the public interface that the application uses.
 */
class ITriton {
public:
    virtual ~ITriton() = default;
    
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
     * Set a single input shape
     */
    virtual void setInputShape(const std::vector<int64_t>& shape) = 0;
    
    /**
     * Print model information for debugging
     */
    virtual void printModelInfo(const TritonModelInfo& model_info) const = 0;
    
    /**
     * Create the underlying Triton client connection
     */
    virtual void createTritonClient() = 0;
    
    /**
     * Register input tensors with shared memory regions
     */
    virtual void registerInputSharedMemory() = 0;
    
    /**
     * Register output tensors with shared memory regions
     */
    virtual void registerOutputSharedMemory() = 0;
    
    /**
     * Unregister all shared memory regions
     */
    virtual void unregisterSharedMemory() = 0;
    
    /**
     * Perform inference using shared memory for inputs and outputs
     */
    virtual std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
            inferWithSharedMemory(const std::vector<std::vector<uint8_t>>& input_data) = 0;
}; 