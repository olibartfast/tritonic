#pragma once

#include <gmock/gmock.h>
#include "ITriton.hpp"

/**
 * Mock implementation of ITriton for unit testing
 */
class MockTriton : public ITriton {
public:
    MOCK_METHOD(TritonModelInfo, getModelInfo, 
                (const std::string& modelName, const std::string& url, 
                 const std::vector<std::vector<int64_t>>& input_sizes), (override));
    
    MOCK_METHOD(std::vector<Tensor>, infer, (const std::vector<std::vector<uint8_t>>& input_data), (override));
    
    MOCK_METHOD(void, setInputShapes, (const std::vector<std::vector<int64_t>>& shapes), (override));
    
    MOCK_METHOD(void, setInputShape, (const std::vector<int64_t>& shape), (override));
    
    MOCK_METHOD(void, printModelInfo, (const TritonModelInfo& model_info), (const, override));
    
    MOCK_METHOD(void, createTritonClient, (), (override));
    
    // Shared memory methods from master branch
    MOCK_METHOD(void, registerInputSharedMemory, (), (override));
    
    MOCK_METHOD(void, registerOutputSharedMemory, (), (override));
    
    MOCK_METHOD(void, unregisterSharedMemory, (), (override));
    
    MOCK_METHOD(std::vector<Tensor>, inferWithSharedMemory, (const std::vector<std::vector<uint8_t>>& input_data), (override));

    MOCK_METHOD(bool, isServerLive, (), (override));
    MOCK_METHOD(bool, isModelInRepository, (const std::string& modelName), (override));
    MOCK_METHOD(bool, isModelReady, (const std::string& modelName, const std::string& modelVersion), (override));
    MOCK_METHOD(void, loadModel, (const std::string& modelName), (override));
    MOCK_METHOD(void, unloadModel, (const std::string& modelName), (override));
}; 